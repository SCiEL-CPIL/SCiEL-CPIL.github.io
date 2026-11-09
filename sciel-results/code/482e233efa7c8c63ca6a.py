import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        cycles.append(tuple(e1))
        cycles.append(tuple(e2))
        cycles.append(tuple(e3))
    return cycles

def build_edge_cycle_map(cycles):
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    cycles = build_cycles()
    edge_to_cycles = build_edge_cycle_map(cycles)
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    def is_edge_change_valid(colors, ei, new_color):
        old_color = colors[ei]
        if new_color == old_color:
            return True
        for ci in edge_to_cycles[ei]:
            cyc = cycles[ci]
            cset = set()
            for e in cyc:
                if e == ei:
                    cset.add(new_color)
                else:
                    cset.add(colors[e])
                if len(cset) > 3:
                    return False
        return True
    
    def count_distinct(colors):
        return len(set(colors))
    
    # Strategy: start with all 0s, try to introduce new colors
    def try_introduce_color(colors, target_color, max_steps=10000):
        """Try to assign target_color to some edge, using local search to fix violations."""
        color_count = [0] * 120
        for c in colors:
            color_count[c] += 1
        
        # Find an edge that can take target_color
        # Try all edges
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for ei in edges:
            if colors[ei] == target_color:
                continue
            if is_edge_change_valid(colors, ei, target_color):
                colors[ei] = target_color
                color_count[colors[ei]] += 1
                if color_count[colors[ei]] == 1:
                    color_count[target_color] = 1
                else:
                    color_count[target_color] += 1
                return True, colors
        
        # If no edge can directly take the color, try to modify the coloring
        # This is complex; skip for now
        return False, colors
    
    # Greedy: try to use as many colors as possible
    def greedy():
        colors = [0] * NUM_EDGES
        distinct = 1
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for target in range(1, 120):
            if time.time() > deadline:
                break
            # Try to assign color 'target' to some edge
            success = False
            for ei in edge_order:
                if colors[ei] == target:
                    success = True
                    break
                if is_edge_change_valid(colors, ei, target):
                    colors[ei] = target
                    success = True
                    break
            
            if not success:
                # Try local search to make room
                # Pick a random edge with color 'target'... no, target isn't used yet
                # Try to find an edge where we can change to target by first changing others
                # For simplicity, just skip
                pass
        
        return colors, count_distinct(colors)
    
    # Better greedy: assign edges one by one, trying new colors
    def greedy_v2():
        colors = [0] * NUM_EDGES
        color_used = [False] * 120
        color_used[0] = True
        distinct = 1
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            # Try unused colors first
            unused = [c for c in range(120) if not color_used[c]]
            random.shuffle(unused)
            
            assigned = False
            for c in unused:
                if is_edge_change_valid(colors, ei, c):
                    colors[ei] = c
                    color_used[c] = True
                    distinct += 1
                    assigned = True
                    break
            
            if not assigned:
                used = [c for c in range(120) if color_used[c]]
                random.shuffle(used)
                for c in used:
                    if is_edge_change_valid(colors, ei, c):
                        colors[ei] = c
                        assigned = True
                        break
            
            if not assigned:
                colors[ei] = 0
        
        return colors, distinct
    
    # Local search with simulated annealing
    def local_search(colors, max_steps=50000):
        cur_distinct = count_distinct(colors)
        best_local = colors[:]
        best_local_distinct = cur_distinct
        
        color_count = [0] * 120
        for c in colors:
            color_count[c] += 1
        distinct = sum(1 for c in color_count if c > 0)
        
        T = 0.5  # temperature
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            ei = random.randint(0, NUM_EDGES - 1)
            old_color = colors[ei]
            new_color = random.randint(0, 119)
            
            if new_color == old_color:
                continue
            
            if not is_edge_change_valid(colors, ei, new_color):
                continue
            
            # Compute delta distinct
            old_count = color_count[old_color]
            new_count = color_count[new_color]
            
            delta_distinct = 0
            if old_count == 1:
                delta_distinct -= 1
            if new_count == 0:
                delta_distinct += 1
            
            # Accept
            colors[ei] = new_color
            color_count[old_color] -= 1
            color_count[new_color] += 1
            distinct += delta_distinct
            
            if distinct > best_local_distinct:
                best_local_distinct = distinct
                best_local = colors[:]
            
            # Annealing: revert if worse with probability
            if delta_distinct < 0:
                if random.random() > 0.1:  # mostly revert
                    colors[ei] = old_color
                    color_count[old_color] += 1
                    color_count[new_color] -= 1
                    distinct -= delta_distinct
            # else keep (improvement or neutral)
            
            T *= 0.9999
        
        return best_local, best_local_distinct
    
    # Run multiple restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        colors, distinct = greedy_v2()
        
        if distinct > best_score:
            best_score = distinct
            best_colors = colors[:]
        
        if time.time() > deadline:
            break
        
        improved_colors, improved_distinct = local_search(colors, max_steps=40000)
        
        if improved_distinct > best_score:
            best_score = improved_distinct
            best_colors = improved_colors[:]
    
    return best_colors
