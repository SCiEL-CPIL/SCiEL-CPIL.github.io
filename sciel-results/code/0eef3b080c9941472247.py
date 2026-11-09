import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        seen = set()
        for e in cycle:
            c = new_color if e == edge else colors[e]
            if c in seen:
                continue
            seen.add(c)
            if len(seen) > 3:
                return False
    return True

def get_cycle_colors(colors, edge, new_color):
    """For each cycle containing edge, return the set of other colors in that cycle."""
    result = []
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        other_colors = set()
        for e in cycle:
            if e != edge:
                other_colors.add(colors[e])
        result.append(other_colors)
    return result

def local_search_sa(colors, deadline, start_temp=10.0, cool_rate=0.995):
    """Simulated annealing local search."""
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    temp = start_temp
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with random perturbation
            for _ in range(10):
                e = random.randint(0, NUM_EDGES - 1)
                # Try to find a valid color
                for attempt in range(30):
                    nc = random.randint(0, 119)
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            used_colors = set(colors)
            no_improve = 0
            max_no_improve = random.randint(500, 2000)
            temp = start_temp
            continue
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose a new color: bias toward unused colors
        all_colors = set(range(120))
        unused = all_colors - used_colors
        if unused:
            # 70% chance to try an unused color, 30% to try any color
            if random.random() < 0.7:
                nc = random.choice(list(unused))
            else:
                nc = random.randint(0, 119)
        else:
            nc = random.randint(0, 119)
        
        if nc == old_color:
            continue
            
        if can_take_color(colors, e, nc):
            # Calculate delta in score
            old_used = len(used_colors)
            colors[e] = nc
            new_used = len(used_colors | {nc})
            # Remove old color if it's no longer used
            if colors.count(old_color) == 0:
                used_colors.discard(old_color)
                new_used = len(used_colors)
            else:
                used_colors.add(nc)
                new_used = len(used_colors)
            
            delta = new_used - old_used
            
            if delta >= 0 or random.random() < np.exp(delta / temp):
                if delta > 0:
                    no_improve = 0
                    if new_used > best_score:
                        best_score = new_used
                        best_colors = list(colors)
                else:
                    no_improve += 1
            else:
                # Reject: revert
                colors[e] = old_color
                if old_color not in used_colors:
                    # This shouldn't happen, but safety
                    pass
                no_improve += 1
        else:
            no_improve += 1
        
        temp *= cool_rate
        if temp < 0.01:
            temp = 0.01
    
    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Strategy: Try multiple construction methods
    methods = []
    
    # Method 1: Partition-based coloring
    # Divide 16 vertices into 4 groups of 4
    # Color edges within group vs between groups differently
    def partition_coloring(group_sizes, seed):
        random.seed(seed)
        perm = list(range(N))
        random.shuffle(perm)
        
        groups = []
        pos = 0
        for size in group_sizes:
            groups.append(perm[pos:pos+size])
            pos += size
        
        colors = [0] * NUM_EDGES
        used = set()
        color_id = 0
        
        # Assign colors to group pairs
        group_color = {}
        for gi in range(len(groups)):
            for gj in range(gi+1, len(groups)):
                group_color[(gi, gj)] = color_id
                color_id += 1
        
        # Within-group colors
        for gi in range(len(groups)):
            group = groups[gi]
            for ii in range(len(group)):
                for ij in range(ii+1, len(group)):
                    u, v = group[ii], group[ij]
                    e = EDGE_IDX[u][v]
                    # Use a color based on position within group
                    colors[e] = color_id + ii * 4 + ij
                    color_id += 1
        
        # Between-group edges
        for gi in range(len(groups)):
            for gj in range(gi+1, len(groups)):
                gc = group_color[(gi, gj)]
                for u in groups[gi]:
                    for v in groups[gj]:
                        e = EDGE_IDX[u][v]
                        colors[e] = gc
        
        return colors
    
    # Method 2: Random greedy with backtracking
    def random_greedy(seed):
        random.seed(seed)
        colors = [0] * NUM_EDGES
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e in edge_order:
            # Try colors in random order
            color_order = list(range(120))
            random.shuffle(color_order)
            for nc in color_order:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    break
        
        return colors
    
    # Method 3: Start from all-zeros and build up
    def build_up(seed):
        random.seed(seed)
        colors = [0] * NUM_EDGES
        used = {0}
        
        for e in range(NUM_EDGES):
            # Try to add a new color
            unused = [c for c in range(120) if c not in used]
            random.shuffle(unused)
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used.add(nc)
                    break
        
        return colors
    
    # Try different partition sizes
    partition_configs = [
        [4, 4, 4, 4],
        [5, 5, 3, 3],
        [6, 4, 3, 3],
        [8, 4, 2, 2],
        [4, 4, 4, 2, 2],
        [3, 3, 3, 3, 2, 2],
    ]
    
    attempts = []
    
    # Add partition-based attempts
    for config in partition_configs:
        for seed in range(10):
            attempts.append(('partition', config, seed))
    
    # Add random greedy attempts
    for seed in range(50):
        attempts.append(('greedy', None, seed))
    
    # Add build-up attempts
    for seed in range(50):
        attempts.append(('buildup', None, seed))
    
    random.shuffle(attempts)
    
    # Run attempts with time budget
    total_attempts = len(attempts)
    time_per_attempt = 50.0 / total_attempts  # ~0.25s each
    
    for method, config, seed in attempts:
        if time.time() >= deadline:
            break
        
        if method == 'partition':
            colors = partition_coloring(config, seed)
        elif method == 'greedy':
            colors = random_greedy(seed)
        elif method == 'buildup':
            colors = build_up(seed)
        else:
            continue
        
        if not is_valid(colors):
            continue
        
        score = len(set(colors))
        
        # Quick local search to improve
        ls_deadline = min(time.time() + 1.0, deadline)
        improved_colors, improved_score = local_search_sa(colors, ls_deadline, start_temp=5.0, cool_rate=0.99)
        
        if improved_score > best_score:
            best_score = improved_score
            best_colors = improved_colors
            
        if best_score >= 30:
            break
    
    # Final long local search on best solution
    if time.time() < deadline - 5:
        final_deadline = deadline
        final_colors, final_score = local_search_sa(best_colors, final_deadline, start_temp=10.0, cool_rate=0.998)
        if final_score > best_score:
            best_score = final_score
            best_colors = final_colors
    
    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
