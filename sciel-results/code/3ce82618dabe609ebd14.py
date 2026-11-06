"""Gallai coloring with local search to maximize distinct colors."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges

def edge_idx(i, j):
    """Index for edge (i,j) in lexicographic order."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s kill
    
    random.seed(42)
    
    # Precompute all 4-cycles: for each set of 4 vertices, the 3 possible 4-cycles
    # A 4-cycle on vertices a,b,c,d (sorted) has 3 forms:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) - but we need to be careful
    # Actually for vertices {a,b,c,d}, the 3 4-cycles are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # Each uses 4 edges.
    
    # Precompute all 4-cycles as edge tuples
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # 3 distinct 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    cycles.append(((a,b), (b,c), (c,d), (a,d)))
                    # Cycle 2: a-b-d-c-a  
                    cycles.append(((a,b), (b,d), (d,c), (a,c)))
                    # Cycle 3: a-c-b-d-a
                    cycles.append(((a,c), (c,b), (b,d), (a,d)))
    
    # Convert to edge indices
    cycle_edges = []
    for cyc in cycles:
        cycle_edges.append(tuple(edge_idx(u, v) for u, v in cyc))
    
    # For fast checking: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(M)]
    for ci, cyc in enumerate(cycle_edges):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    
    # Start with a Gallai coloring (recursive partition structure)
    # This guarantees no rainbow triangles, hence no rainbow 4-cycles
    def gallai_coloring(vertices):
        """Recursively color edges within vertex set using Gallai structure."""
        n = len(vertices)
        if n <= 1:
            return {}
        if n == 2:
            i, j = vertices[0], vertices[1]
            return {(i,j): 0}
        
        # Random partition into two non-empty parts
        if n == 3:
            # Color with at most 2 colors to avoid rainbow triangle
            colors = random.sample([0, 1, 2], 2)
            edges = [(vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[0], vertices[2])]
            result = {}
            result[edges[0]] = colors[0]
            result[edges[1]] = colors[1]
            # Third edge: pick color to avoid rainbow
            # If first two are different, third can be either
            result[edges[2]] = random.choice(colors)
            return result
        
        # Partition
        split = random.randint(1, n - 1)
        part1 = vertices[:split]
        part2 = vertices[split:]
        
        # Color between parts with 1 or 2 colors
        between_colors = random.sample(range(10), min(2, random.randint(1, 2)))
        result = {}
        
        # Between parts
        for i in part1:
            for j in part2:
                result[(i,j)] = random.choice(between_colors)
        
        # Within parts recursively
        result.update(gallai_coloring(part1))
        result.update(gallai_coloring(part2))
        
        return result
    
    # Build initial coloring
    def build_coloring():
        color_map = gallai_coloring(list(range(N)))
        colors = [0] * M
        for (i, j), c in color_map.items():
            colors[edge_idx(i, j)] = c
        return colors
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check no rainbow 4-cycle."""
        for cyc in cycle_edges:
            c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
            if len({c0, c1, c2, c3}) == 4:
                return False
        return True
    
    def try_add_color(colors, new_colors_target=30):
        """Try to add new colors to edges while maintaining validity."""
        used = set(colors)
        max_color = max(used) if used else 0
        available = [c for c in range(120) if c not in used]
        random.shuffle(available)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to change random edges to new colors
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                e = random.randint(0, M - 1)
                if not available:
                    break
                new_c = random.choice(available)
                old_c = colors[e]
                if new_c == old_c:
                    continue
                
                colors[e] = new_c
                # Check all cycles containing edge e
                valid = True
                for ci in edge_to_cycles[e]:
                    cyc = cycle_edges[ci]
                    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    if new_c not in used:
                        used.add(new_c)
                        if new_c in available:
                            available.remove(new_c)
                    improved = True
                else:
                    colors[e] = old_c
        
        return colors
    
    # Try multiple random initial colorings and local search
    best_count = 0
    best_colors = None
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        colors = build_coloring()
        if not is_valid(colors):
            continue
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        # Local search to add colors
        colors = try_add_color(colors)
        if is_valid(colors):
            cur_count = count_colors(colors)
            if cur_count > best_count:
                best_count = cur_count
                best_colors = colors[:]
        
        # Additional local search: try changing edges to existing new colors
        if time.time() < deadline:
            used = set(best_colors) if best_colors else set()
            available = [c for c in range(120) if c not in used]
            random.shuffle(available)
            
            for attempt in range(20000):
                if time.time() > deadline:
                    break
                if not available:
                    break
                e = random.randint(0, M - 1)
                new_c = random.choice(available)
                old_c = best_colors[e]
                if new_c == old_c:
                    continue
                
                best_colors[e] = new_c
                valid = True
                for ci in edge_to_cycles[e]:
                    cyc = cycle_edges[ci]
                    c0, c1, c2, c3 = best_colors[cyc[0]], best_colors[cyc[1]], best_colors[cyc[2]], best_colors[cyc[3]]
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    if new_c not in used:
                        used.add(new_c)
                        if new_c in available:
                            available.remove(new_c)
                        best_count = len(used)
                else:
                    best_colors[e] = old_c
    
    # Final check
    if best_colors is None:
        best_colors = [0] * M
    
    if not is_valid(best_colors):
        # Fallback to simple Gallai
        best_colors = build_coloring()
    
    return best_colors

# Precompute for speed - do it at module level
N = 16
M = N * (N - 1) // 2

def edge_idx(i, j):
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute cycles at module level
_cycles = []
for _a in range(N):
    for _b in range(_a+1, N):
        for _c in range(_b+1, N):
            for _d in range(_c+1, N):
                _cycles.append(((edge_idx(_a,_b), edge_idx(_b,_c), edge_idx(_c,_d), edge_idx(_a,_d)),
                                (edge_idx(_a,_b), edge_idx(_b,_d), edge_idx(_d,_c), edge_idx(_a,_c)),
                                (edge_idx(_a,_c), edge_idx(_c,_b), edge_idx(_b,_d), edge_idx(_a,_d))))

_cycle_edges = []
for _trip in _cycles:
    for _cyc in _trip:
        _cycle_edges.append(_cyc)

_edge_to_cycles = [[] for _ in range(M)]
for _ci, _cyc in enumerate(_cycle_edges):
    for _e in _cyc:
        _edge_to_cycles[_e].append(_ci)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(42)
    
    def gallai_coloring(vertices):
        n = len(vertices)
        if n <= 1:
            return {}
        if n == 2:
            i, j = vertices[0], vertices[1]
            return {(i,j): 0}
        
        if n == 3:
            colors = random.sample([0, 1, 2], 2)
            e0 = (vertices[0], vertices[1])
            e1 = (vertices[1], vertices[2])
            e2 = (vertices[0], vertices[2])
            result = {e0: colors[0], e1: colors[1], e2: random.choice(colors)}
            return result
        
        split = random.randint(1, n - 1)
        part1 = vertices[:split]
        part2 = vertices[split:]
        
        between_colors = random.sample(range(10), random.randint(1, 2))
        result = {}
        
        for i in part1:
            for j in part2:
                result[(i,j)] = random.choice(between_colors)
        
        result.update(gallai_coloring(part1))
        result.update(gallai_coloring(part2))
        
        return result
    
    def build_coloring():
        color_map = gallai_coloring(list(range(N)))
        colors = [0] * M
        for (i, j), c in color_map.items():
            colors[edge_idx(i, j)] = c
        return colors
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        for cyc in _cycle_edges:
            c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
            if len({c0, c1, c2, c3}) == 4:
                return False
        return True
    
    def try_add_color(colors):
        used = set(colors)
        available = [c for c in range(120) if c not in used]
        random.shuffle(available)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                if not available:
                    break
                e = random.randint(0, M - 1)
                new_c = random.choice(available)
                old_c = colors[e]
                if new_c == old_c:
                    continue
                
                colors[e] = new_c
                valid = True
                for ci in _edge_to_cycles[e]:
                    cyc = _cycle_edges[ci]
                    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    if new_c not in used:
                        used.add(new_c)
                        if new_c in available:
                            available.remove(new_c)
                    improved = True
                else:
                    colors[e] = old_c
        
        return colors
    
    best_count = 0
    best_colors = None
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        colors = build_coloring()
        if not is_valid(colors):
            continue
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        colors = try_add_color(colors)
        if is_valid(colors):
            cur_count = count_colors(colors)
            if cur_count > best_count:
                best_count = cur_count
                best_colors = colors[:]
        
        # More aggressive search on best
        if time.time() < deadline - 5 and best_colors:
            used = set(best_colors)
            available = [c for c in range(120) if c not in used]
            random.shuffle(available)
            
            for attempt in range(30000):
                if time.time() > deadline - 2:
                    break
                if not available:
                    break
                e = random.randint(0, M - 1)
                new_c = random.choice(available)
                old_c = best_colors[e]
                if new_c == old_c:
                    continue
                
                best_colors[e] = new_c
                valid = True
                for ci in _edge_to_cycles[e]:
                    cyc = _cycle_edges[ci]
                    c0, c1, c2, c3 = best_colors[cyc[0]], best_colors[cyc[1]], best_colors[cyc[2]], best_colors[cyc[3]]
                    if len({c0, c1, c2, c3}) == 4:
                        valid = False
                        break
                if valid:
                    if new_c not in used:
                        used.add(new_c)
                        if new_c in available:
                            available.remove(new_c)
                        best_count = len(used)
                else:
                    best_colors[e] = old_c
    
    if best_colors is None:
        best_colors = [0] * M
    
    if not is_valid(best_colors):
        best_colors = build_coloring()
    
    return best_colors
