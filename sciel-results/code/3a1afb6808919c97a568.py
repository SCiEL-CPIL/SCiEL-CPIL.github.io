import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Build edge index: (i,j) -> index in lexicographic order
    edge_idx = {}
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = count
            count += 1
    
    # Build all 4-cycles: for each 4-vertex subset {a,b,c,d}, three 4-cycles
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append([edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]])
                    # Cycle 2: a-b-d-c-a
                    cycles.append([edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]])
                    # Cycle 3: a-c-b-d-a
                    cycles.append([edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]])
    
    num_cycles = len(cycles)
    
    # For each edge, list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    def is_valid(coloring):
        for cycle in cycles:
            c0 = coloring[cycle[0]]
            c1 = coloring[cycle[1]]
            c2 = coloring[cycle[2]]
            c3 = coloring[cycle[3]]
            # Check if all four are distinct
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    # Try algebraic base colorings (i+j)%K for various K
    best_coloring = [0] * NUM_EDGES
    best_count = 1
    
    for K in range(2, 17):
        coloring = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                coloring[edge_idx[(i, j)]] = (i + j) % K
        if is_valid(coloring):
            nd = len(set(coloring))
            if nd > best_count:
                best_count = nd
                best_coloring = coloring.copy()
    
    # If no valid algebraic coloring found, use all zeros
    if best_count <= 1:
        best_coloring = [0] * NUM_EDGES
        best_count = 1
    
    # Simulated annealing local search
    current = best_coloring.copy()
    
    # Track color usage
    color_usage = [0] * MAX_COLOR
    for c in current:
        color_usage[c] += 1
    current_distinct = sum(1 for c in range(MAX_COLOR) if color_usage[c] > 0)
    
    # Per-cycle color counts for O(1) feasibility checks
    cycle_counts = [{} for _ in range(num_cycles)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            c = current[e]
            cycle_counts[ci][c] = cycle_counts[ci].get(c, 0) + 1
    
    best_found = current.copy()
    best_found_distinct = current_distinct
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999995
    
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current[e]
        new_color = random.randint(0, MAX_COLOR - 1)
        
        if new_color == old_color:
            continue
        
        affected = edge_to_cycles[e]
        
        # Check feasibility: ensure no cycle ends up with 4 distinct colors
        feasible = True
        for ci in affected:
            cc = cycle_counts[ci]
            old_count = cc.get(old_color, 0)
            new_count = cc.get(new_color, 0)
            old_distinct = len(cc)
            
            # After removing old_color
            if old_count == 1:
                mid_distinct = old_distinct - 1
            else:
                mid_distinct = old_distinct
            
            # After adding new_color
            if new_count == 0:
                final_distinct = mid_distinct + 1
            else:
                final_distinct = mid_distinct
            
            if final_distinct == 4:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Compute change in global distinct color count
        new_color_is_new = (color_usage[new_color] == 0)
        old_color_lost = (color_usage[old_color] == 1)
        
        if new_color_is_new and not old_color_lost:
            delta = 1
        elif new_color_is_new and old_color_lost:
            delta = 0
        elif not new_color_is_new and not old_color_lost:
            delta = 0
        else:
            delta = -1
        
        # Metropolis acceptance
        if delta >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta / T)
        
        if accept:
            current[e] = new_color
            color_usage[old_color] -= 1
            color_usage[new_color] += 1
            current_distinct += delta
            
            for ci in affected:
                cc = cycle_counts[ci]
                cc[old_color] -= 1
                if cc[old_color] == 0:
                    del cc[old_color]
                cc[new_color] = cc.get(new_color, 0) + 1
            
            if current_distinct > best_found_distinct:
                best_found_distinct = current_distinct
                best_found = current.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if best_found_distinct >= best_count:
        return best_found
    return best_coloring
