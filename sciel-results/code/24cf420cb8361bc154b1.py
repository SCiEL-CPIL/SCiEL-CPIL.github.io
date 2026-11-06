import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # We will try n from 18 down to 10
    # If we find a valid coloring for n, we return it immediately?
    # No, we want the MAX n.
    # So we should try 18, then 17, etc.
    # But if 18 is impossible, we waste time.
    # However, local search is fast. We can give 18 a chance, then 17.
    
    best_result = None
    best_n = 0
    
    # Fallback seed for n=9
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': colors_seed}
    best_n = n_seed
    
    # Try n=18 first with a limited time, then n=17 with remaining time?
    # Or just run local search for n=18 for 50s, if fail, n=17 for 50s.
    # Actually, since 17 is known, 17 should be found very fast.
    # 18 is likely unsolvable.
    
    # Strategy: 
    # 1. Try n=18 for 20 seconds.
    # 2. If not found, try n=17 for remaining time.
    # 3. If n=17 found, return it.
    # 4. If n=17 not found (unlikely), try 16, etc.
    
    target_ns = [18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    # Allocate time
    # 18: 30s
    # 17: 60s
    # 16: 10s
    # ...
    
    time_alloc = {
        18: 30,
        17: 60,
        16: 10,
        15: 5,
        14: 5,
        13: 2,
        12: 2,
        11: 1,
        10: 1
    }
    
    for n in target_ns:
        if time.time() > deadline - 5: # Reserve 5s for return
            break
            
        t_limit = time_alloc.get(n, 5)
        t_start = time.time()
        
        solution = local_search(n, t_start + t_limit)
        
        if solution is not None:
            # Found valid coloring for n
            # Since we iterate from high to low, this is the best found so far
            # But wait, if we find 17, we should stop? 
            # Yes, because we are looking for MAX n.
            # If we found 17, we don't need to check 16.
            return {'n': n, 'colors': solution}
            
    return best_result

def local_search(n, deadline):
    """
    Finds a valid 4-coloring of K_n with no monochromatic C6.
    Returns list of colors or None.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute cycles
    # This is the expensive part, but done once per n.
    # For n=18, ~1.1M cycles.
    
    # To speed up, we can generate cycles on the fly? No, delta needs precomputed links.
    # Let's precompute.
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    all_cycles = []
    
    # Optimization: only generate cycles if n is small enough?
    # n=18 is the max we try.
    
    # Generate all combinations of 6 vertices
    # For each combination, generate 60 unique cycles
    
    vertices = list(range(n))
    
    # We can skip precomputation if n is too large and time is tight?
    # For n=18, precomputation takes ~1-2s.
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        # Generate 60 cycles
        # Fix v0 at start. Permute v1..v5.
        # To get unique cycles, we can iterate permutations of v1..v5 
        # and keep only those where v1 < v5 (to break reversal symmetry) 
        # and v1 < v2? No, just v1 < v5 is enough for undirected cycles starting at v0?
        # Actually, (v0, p1, ..., p5) is same as (v0, p5, ..., p1).
        # So we can restrict p1 < p5.
        # This gives 120/2 = 60 cycles.
        
        perms = list(itertools.permutations([v1, v2, v3, v4, v5]))
        for perm in perms:
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            # Store cycle
            cycle_id = len(all_cycles)
            all_cycles.append(edge_indices)
            for idx in edge_indices:
                cycles_for_edge[idx].append(cycle_id)
                
    if time.time() > deadline:
        return None
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Count initial mono cycles
    def count_mono_cycles(col):
        count = 0
        for cycle in all_cycles:
            c0 = col[cycle[0]]
            # Unroll for speed?
            if col[cycle[1]] == c0 and col[cycle[2]] == c0 and col[cycle[3]] == c0 and col[cycle[4]] == c0 and col[cycle[5]] == c0:
                count += 1
        return count
        
    current_cost = count_mono_cycles(colors)
    if current_cost == 0:
        return colors[:]
        
    best_colors = colors[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # We will run until deadline
    # Use a step counter or just time
    
    while time.time() < deadline - 0.5:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        
        # Try changing to a different color
        c_old = colors[e_idx]
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        # Calculate delta
        delta = 0
        for cycle_id in cycles_for_edge[e_idx]:
            cycle = all_cycles[cycle_id]
            # Check if other 5 edges are all c_old
            # And if they are all c_new
            
            all_old = True
            all_new = True
            for e in cycle:
                if e == e_idx:
                    continue
                c = colors[e]
                if c != c_old:
                    all_old = False
                if c != c_new:
                    all_new = False
                # Early break if both false
                if not all_old and not all_new:
                    break
                    
            if all_old:
                delta -= 1
            if all_new:
                delta += 1
                
        new_cost = current_cost + delta
        
        # Acceptance criterion
        if new_cost < current_cost:
            # Accept
            colors[e_idx] = c_new
            current_cost = new_cost
            
            if current_cost == 0:
                return colors[:]
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                
        else:
            # Metropolis criterion
            # delta is positive (cost increases)
            if delta > 0:
                # Probability exp(-delta / T)
                # delta is integer, T is float
                # If T is small, this is very small
                import math
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[e_idx] = c_new
                    current_cost = new_cost
            # else delta == 0, accept with prob 1? Or 0.5?
            # If delta == 0, it's a neutral move. Accept to explore.
            else:
                if random.random() < 0.5:
                    colors[e_idx] = c_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return None

# Helper to check validity for the final result?
# The task says "No colour class may contain a six-cycle".
# The local search minimizes the count. If count is 0, it's valid.
# We return None if not 0.

# One issue: The precomputation for n=18 might be slow if done in pure Python.
# Let's estimate.
# 18564 combos * 60 cycles = 1.1M cycles.
# Inner loop: 6 edge lookups.
# 1.1M * 6 = 6.6M operations.
# In Python, this might take 1-3 seconds.
# This is acceptable.

# However, if we restart local search multiple times (e.g. for different n),
# we precompute each time.
# We can cache the cycle structures for n=18 and n=17?
# But the function construct() is called once.
# So we just do it inside.

# To improve speed of delta calculation:
# The inner loop `for e in cycle` is executed for every neighbor cycle of e_idx.
# Average degree of an edge in the cycle graph?
# Total incidences = 1.1M * 6 = 6.6M.
# Edges = 153.
# Avg cycles per edge = 6.6M / 153 ~ 43000.
# So for each move, we check ~43000 cycles.
# Each check takes ~5-10 ops.
# So ~400k ops per move.
# If we do 10k moves/sec, that's 4 billion ops/sec. Too slow.
# We need to optimize delta calculation.

# Optimization:
# Instead of checking all 5 edges every time, we can maintain counts?
# Or just use a faster check.
# Actually, 43000 cycles per edge is high.
# Let's re-evaluate.
# Total cycles ~ 1.1M.
# Each cycle has 6 edges.
# Sum of lengths of cycles_for_edge lists = 1.1M * 6 = 6.6M.
# Number of edges = 153.
# Average list size = 6.6M / 153 = 43137.
# Yes.
# 43k checks per move is too slow for Python if we do it naively.
# 1000 moves/sec * 43k = 43M ops/sec. Python can do ~10-50M simple ops/sec.
# So we might get ~100-1000 moves/sec.
# In 60s, that's 6000-60000 moves.
# Might be enough to escape local minima? Maybe not.

# How to speed up?
# 1. Use NumPy? Hard to vectorize the "all equal" check easily without overhead.
# 2. Reduce cycles? No, we need exact.
# 3. Better heuristic?
# 4. Parallelism? Not allowed.

# Alternative:
# Instead of checking ALL cycles for an edge, can we prune?
# No, any cycle can become mono.

# Let's try to optimize the inner loop in Python.
# `for e in cycle: ...`
# We can store the other 5 edges in a tuple for each (edge, cycle) pair?
# `cycles_for_edge[e_idx]` currently stores `cycle_id`.
# We can store `(other_edges_tuple, cycle_id)`?
# Then we don't need to lookup `all_cycles[cycle_id]` and iterate.
# We just iterate the tuple.
# This saves the list lookup and the `if e == e_idx` check.

# Let's modify precomputation to store `other_edges` for each edge in the cycle.
# `edge_cycle_info[e_idx]` = list of tuples `(other_edges_tuple, cycle_id)`?
# Actually, we just need the other edges to check colors. We don't need cycle_id for delta.
# So `edge_cycle_info[e_idx]` = list of tuples of 5 edge indices.

# Let's re-implement precomputation with this structure.

def local_search_optimized(n, deadline):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute: for each edge, list of tuples of the other 5 edges in each C6 containing it.
    edge_other_edges = [[] for _ in range(num_edges)]
    
    vertices = list(range(n))
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        perms = list(itertools.permutations([v1, v2, v3, v4, v5]))
        for perm in perms:
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            # For each edge in this cycle, add the other 5 to its list
            for i in range(6):
                e_idx = edge_indices[i]
                others = [edge_indices[j] for j in range(6) if j != i]
                edge_other_edges[e_idx].append(tuple(others))
                
    if time.time() > deadline:
        return None
        
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    def count_mono_cycles(col):
        count = 0
        # We don't have all_cycles list anymore.
        # We can reconstruct or just count from scratch using a different method?
        # Or keep all_cycles for initial count.
        # Let's keep all_cycles for initial count.
        pass
        
    # Actually, for initial count, we can just iterate all cycles.
    # Let's store all_cycles as well.
    all_cycles = []
    # Re-do precomputation to store both?
    # Or just calculate initial count by checking all cycles generated.
    # We can store all_cycles in the first pass.
    
    # Let's restart precomputation to store both.
    edge_other_edges = [[] for _ in range(num_edges)]
    all_cycles = []
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        perms = list(itertools.permutations([v1, v2, v3, v4, v5]))
        for perm in perms:
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            all_cycles.append(edge_indices)
            for i in range(6):
                e_idx = edge_indices[i]
                others = [edge_indices[j] for j in range(6) if j != i]
                edge_other_edges[e_idx].append(tuple(others))
                
    if time.time() > deadline:
        return None
        
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    def count_mono_cycles(col):
        count = 0
        for cycle in all_cycles:
            c0 = col[cycle[0]]
            if col[cycle[1]] == c0 and col[cycle[2]] == c0 and col[cycle[3]] == c0 and col[cycle[4]] == c0 and col[cycle[5]] == c0:
                count += 1
        return count
        
    current_cost = count_mono_cycles(colors)
    if current_cost == 0:
        return colors[:]
        
    best_colors = colors[:]
    best_cost = current_cost
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    import math
    
    while time.time() < deadline - 0.5:
        e_idx = random.randint(0, num_edges - 1)
        c_old = colors[e_idx]
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        delta = 0
        for others in edge_other_edges[e_idx]:
            # Check if all others are c_old
            all_old = True
            all_new = True
            for e in others:
                c = colors[e]
                if c != c_old:
                    all_old = False
                if c != c_new:
                    all_new = False
                if not all_old and not all_new:
                    break
            if all_old:
                delta -= 1
            if all_new:
                delta += 1
                
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            colors[e_idx] = c_new
            current_cost = new_cost
            if current_cost == 0:
                return colors[:]
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
        else:
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[e_idx] = c_new
                    current_cost = new_cost
            else:
                if random.random() < 0.5:
                    colors[e_idx] = c_new
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_result = None
    best_n = 0
    
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': colors_seed}
    best_n = n_seed
    
    target_ns = [18, 17, 16, 15]
    
    time_alloc = {
        18: 40,
        17: 60,
        16: 10,
        15: 5
    }
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        t_limit = time_alloc.get(n, 5)
        t_start = time.time()
        
        solution = local_search_optimized(n, t_start + t_limit)
        
        if solution is not None:
            return {'n': n, 'colors': solution}
            
    return best_result
