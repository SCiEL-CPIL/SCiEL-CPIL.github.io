"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _generate_candidates():
    """
    Generate a pool of candidate vectors.
    We include shells 1 through 4. 
    Shell 1: 11 vectors (norm 1)
    Shell 2: 220 vectors (norm sqrt(2))
    Shell 3: 1320 vectors (norm sqrt(3))
    Shell 4: 5280 vectors (norm 4)
    Total: ~6800 vectors.
    """
    vecs = []
    
    # Shell 1: e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2: e_i - e_j
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3: Weight 3 vectors with entries in {1, -1}
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4: Weight 4 vectors with entries in {1, -1}
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # 1. Generate candidates
    candidates = _generate_candidates()
    n_cand = len(candidates)
    
    # 2. Compute conflict graph
    # Constraint: ||u - v||^2 >= max(||u||^2, ||v||^2)
    # Let W = candidates @ candidates.T (Gram matrix)
    # Let N = norms_sq[:, None]
    # dist_sq[i, j] = N[i] + N[j] - 2*W[i, j]
    # max_norm[i, j] = max(N[i], N[j])
    # conflict[i, j] = dist_sq[i, j] < max_norm[i, j]
    
    # Memory check: 6800 x 6800 int64 is ~360MB. Might be tight but usually OK.
    # Let's compute in chunks if needed, or just use float32 for intermediate if safe?
    # Integers are exact. Let's try to be memory efficient.
    # Actually, we only need the adjacency list or bitsets.
    
    # Precompute norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Initialize bitsets for neighbors (conflicts)
    # neighbors[i] is a Python int representing a bitmask of nodes j that conflict with i.
    neighbors = [0] * n_cand
    
    # We can compute conflicts in blocks to save memory if we don't store the full matrix.
    # Block size 1000.
    block_size = 1000
    for i_start in range(0, n_cand, block_size):
        i_end = min(i_start + block_size, n_cand)
        block = candidates[i_start:i_end]
        
        # Gram matrix for this block against ALL candidates?
        # That's (block_size x n_cand). 1000 x 6800 = 6.8M entries. Fine.
        dots = block @ candidates.T  # (block_size, n_cand)
        
        # norms for block: (block_size, 1)
        # norms for all: (1, n_cand)
        n_block = norms_sq[i_start:i_end, None]
        n_all = norms_sq[None, :]
        
        # dist_sq = n_block + n_all - 2 * dots
        dists_sq = n_block + n_all - 2 * dots
        
        # max_norms = max(n_block, n_all)
        max_norms = np.maximum(n_block, n_all)
        
        # conflicts = dists_sq < max_norms
        conflicts = dists_sq < max_norms
        
        # Convert conflict matrix to bitsets
        # For each row i in the block, we need a bitmask of columns j where conflicts[i, j] is True.
        # We can iterate rows.
        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            row_conflicts = conflicts[local_i]
            
            # Find indices where True
            # np.where returns array of indices.
            # We need to set bits.
            # Bit i is 1 << i.
            # For large i, 1 << i is fine in Python.
            
            # Optimization: 
            # row_conflicts is a boolean array.
            # We can convert to a string or bytes and parse? 
            # Or just loop over indices.
            # n_cand is ~6800. Looping 6800 times per row * 1000 rows = 6.8M iterations.
            # Python loop is slow.
            
            # Better: Use bitarray if available? No, stdlib only.
            # We can pack the boolean array into bytes and then to int?
            # bool array -> uint8 array -> bytes -> int.from_bytes
            # But int.from_bytes treats it as base 256, not base 2.
            # We need to map bits.
            
            # Trick:
            # Create a bytearray of length ceil(n_cand/8).
            # Set bits manually? Still slow in Python loop.
            
            # Alternative:
            # Since we are using Python ints as bitsets, we can build them by summing (1 << j) for j in conflicts.
            # sum(1 << j for j in np.where(row_conflicts)[0])
            # This creates a generator. Summing big ints is O(N^2) bit ops?
            # Actually, adding 1 << j to a big int is fast if the int is sparse?
            
            # Let's try a hybrid:
            # If the number of conflicts is small, loop.
            # If dense, maybe use a different representation?
            # In kissing number problems, the graph is usually sparse enough or structured.
            
            # Let's use the sum approach but optimize the inner loop.
            idxs = np.where(row_conflicts)[0]
            mask = 0
            for j in idxs:
                mask |= (1 << j)
            neighbors[global_i] = mask
            
        if time.time() > deadline - 20:
            break

    # 3. Solve MIS using Branch and Bound with Bitsets
    # State: available_mask (bits set for nodes we can still pick)
    # Best set: list of nodes in the current best solution
    # Current set: list of nodes in the current path
    
    best_set = []
    best_size = 0
    
    # Global variables for recursion (closures or nonlocal)
    # To avoid deep recursion limits and overhead, we can use an explicit stack or optimize recursion.
    # Python recursion limit is usually 1000. MIS depth can be > 1000?
    # Kissing number in 11D is ~ 2300? No, it's around 2300?
    # Actually, kissing number in 11D is 2300? 
    # Wait, kissing number A(11) is 2300?
    # Lower bound is 2300? 
    # If the solution size is > 1000, recursion might hit limits.
    # However, we can increase recursion limit.
    import sys
    sys.setrecursionlimit(10000)
    
    def solve_mis(available_mask):
        nonlocal best_set, best_size
        
        if available_mask == 0:
            return
            
        # Upper bound: popcount of available_mask
        # Python int bit_count() is fast
        if available_mask.bit_count() <= best_size:
            return
            
        # Greedy coloring bound? 
        # For now, simple bound: popcount.
        # This might be weak. 
        # A better bound: 
        # The graph is unit disk-like? 
        # Let's try a simple heuristic bound: 
        # If we assume we can pick at most popcount / avg_degree? No.
        
        # Let's use a simple greedy coloring on the subgraph induced by available_mask to get an upper bound.
        # Coloring gives an upper bound on independent set size.
        # But coloring is expensive.
        # Let's stick with popcount for now, but it might be loose.
        
        # Branching:
        # Pick a vertex v from available_mask.
        # Strategy: Pick vertex with max degree in the induced subgraph?
        # Or min degree?
        # Max degree branching: 
        # Branch 1: Include v. Available becomes available & ~neighbors[v] & ~(1<<v)
        # Branch 2: Exclude v. Available becomes available & ~(1<<v)
        
        # Which v to pick?
        # Picking a high degree vertex reduces the "Include" branch's available set significantly.
        # Picking a low degree vertex reduces the "Exclude" branch's available set minimally (just removes 1 bit).
        # Usually, picking a vertex with high degree is good for pruning the "Include" branch.
        # But we also need to consider the "Exclude" branch.
        
        # Heuristic: Pick the vertex with the highest degree in the available subgraph.
        # Degree of v in available subgraph = popcount(neighbors[v] & available_mask)
        
        # Finding max degree vertex in available_mask:
        # Iterate through set bits in available_mask.
        # This is O(N) in the worst case if we do it naively.
        # We can cache degrees? No, they change.
        
        # Optimization: 
        # Instead of finding the absolute max degree every time (which is expensive),
        # we can just pick the first set bit? 
        # Random picking?
        # Let's try picking the vertex with the highest "conflict count" within the available set.
        
        # To speed this up, we can just pick a random vertex from the available set.
        # Or, pick the vertex with the smallest index?
        # Let's try to pick a vertex that has many neighbors in the available set.
        
        # Efficiently finding a "good" vertex:
        # We can just iterate through the set bits and compute degree for the first few?
        # Or just pick the lowest index set bit?
        # Let's try lowest index set bit for simplicity and determinism.
        # v_idx = (available_mask & -available_mask).bit_length() - 1
        
        # Actually, let's try to pick a vertex with high degree.
        # We can sample a few vertices.
        
        # For now, let's use the lowest set bit. It's fast.
        lsb = available_mask & -available_mask
        v_idx = lsb.bit_length() - 1
        
        # Branch 1: Include v_idx
        # New available: remove v_idx and its neighbors
        new_available_include = available_mask & ~(neighbors[v_idx] | lsb)
        
        # Recursive call
        # We need to track the current set to update best_set if we find a better one.
        # Passing the current set list is expensive.
        # We can use a global stack or pass a tuple?
        # Let's use a global list `current_path` and backtrack.
        
        current_path.append(v_idx)
        solve_mis(new_available_include)
        current_path.pop()
        
        # Update best if current path is longer
        if len(current_path) > best_size:
            best_size = len(current_path)
            best_set = current_path[:]
            
        # Branch 2: Exclude v_idx
        # New available: remove v_idx
        new_available_exclude = available_mask & ~lsb
        
        # Pruning: If the upper bound of the exclude branch is <= best_size, skip.
        # Upper bound is popcount.
        if new_available_exclude.bit_count() > best_size:
             solve_mis(new_available_exclude)
             
        # Note: The "Include" branch already updated best_size.
        # So the check for "Exclude" uses the potentially updated best_size.

    current_path = []
    initial_mask = (1 << n_cand) - 1
    solve_mis(initial_mask)
    
    # 4. Finalize
    if not best_set:
        # Fallback
        return candidates[:1]
        
    result = candidates[best_set]
    return result

# EVOLVE-BLOCK-END
