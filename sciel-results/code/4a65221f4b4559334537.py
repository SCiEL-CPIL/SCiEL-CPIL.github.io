import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We include weight-2, weight-3, and weight-4 vectors.
    # Weight-2: norm^2 = 2. C(11,2)*4 = 220 vectors.
    # Weight-3: norm^2 = 3. C(11,3)*8 = 165*8 = 1320 vectors.
    # Weight-4: norm^2 = 4. C(11,4)*16 = 330*16 = 5280 vectors.
    # Total candidates: 220 + 1320 + 5280 = 6820.
    
    candidates = []
    
    # Weight 2
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 3
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute Gram matrix
    G = candidates @ candidates.T
    
    # Build Conflict Matrix
    # Condition: ||u - v||^2 >= max(||u||^2, ||v||^2)
    # n_i + n_j - 2 G[i,j] >= max(n_i, n_j)
    # 2 G[i,j] <= min(n_i, n_j)
    # G[i,j] <= min(n_i, n_j) / 2.0
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    
    # Conflict if G[i,j] > limits[i,j]
    conflict_mat = G > limits
    np.fill_diagonal(conflict_mat, False)
    
    # Convert to bitsets for fast operations
    # n_cand is ~6820. We need 6820 bitsets of length 6820.
    # 6820 / 64 ~ 107 uint64s per bitset.
    # Total memory: 6820 * 107 * 8 bytes ~ 5.8 MB. Very small.
    
    n_words = (n_cand + 63) // 64
    # adj_bits[i] is a list of n_words uint64s representing the neighbors of i
    adj_bits = [np.zeros(n_words, dtype=np.uint64) for _ in range(n_cand)]
    
    # Fill adj_bits
    # conflict_mat is bool. 
    # We can iterate rows.
    for i in range(n_cand):
        # Get neighbors
        neighbors = np.nonzero(conflict_mat[i])[0]
        for j in neighbors:
            word_idx = j // 64
            bit_idx = j % 64
            adj_bits[i][word_idx] |= (1 << bit_idx)
            
    # Branch and Bound for Maximum Independent Set
    # We want to find the largest subset of vertices such that no two are connected in conflict_mat.
    # This is equivalent to finding the Maximum Independent Set in the graph defined by conflict_mat.
    
    best_size = 0
    best_set = []
    
    # Heuristic ordering: sort vertices by degree (descending) or just use natural order?
    # Often, high degree vertices are more constraining, so branching on them early might prune faster.
    # However, for MIS, branching on a vertex and its non-neighbors is standard.
    # Let's use a simple recursive B&B with bitsets.
    
    # To improve performance, we can use a greedy coloring bound (Tomita algorithm).
    # But implementing Tomita with bitsets in Python might be complex.
    # A simpler B&B with bitsets might suffice for N=6820 if the graph is dense enough to prune well.
    # Wait, if the graph is dense, the MIS is small.
    # If the graph is sparse, the MIS is large.
    # In our case, conflicts are specific.
    # Let's estimate density.
    # Most pairs are NOT conflicting.
    # So the graph is SPARSE.
    # For sparse graphs, B&B can be slow if the MIS is large.
    # However, we have a good initial lower bound from a greedy algorithm.
    
    # Greedy Initial Solution
    # Order by degree (ascending) to pick vertices with fewest conflicts first?
    # Or just random/greedy?
    # Let's try a greedy that picks the vertex with minimum degree in the remaining graph.
    
    remaining = np.ones(n_cand, dtype=bool)
    greedy_set = []
    degrees = np.sum(conflict_mat, axis=1)
    
    # To speed up greedy, we can just iterate and pick.
    # Since N is 6820, O(N^2) is ~46M ops, which is fast in NumPy but slow in Python loop.
    # Let's use a simple vectorized greedy step or just a few random restarts.
    
    # Actually, let's use the "min-degree" heuristic but implemented efficiently.
    # We'll just do a simple greedy:
    # 1. Calculate degrees.
    # 2. Pick vertex with min degree.
    # 3. Remove it and its neighbors.
    # 4. Repeat.
    
    # This is O(N^2) if we recalculate degrees.
    # N=6820, N^2=46M. In Python, this loop is too slow if done naively 6820 times.
    # 6820 iterations * (vector ops on size 6820) ~ 6820 * 6820 / 100 ~ 460k ops? 
    # NumPy vector ops on 6820 elements are fast (~microseconds).
    # 6820 * 1us = 6.8ms. This is very fast!
    
    current_degrees = degrees.copy()
    active = np.ones(n_cand, dtype=bool)
    
    while np.any(active):
        # Find active vertex with min degree
        # Set degrees of inactive to inf
        active_degrees = np.where(active, current_degrees, np.inf)
        v = np.argmin(active_degrees)
        
        greedy_set.append(v)
        
        # Remove v and its neighbors
        # neighbors of v are where conflict_mat[v] is True
        to_remove = np.zeros(n_cand, dtype=bool)
        to_remove[v] = True
        to_remove[conflict_mat[v]] = True
        
        active[to_remove] = False
        # Update degrees for remaining active vertices
        # degree of u decreases by 1 if u was neighbor of v
        # current_degrees[active & conflict_mat[v]] -= 1
        # This is a vector op.
        current_degrees[active & conflict_mat[v]] -= 1
        
    best_size = len(greedy_set)
    best_set = greedy_set.copy()
    
    # Now B&B to try to improve.
    # We use a recursive function.
    # State: available vertices (bitset), current set size, current set list.
    # To avoid deep recursion and overhead, we can use an iterative stack or limit depth.
    # Given N=6820, recursion depth could be up to N. Python recursion limit is 1000.
    # We must increase recursion limit or use iterative approach.
    sys.setrecursionlimit(10000)
    
    # Bitset operations in Python lists of uint64s are slow if not careful.
    # Let's use Python integers for bitsets?
    # N=6820 bits. Python int handles this natively and bitwise ops are in C.
    # This is MUCH faster than numpy uint64 arrays for single-bitset operations.
    
    # Rebuild adj_bits as Python ints
    adj_ints = [0] * n_cand
    for i in range(n_cand):
        mask = 0
        # conflict_mat[i] is a bool array
        # We can construct the int by iterating set bits? No, that's slow.
        # We can use numpy to pack bits into bytes and then to int?
        # Or just use the previous adj_bits (numpy uint64) and convert?
        # Let's just use the numpy uint64 arrays but operate on them carefully.
        # Actually, for B&B, we need AND, OR, NOT, POPCOUNT.
        # Numpy doesn't have efficient popcount for uint64 arrays in older versions.
        # Python int has bit_count() (Python 3.10+).
        # Let's convert adj_bits to Python ints.
        
        # Efficient conversion from numpy uint64 array to Python int
        # b = adj_bits[i]
        # val = 0
        # for word in reversed(b):
        #    val = (val << 64) | int(word)
        # This is O(N/64) per vertex. 6820 * 107 = 730k ops. Fast enough.
        
        val = 0
        for word in reversed(adj_bits[i]):
            val = (val << 64) | int(word)
        adj_ints[i] = val
        
    # Full mask
    full_mask = (1 << n_cand) - 1
    
    # Recursive B&B
    # avail: int bitmask of available vertices
    # current_size: int
    # current_set: list of vertices
    
    # Optimization: If current_size + popcount(avail) <= best_size, prune.
    
    def branch_and_bound(avail, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Bound
        if current_size + avail.bit_count() <= best_size:
            return
            
        if avail == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            return
            
        # Heuristic: Pick a vertex to branch on.
        # Common choice: vertex with max degree in the subgraph induced by avail.
        # Or just the lowest index available?
        # Let's pick the vertex with the highest degree in the available set to maximize pruning.
        # Calculating degree for each available vertex is O(N^2) in worst case.
        # Let's just pick the first available vertex for simplicity, or one with high global degree.
        
        # Let's use a simple heuristic: pick the vertex in avail with the highest degree in the original graph.
        # This is static.
        
        # To find a vertex:
        # We can iterate through bits in avail.
        # But finding the one with max degree requires checking degrees.
        # Let's just pick the lowest index for now, or random.
        # Random might be better to avoid bias.
        
        # Let's try picking a vertex with high degree in the induced subgraph.
        # This is expensive.
        # Let's stick to a simple strategy: Pick the vertex in `avail` that has the most neighbors in `avail`.
        # We can approximate this by checking a few candidates.
        
        # For simplicity and speed in Python, let's just pick the first set bit.
        # This is O(1) to find (using bit_length or trailing zero).
        # But it might be suboptimal.
        
        # Better: Pick a vertex v in avail.
        # Branch 1: Include v.
        #   New avail = avail & ~adj_ints[v] & ~(1 << v)
        # Branch 2: Exclude v.
        #   New avail = avail & ~(1 << v)
        
        # To improve, we can order vertices by degree.
        # Let's pre-sort vertices by degree descending.
        # Then in B&B, we pick the first available vertex in this order.
        
        # Pre-sorted indices
        sorted_indices = np.argsort(-degrees)
        
        # Find the first available vertex in sorted_indices
        v = -1
        for idx in sorted_indices:
            if (avail >> idx) & 1:
                v = idx
                break
                
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Branch 1: Include v
        # New available: remove v and its neighbors
        new_avail_1 = avail & ~adj_ints[v] & ~bit_v
        current_set.append(v)
        branch_and_bound(new_avail_1, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_avail_2 = avail & ~bit_v
        branch_and_bound(new_avail_2, current_size, current_set)
        
    # Start B&B
    # We already have a good best_size from greedy.
    # Start with full_mask
    # To avoid exploring too much, we can restrict the search space?
    # No, let's run it.
    
    # One issue: Recursion depth.
    # If the MIS is large (e.g., 1000), depth is 1000.
    # sys.setrecursionlimit(10000) should handle it.
    
    # However, Python recursion is slow.
    # For N=6820, this might be too slow if the branching factor is bad.
    # Let's add a time check in the loop? No, it's recursive.
    # The time check is inside.
    
    # Let's run it.
    # If it takes too long, it will stop at deadline.
    
    # To make it faster, we can use a "color bound" but that's complex.
    # Let's rely on the strong lower bound from greedy.
    
    # If the greedy solution is already optimal or near-optimal, B&B will prune heavily.
    # If not, it might take a while.
    
    # Let's try.
    # Note: `avail.bit_count()` is fast.
    
    # Initial call
    # We can pass the greedy set as initial best, which we did.
    
    # To prevent stack overflow or extreme slowness, we can limit the depth or use iterative deepening?
    # No, standard B&B is fine.
    
    # Let's run B&B
    branch_and_bound(full_mask, 0, [])
    
    # Convert best_set indices to vectors
    final_vecs = candidates[best_set]
    
    return final_vecs

