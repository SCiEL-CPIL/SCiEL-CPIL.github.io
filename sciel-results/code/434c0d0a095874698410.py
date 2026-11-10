# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # 1. Generate Norm 2 vectors (Baseline: 220 points)
    # These always form a valid set.
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution
    best_k = len(base_solution)

    # 2. Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph using bitsets for speed
    # Two norm-4 vectors conflict if their dot product > 2
    # For vectors with 4 nonzero ±1 entries, dot product = sum of products of overlapping coords
    # If they share s coordinates with same sign and t with opposite sign, dot = s - t
    # Conflict when s - t > 2, i.e., s >= 3 (since s+t <= 4)
    # So conflict iff they share >= 3 coordinates with matching signs
    
    # Precompute for each vector: its support (4 coords) and signs
    # Use bit representation: each vector can be encoded as (support_mask, sign_pattern)
    # support_mask: 11-bit integer indicating which coords are nonzero
    # sign_pattern: 4-bit integer indicating +1 or -1 at each nonzero coord
    
    # Better: precompute conflict adjacency using numpy vectorization
    # dots = V4 @ V4.T gives all pairwise dot products
    # This is 15840 x 15840 = ~250M entries, too large for memory
    
    # Instead, use a smarter approach:
    # For each vector i, find all j where dot(V4[i], V4[j]) > 2
    # We can do this in batches to avoid memory issues
    
    # Actually, let's use a different strategy:
    # Group vectors by their support (which 4 coordinates are nonzero)
    # There are C(11,4) = 330 possible supports
    # For each support, there are 16 sign patterns (2^4)
    
    # Two vectors with the same support:
    #   dot = sum of products = 4 - 2*(number of sign differences)
    #   They conflict if dot > 2, i.e., 4 - 2*d > 2, i.e., d < 1, i.e., d = 0
    #   So only identical sign patterns conflict (dot=4)
    #   Wait: if d=0, dot=4 > 2: conflict
    #   If d=1, dot=2: not conflict
    #   If d=2, dot=0: not conflict
    #   etc.
    
    # Two vectors with different supports:
    #   Let s = number of shared coordinates
    #   On shared coords, let m = number with matching signs
    #   dot = m - (s-m) = 2m - s
    #   Conflict if 2m - s > 2
    
    # Precompute: for each pair of supports, the number of shared coordinates
    all_supports = list(itertools.combinations(range(d), 4))
    support_to_idx = {s: i for i, s in enumerate(all_supports)}
    n_supports = len(all_supports)
    
    # shared_count[i][j] = number of shared coordinates between support i and support j
    shared_count = np.zeros((n_supports, n_supports), dtype=np.int8)
    for i in range(n_supports):
        si = set(all_supports[i])
        for j in range(n_supports):
            sj = set(all_supports[j])
            shared_count[i][j] = len(si & sj)
    
    # Now build the vector list grouped by support
    # Each vector is (support_idx, sign_pattern) where sign_pattern is 0..15
    # Vector index = support_idx * 16 + sign_pattern
    
    # For each support s, precompute which other supports t can conflict
    # For supports s and t with shared_count = c:
    #   For sign patterns a (on s) and b (on t):
    #   m = number of shared coords where signs match
    #   Conflict if 2m - c > 2, i.e., m > (c+2)/2
    
    # Precompute conflict matrix between sign patterns for each pair of supports
    # This is complex; let's just compute the full adjacency using a batched approach
    
    # Alternative: use the fact that N4 = 15840 is manageable
    # Compute conflicts in chunks to limit memory
    chunk_size = 2000
    adj_bitsets = [0] * N4  # We'll use Python ints as bitsets (arbitrary precision)
    
    # Actually, for 15840 nodes, we can use a list of sets
    # But that's what the current code does. Let's try a different greedy.
    
    # Use a faster greedy: pick the vertex with minimum degree first (smallest last)
    # Or use a randomized greedy with multiple restarts but with numpy acceleration
    
    # Let me try: precompute the conflict matrix in chunks
    # Then use a greedy that processes vertices in a smart order
    
    # Precompute conflicts as a list of numpy arrays (one per vertex)
    # This uses ~15840 * avg_degree * 8 bytes
    # Avg degree might be ~50-100, so ~15840 * 75 * 8 = ~9.5 MB, fine
    
    conflict_adj = [None] * N4
    for start in range(0, N4, chunk_size):
        end = min(start + chunk_size, N4)
        Vchunk = V4[start:end]
        dots_chunk = Vchunk @ V4.T  # shape (chunk, N4)
        conf_chunk = dots_chunk > 2
        for i in range(start, end):
            conflict_adj[i] = np.where(conf_chunk[i - start])[0]
        del dots_chunk
        del conf_chunk
        if time.time() > deadline - 60:
            break
    
    # Verify all adjacencies are computed
    if any(adj is None for adj in conflict_adj):
        # Fallback: return base solution
        return best_points
    
    # 4. Greedy Search with Random Restarts
    # Use numpy-based active set management for speed
    best_indices = []
    
    n_restarts = 0
    while time.time() < deadline - 5:
        order = np.random.permutation(N4)
        
        # active_mask: boolean array, True if vertex is still available
        active_mask = np.ones(N4, dtype=bool)
        current_indices = []
        
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            # Remove v and all its neighbors from active
            active_mask[v] = False
            if len(conflict_adj[v]) > 0:
                active_mask[conflict_adj[v]] = False
                    
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            if len(best_indices) > best_k:
                new_points = V4[best_indices].astype(np.int64)
                best_points = new_points
                best_k = len(best_indices)
        
        n_restarts += 1
    
    # 5. Try to add Norm 2 vectors to the best Norm 4 solution
    # For 2-norm vs 4-norm: need dot <= 1 (||u-v||^2 >= 4 => 2+4-2dot >= 4 => dot <= 1)
    # For 2-norm vs 2-norm: need dot <= 0
    
    if best_indices:
        sel4_arr = V4[best_indices].astype(np.int64)
        sel2 = []
        sel2_arr = np.zeros((0, d), dtype=np.int64)
        
        for v in candidates_2:
            if time.time() > deadline - 2:
                break
            valid = True
            if len(sel4_arr) > 0:
                dots4 = sel4_arr @ v
                if np.max(dots4) > 1:
                    valid = False
            if valid and len(sel2) > 0:
                dots2 = sel2_arr @ v
                if np.max(dots2) > 0:
                    valid = False
            if valid:
                sel2.append(v)
                sel2_arr = np.vstack((sel2_arr, v.reshape(1, -1).astype(np.int64)))
        
        if sel2:
            all_points = list(V4[best_indices]) + sel2
            result = np.array(all_points, dtype=np.int64)
            # Verify the combined set is valid
            # Check min squared distance >= max squared norm
            norms_sq = np.sum(result * result, axis=1)
            max_norm_sq = np.max(norms_sq)
            
            min_dist_sq = np.inf
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    diff = result[i] - result[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
            
            if min_dist_sq >= max_norm_sq:
                best_points = result
                best_k = len(result)
    
    return np.around(best_points).astype(np.int64)

# EVOLVE-BLOCK-END
