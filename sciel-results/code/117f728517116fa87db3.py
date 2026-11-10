import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = P['n']
    m = P['m']
    total = 1 << n  # 4096
    
    # Precompute comparability: comp[i][j] = 1 if i is strict subset of j or vice versa
    # Two masks a, b are comparable if (a & b) == a or (a & b) == b
    # Precompute for all pairs - use numpy for speed
    masks = np.arange(total, dtype=np.int64)
    
    # Build comparability matrix using broadcasting
    # comp[i,j] = 1 if (masks[i] & masks[j]) == masks[i] or (masks[i] & masks[j]) == masks[j]
    # This is expensive for 4096x4096, but doable with numpy
    # Actually, let's think smarter: for each pair, check subset relation
    
    # Precompute: for each mask, which other masks are supersets/subsets
    # Use the fact that a is subset of b iff (a | b) == b iff (a & ~b) == 0
    
    # Let's precompute a dense matrix using numpy
    # a_ij = (masks[i] & masks[j]) == masks[i]  (i subset of j)
    # b_ij = (masks[i] & masks[j]) == masks[j]  (j subset of i)
    # comparable = a_ij | b_ij, excluding i==j
    
    # For memory efficiency, let's use a different approach
    # Actually 4096*4096 = 16M entries, manageable as uint8
    
    # Compute subset relation matrix
    # S[i,j] = 1 if masks[i] is subset of masks[j]
    # S[i,j] = 1 if (masks[i] & masks[j]) == masks[i]
    
    # Using numpy broadcasting would create 4096*4096 array
    # Let's do it in chunks or use a smarter method
    
    # Alternative: for each mask i, the supersets are all masks j where j | i == j
    # i.e., j has all bits of i plus possibly more
    # Number of supersets of i is 2^(n - popcount(i))
    
    # Let's precompute popcount
    popcount = np.array([bin(x).count('1') for x in range(total)], dtype=np.int32)
    
    # For the QUBO/SA approach, I need to compute the objective efficiently
    # Let x be a binary vector of length total, with sum(x) = m
    # Objective: sum_{i<j} x[i]*x[j]*comparable[i][j]
    # = 0.5 * (sum_{i,j} x[i]*x[j]*comparable[i][j] - sum_i x[i]*comparable[i][i])
    # But comparable[i][i] = 0 (strict subset), so:
    # = 0.5 * sum_{i,j} x[i]*x[j]*comparable[i][j]
    
    # For SA, I need to evaluate the change when flipping x[k]
    # delta = sum_{j != k} x[j] * comparable[k][j]
    # (with sign depending on whether we're adding or removing k)
    
    # Let's precompute the comparable matrix as a sparse-like structure
    # Actually, let's just build the full matrix since 4096^2 = 16M is fine
    
    # Build comparable matrix
    # comp[i,j] = 1 if i is strict subset of j or j is strict subset of i
    # i strict subset of j: (i & j) == i and i != j
    # j strict subset of i: (i & j) == j and i != j
    
    # Using numpy:
    # A = (masks[:,None] & masks[None,:]) == masks[:,None]  # i subset of j
    # B = (masks[:,None] & masks[None,:]) == masks[None,:]  # j subset of i
    # comp = (A | B) & (masks[:,None] != masks[None,:])
    
    # This creates 4096*4096 arrays, which is ~16M * 8 bytes = 128MB per array
    # 3 arrays = 384MB, might be tight but should work
    
    # Let's be more memory efficient: compute row by row or use int32
    # Actually, let's use a smarter approach: precompute for each mask, the list of comparable masks
    
    # For SA, I'll maintain:
    # - x: binary array of length total
    # - degree[k] = sum_{j} x[j] * comp[k][j] for current x
    # When flipping k:
    #   if adding k: delta = degree[k]
    #   if removing k: delta = -degree[k]
    #   then update degree for all j where comp[k][j]=1
    
    # Let's precompute for each k, the list of j where comp[k][j] = 1
    # And also the count of such j
    
    # First, let's build the comparable matrix compactly
    # comp[k] is a bitmask or list of comparable indices
    
    # For efficiency, let's precompute:
    # supersets[k] = list of j > k such that k is subset of j (strict)
    # subsets[k] = list of j < k such that j is subset of k (strict)
    # comparable[k] = supersets[k] + subsets[k]
    
    # But for SA, I need to update degree efficiently
    # Let me use a different approach: precompute the full comp matrix as uint8
    
    # Memory check: 4096*4096 = 16,777,216 bytes = 16 MB as uint8. Fine!
    
    # Build comp matrix
    masks_arr = masks.astype(np.uint32)
    # For each pair, check subset relation
    # Do it in blocks to avoid creating huge temporary arrays
    comp = np.zeros((total, total), dtype=np.uint8)
    
    # Block size for memory
    block_size = 512
    for i_start in range(0, total, block_size):
        i_end = min(i_start + block_size, total)
        block = masks_arr[i_start:i_end]  # shape (block_len, )
        # Compare with all j
        # a = (block[:,None] & masks_arr[None,:]) == block[:,None]  # block subset of j
        # b = (block[:,None] & masks_arr[None,:]) == masks_arr[None,:]  # j subset of block
        # This creates (block_len, total) arrays
        a = (block[:, None] & masks_arr[None, :]) == block[:, None]
        b = (block[:, None] & masks_arr[None, :]) == masks_arr[None, :]
        # Strict: exclude diagonal
        diag = np.eye(i_end - i_start, total, dtype=bool)
        comp[i_start:i_end, :] = ((a | b) & ~diag).astype(np.uint8)
    
    # Now run simulated annealing
    # State: x (binary vector of length total), with sum(x) = m
    
    def evaluate(x):
        """Compute number of comparable pairs in the selected set."""
        selected = np.where(x == 1)[0]
        if len(selected) < 2:
            return 0
        # Get submatrix
        sub = comp[np.ix_(selected, selected)]
        # Count upper triangle (to avoid double counting)
        return int(np.sum(np.triu(sub, k=1)))
    
    # Simulated annealing
    best_x = None
    best_score = -1
    
    # Initial solution: greedy or random
    # Let's start with a greedy approach for good initial solution
    
    # Greedy: start with empty, add set that maximizes marginal gain
    # Or start with all sets of small cardinality
    
    # Better initial: take the m sets with smallest cardinality, breaking ties randomly
    # This tends to create many comparable pairs
    
    # Let's try multiple initializations
    
    def greedy_init():
        """Greedy initialization: add sets one by one, choosing the one with max marginal gain."""
        x = np.zeros(total, dtype=np.uint8)
        degree = np.zeros(total, dtype=np.int32)
        
        for _ in range(m):
            # Find best candidate: max degree[k] among k not in x
            # degree[k] = sum_{j in x} comp[k][j]
            candidates = np.where(x == 0)[0]
            if len(candidates) == 0:
                break
            # Get degrees for candidates
            cand_degrees = degree[candidates]
            best_idx = candidates[np.argmax(cand_degrees)]
            x[best_idx] = 1
            # Update degrees for all j where comp[best_idx][j] = 1
            neighbors = np.where(comp[best_idx] == 1)[0]
            degree[neighbors] += 1
        
        return x
    
    def random_init():
        """Random initialization."""
        x = np.zeros(total, dtype=np.uint8)
        chosen = np.random.choice(total, size=m, replace=False)
        x[chosen] = 1
        return x
    
    def small_cardinality_init():
        """Initialize with smallest cardinality sets."""
        # Sort masks by popcount
        order = np.argsort(popcount)
        x = np.zeros(total, dtype=np.uint8)
        x[order[:m]] = 1
        return x
    
    # Run SA
    def simulated_annealing(x0, t_max=100000, t_min=0.01, alpha=0.9999):
        """Run simulated annealing starting from x0."""
        x = x0.copy()
        # Compute initial degree
        degree = np.zeros(total, dtype=np.int32)
        selected = np.where(x == 1)[0]
        for k in selected:
            neighbors = np.where(comp[k] == 1)[0]
            degree[neighbors] += 1
        
        current_score = evaluate(x)
        best_x_local = x.copy()
        best_score_local = current_score
        
        t = t_max
        step = 0
        no_improve = 0
        
        while t > t_min and no_improve < 50000:
            # Pick a random pair to flip: add one, remove one (to maintain cardinality)
            selected = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            
            if len(selected) == 0 or len(unselected) == 0:
                break
            
            k_add = unselected[np.random.randint(len(unselected))]
            k_remove = selected[np.random.randint(len(selected))]
            
            if k_add == k_remove:
                continue
            
            # Compute delta: change in score when adding k_add and removing k_remove
            # New score = old_score + gain_add + gain_remove + interaction
            # gain_add = sum_{j in selected, j != k_remove} comp[k_add][j]
            #          = degree[k_add] - comp[k_add][k_remove]  (since k_remove is in selected)
            # gain_remove = -sum_{j in selected, j != k_remove} comp[k_remove][j]
            #             = -(degree[k_remove] - comp[k_remove][k_add])
            # interaction: the pair (k_add, k_remove) is no longer both selected, 
            # but we're swapping, so we need to be careful
            # 
            # Actually, let me think again:
            # Current selected set S, with k_remove in S
            # New set S' = S - {k_remove} + {k_add}
            # 
            # Pairs in S': 
            #   - Pairs within S \ {k_remove}: same as before
            #   - Pairs involving k_add and (S \ {k_remove}): new
            #   - Pair (k_add, k_remove): not in S'
            #   - Pairs involving k_remove and others: removed
            #
            # Delta = [sum_{j in S\{k_remove}} comp[k_add][j]] - [sum_{j in S\{k_remove}} comp[k_remove][j]]
            #       = (degree[k_add] - comp[k_add][k_remove]) - (degree[k_remove] - comp[k_remove][k_add])
            #       = degree[k_add] - degree[k_remove]  (since comp is symmetric)
            
            delta = int(degree[k_add]) - int(degree[k_remove])
            
            # Accept?
            if delta >= 0 or random.random() < np.exp(delta / t):
                # Apply swap
                x[k_remove] = 0
                x[k_add] = 1
                
                # Update degrees
                # For all j where comp[k_remove][j] = 1: degree[j] -= 1
                # For all j where comp[k_add][j] = 1: degree[j] += 1
                neighbors_remove = np.where(comp[k_remove] == 1)[0]
                degree[neighbors_remove] -= 1
                neighbors_add = np.where(comp[k_add] == 1)[0]
                degree[neighbors_add] += 1
                
                current_score += delta
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_x_local = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            
            t *= alpha
            step += 1
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return best_x_local, best_score_local
    
    # Try multiple initializations and SA runs
    inits = [
        ("greedy", greedy_init()),
        ("small_card", small_cardinality_init()),
        ("random1", random_init()),
        ("random2", random_init()),
    ]
    
    # Evaluate initial solutions
    for name, x0 in inits:
        if time.time() > deadline - 10:
            break
        score = evaluate(x0)
        if score > best_score:
            best_score = score
            best_x = x0.copy()
    
    # Run SA from each initialization
    for name, x0 in inits:
        if time.time() > deadline - 15:
            break
        x_best, score = simulated_annealing(x0, t_max=50000, t_min=0.01, alpha=0.99995)
        if score > best_score:
            best_score = score
            best_x = x_best
    
    # Additional SA restarts from best solution if time permits
    if best_x is not None and time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb best solution
            x_pert = best_x.copy()
            selected = np.where(x_pert == 1)[0]
            unselected = np.where(x_pert == 0)[0]
            # Swap 5 random pairs
            for _ in range(5):
                if len(selected) == 0 or len(unselected) == 0:
                    break
                k_r = selected[np.random.randint(len(selected))]
                k_a = unselected[np.random.randint(len(unselected))]
                x_pert[k_r] = 0
                x_pert[k_a] = 1
                selected = np.where(x_pert == 1)[0]
                unselected = np.where(x_pert == 0)[0]
            
            x_best, score = simulated_annealing(x_pert, t_max=30000, t_min=0.01, alpha=0.99995)
            if score > best_score:
                best_score = score
                best_x = x_best
    
    # Final: convert best_x to list of lists
    if best_x is None:
        # Fallback
        best_x = greedy_init()
    
    selected = np.where(best_x == 1)[0]
    result = []
    for mask in selected:
        s = [b for b in range(n) if (mask >> b) & 1]
        result.append(s)
    
    # Ensure exactly m sets
    assert len(result) == m, f"Expected {m} sets, got {len(result)}"
    # Ensure distinct
    assert len(set(tuple(s) for s in result)) == m, "Sets are not distinct"
    # Ensure types are correct
    for s in result:
        assert isinstance(s, list), f"Set is not a list: {type(s)}"
        for v in s:
            assert isinstance(v, int) and not isinstance(v, bool), f"Value {v} is not a proper int"
    
    return result
