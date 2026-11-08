import time
import random
import sys

DIMENSIONS = [3, 4, 7]
_DEADLINE = 145.0 # Safety margin for 150s limit

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
    block_terms = []
    
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            terms = set()
                            for i in (i0, i1):
                                for j in (j0, j1):
                                    for k in (k0, k1):
                                        terms.add((i, j, k))
                            blocks.append({
                                'i0': i0, 'i1': i1,
                                'j0': j0, 'j1': j1,
                                'k0': k0, 'k1': k1,
                                'terms': terms
                            })
                            block_terms.append(terms)
    
    num_blocks = len(blocks)
    
    # Precompute conflict matrix (bitmask or set of indices)
    # Since num_blocks is ~378, a set of conflicts is fine.
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        ti = block_terms[i]
        for j in range(i + 1, num_blocks):
            tj = block_terms[j]
            if not ti.isdisjoint(tj):
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    best_rank = 84 # Trivial rank
    best_config = []
    
    # Greedy Set Packing with Local Search
    def greedy_search(seed):
        rng = random.Random(seed)
        available = list(range(num_blocks))
        selected = []
        covered = set()
        
        # Heuristic: Pick block with fewest conflicts among available
        while available:
            # Find block with minimum conflicts in 'available'
            min_conf = float('inf')
            best_idx = -1
            # To speed up, we can just pick a few candidates
            # For 378 blocks, full scan is cheap
            for idx in available:
                # Count conflicts with other available blocks
                # Note: 'available' changes as we select, but conflicts are static
                # We want to pick a block that doesn't "waste" too many other blocks
                # Approximation: pick one with fewest total conflicts in available set
                # This requires checking intersection of conflicts[idx] and available
                # This is O(N) per pick, O(N^2) total. N=378, N^2 ~ 140k ops. Very fast.
                conf_count = 0
                for c in conflicts[idx]:
                    if c in available: # 'available' is a list, so 'in' is O(N). 
                        # Optimization: use a set for available
                        pass
                # Let's use a set for available for O(1) lookups
                # We'll maintain 'available_set'
                pass
            
            # Let's rewrite the loop to be efficient
            # We'll maintain a set of available indices
            break # Placeholder, will implement properly below

    # Better Greedy Implementation
    def greedy_search_v2(seed):
        rng = random.Random(seed)
        available_set = set(range(num_blocks))
        selected = []
        covered = set()
        
        while available_set:
            # Pick a block. Strategy: Random among those with fewest conflicts in available_set
            # Calculate conflict count for each in available_set
            min_conf = float('inf')
            candidates = []
            for idx in available_set:
                # Count how many of its conflicts are also in available_set
                # This determines how many blocks it "kills"
                kill_count = 0
                for c in conflicts[idx]:
                    if c in available_set:
                        kill_count += 1
                
                if kill_count < min_conf:
                    min_conf = kill_count
                    candidates = [idx]
                elif kill_count == min_conf:
                    candidates.append(idx)
            
            # Pick randomly from best candidates to allow exploration
            if not candidates:
                break
            pick_idx = rng.choice(candidates)
            
            selected.append(pick_idx)
            available_set.remove(pick_idx)
            
            # Remove all blocks that conflict with pick_idx
            to_remove = conflicts[pick_idx] & available_set
            available_set.difference_update(to_remove)
            
        return selected

    # Local Search: Try to improve a configuration
    def local_search(config, seed):
        rng = random.Random(seed)
        config_set = set(config)
        covered = set()
        for idx in config:
            covered.update(block_terms[idx])
        
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            # Try to swap a block in config with a block not in config
            # Or just try to add a block if we remove one and can add two?
            # Simpler: Randomly remove a block, then try to add back as many as possible greedily?
            # Let's try: Pick a random block from config, remove it. 
            # Then run greedy on the remaining available (including the removed one's neighbors)
            # If the new set is larger, accept.
            
            if not config:
                break
                
            idx_to_remove = rng.choice(config)
            config_set.remove(idx_to_remove)
            
            # Rebuild available set based on current config_set
            # Available are blocks NOT in config_set AND NOT conflicting with config_set
            # Actually, we just need to check if we can add blocks to the remaining config.
            # Let's restart greedy from the state of (config - {idx_to_remove})
            
            base_config = list(config_set)
            base_covered = set()
            for idx in base_config:
                base_covered.update(block_terms[idx])
                
            # Find all blocks disjoint from base_covered
            candidates = []
            for i in range(num_blocks):
                if i in base_config:
                    continue
                if block_terms[i].isdisjoint(base_covered):
                    candidates.append(i)
            
            # Greedily add from candidates
            new_config = list(base_config)
            new_covered = set(base_covered)
            cand_set = set(candidates)
            
            while cand_set:
                # Pick block with fewest conflicts in cand_set
                min_conf = float('inf')
                best_cand = -1
                for idx in cand_set:
                    kill_count = 0
                    for c in conflicts[idx]:
                        if c in cand_set:
                            kill_count += 1
                    if kill_count < min_conf:
                        min_conf = kill_count
                        best_cand = idx
                        # Can't break early if we want random among ties, but for speed let's just take first best
                        # Or collect ties
                if best_cand == -1:
                    break
                new_config.append(best_cand)
                new_covered.update(block_terms[best_cand])
                cand_set.remove(best_cand)
                to_remove = conflicts[best_cand] & cand_set
                cand_set.difference_update(to_remove)
                
            if len(new_config) > len(config):
                config = new_config
                config_set = set(config)
                improved = True
        
        return config

    # Main Search Loop
    num_restarts = 0
    while time.time() - _t0 < _DEADLINE - 5:
        seed = num_restarts
        initial_config = greedy_search_v2(seed)
        
        # Local search improvement
        improved_config = local_search(initial_config, seed)
        
        # Calculate rank
        K = len(improved_config)
        rank = 84 - K
        
        if rank < best_rank:
            best_rank = rank
            best_config = improved_config
            # Print(f"Found rank {best_rank} with {K} blocks")
            
        num_restarts += 1
        # If we hit the theoretical max for this heuristic or time is up, break
        if best_rank == 74: # Max possible with 10 blocks
            break

    # If no blocks found (shouldn't happen), fallback to trivial
    if not best_config:
        best_rank = 84
        best_config = []

    # Construct matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    covered_terms = set()
    
    for idx in best_config:
        b = blocks[idx]
        i0, i1 = b['i0'], b['i1']
        j0, j1 = b['j0'], b['j1']
        k0, k1 = b['k0'], b['k1']
        
        # M1
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        covered_terms.update([(i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
                              (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)])
        col += 1
        
        # M2
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1
        
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M4
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # M6
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M7
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    # Fill remaining terms with trivial rank-1 tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered_terms:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k})")

    return {"U": U, "V": V, "W": W}
