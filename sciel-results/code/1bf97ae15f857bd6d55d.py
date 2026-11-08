import time
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Generate all possible 2x2x2 Strassen blocks
    # Block defined by row pair (i0, i1), col pair (j0, j1), depth pair (k0, k1)
    # with i0 < i1, j0 < j1, k0 < k1
    
    blocks = []
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            # Terms covered:
                            # (i0, j0, k0), (i0, j0, k1), (i0, j1, k0), (i0, j1, k1)
                            # (i1, j0, k0), (i1, j0, k1), (i1, j1, k0), (i1, j1, k1)
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
    
    # Find a set of disjoint blocks
    # We want to maximize the number of blocks in the set, such that no two blocks share a term.
    # This is equivalent to finding a large independent set in the conflict graph.
    
    best_rank = 78
    best_config = None
    
    # Initial guess: The fixed set from the original program
    # (0,1,0,1,0,1), (0,1,0,1,2,3), (0,1,0,1,4,5), (0,1,2,3,0,1), (0,1,2,3,2,3), (0,1,2,3,4,5)
    # Let's just use a greedy algorithm with random restarts to find a better packing.
    
    def evaluate_set(selected_blocks):
        covered = set()
        for b in selected_blocks:
            covered.update(b['terms'])
        rank = 7 * len(selected_blocks) + (n * m * p - len(covered))
        return rank, covered
    
    def greedy_set_packing(random_seed=0):
        rng = random.Random(random_seed)
        remaining = list(range(len(blocks)))
        selected = []
        covered = set()
        
        while remaining:
            # Pick a random block from remaining? Or one with fewest conflicts?
            # Let's pick randomly to explore different packings
            idx = rng.choice(remaining)
            b = blocks[idx]
            
            # Check if it conflicts with already selected? 
            # The "remaining" list should only contain blocks disjoint from 'covered'
            # So we need to filter 'remaining' before picking.
            
            # Optimization: Filter remaining to only those disjoint from covered
            new_remaining = []
            for r_idx in remaining:
                if blocks[r_idx]['terms'].isdisjoint(covered):
                    new_remaining.append(r_idx)
            
            if not new_remaining:
                break
                
            # Pick one from new_remaining
            # Heuristic: Pick one that has fewest conflicts with other candidates in new_remaining?
            # Or just random.
            pick_idx = rng.choice(new_remaining)
            b = blocks[pick_idx]
            
            selected.append(b)
            covered.update(b['terms'])
            
            # Update remaining: remove pick_idx and all that conflict with b
            # Actually, we just rebuilt new_remaining. We can just set remaining = new_remaining 
            # and remove pick_idx? No, new_remaining contains pick_idx.
            # And new_remaining contains blocks disjoint from covered (which now includes b).
            # So we can just set remaining = [x for x in new_remaining if x != pick_idx]
            remaining = [x for x in new_remaining if x != pick_idx]
            
        return selected

    # Run greedy set packing with multiple restarts
    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() - _t0 > _DEADLINE - 10:
            break
            
        selected = greedy_set_packing(r)
        rank, covered = evaluate_set(selected)
        
        if rank < best_rank:
            best_rank = rank
            best_config = selected
            # Print for debugging? No.
            
    # If we didn't find anything better, use the original fixed set
    if best_config is None:
        # Reconstruct original fixed set indices
        fixed_indices = []
        fixed_tuples = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]
        for ft in fixed_tuples:
            i0, i1, j0, j1, k0, k1 = ft
            for idx, b in enumerate(blocks):
                if b['i0']==i0 and b['i1']==i1 and b['j0']==j0 and b['j1']==j1 and b['k0']==k0 and b['k1']==k1:
                    fixed_indices.append(idx)
                    break
        best_config = [blocks[i] for i in fixed_indices]
        best_rank, _ = evaluate_set(best_config)

    # Construct matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    covered_terms = set()
    
    for b in best_config:
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
