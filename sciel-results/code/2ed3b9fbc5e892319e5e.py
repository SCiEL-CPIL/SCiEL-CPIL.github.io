import time
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
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
    
    # Precompute conflict graph
    # conflict[i] is a set of indices j such that block i and block j share a term
    num_blocks = len(blocks)
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if blocks[i]['terms'].intersection(blocks[j]['terms']):
                conflicts[i].add(j)
                conflicts[j].add(i)

    def evaluate_set(selected_indices):
        covered = set()
        for idx in selected_indices:
            covered.update(blocks[idx]['terms'])
        rank = 7 * len(selected_indices) + (n * m * p - len(covered))
        return rank, covered

    best_rank = 78
    best_config = None
    
    # Heuristic: Greedy with Minimum Degree
    # At each step, pick the block with the fewest conflicts among the remaining available blocks.
    # This tends to pick "isolated" blocks early, which might allow more blocks later.
    
    def greedy_min_degree(random_seed=0):
        rng = random.Random(random_seed)
        available = set(range(num_blocks))
        selected = []
        
        while available:
            # Find block in available with min degree within available
            # Degree of i in available is len(conflicts[i] & available)
            min_deg = float('inf')
            candidates = []
            for idx in available:
                deg = len(conflicts[idx].intersection(available))
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                elif deg == min_deg:
                    candidates.append(idx)
            
            if not candidates:
                break
                
            # Pick randomly from candidates to break ties and explore
            pick = rng.choice(candidates)
            selected.append(pick)
            
            # Remove pick and all its conflicts from available
            to_remove = {pick} | (conflicts[pick] & available)
            available -= to_remove
            
        return selected

    # Run greedy min degree with multiple restarts
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() - _t0 > _DEADLINE - 10:
            break
            
        selected = greedy_min_degree(r)
        rank, covered = evaluate_set(selected)
        
        if rank < best_rank:
            best_rank = rank
            best_config = selected
            # We can stop early if we reach a theoretical lower bound?
            # Max blocks N. Rank = 84 - N.
            # If we find N=10, rank=74.
            # Let's just keep searching for better.
            
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
