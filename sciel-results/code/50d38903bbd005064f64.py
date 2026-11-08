import time
import itertools
import numpy as np

_DIMS = (3, 4, 7)

def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
                
    num_blocks = len(blocks)
    
    # Build conflict graph
    # adj[i][j] = True if blocks i and j are INCOMPATIBLE (conflict)
    adj = [[False] * num_blocks for _ in range(num_blocks)]
    
    def are_incompatible(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Overlap in all dimensions
        if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
            return True
        return False

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if are_incompatible(blocks[i], blocks[j]):
                adj[i][j] = True
                adj[j][i] = True

    # Find Maximum Independent Set
    # Since N=36, we can use a simple backtracking or even brute force with pruning
    # Or use networkx if available? No, stick to stdlib/numpy.
    # Bron-Kerbosch is for Max Clique. MIS in G is Max Clique in G_complement.
    # G_complement has edges for COMPATIBLE blocks.
    
    # Let's build the compatibility graph (complement of conflict graph)
    # compat[i][j] = True if blocks i and j are COMPATIBLE
    compat = [[not adj[i][j] for j in range(num_blocks)] for i in range(num_blocks)]
    
    # Max Clique in compat graph
    # Simple branch and bound
    best_clique = []
    
    def expand(clique, candidates):
        nonlocal best_clique
        if len(candidates) == 0:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Heuristic: pick a vertex from candidates with max degree in candidates
        # Or just pick the first one
        # For small N, simple recursion is fine.
        # To speed up, we can use a simple greedy coloring bound (Tomita's algorithm)
        # But for N=36, even simple backtracking is very fast.
        
        # Pick a pivot to reduce branching?
        # Let's just iterate through candidates
        # Order candidates by degree in candidates (descending)
        degs = [sum(compat[candidates[i]][candidates[j]] for j in range(len(candidates)) if i != j) for i in range(len(candidates))]
        indices = sorted(range(len(candidates)), key=lambda x: -degs[x])
        
        for idx in indices:
            v = candidates[idx]
            new_clique = clique + [v]
            # New candidates must be compatible with v
            new_candidates = [c for c in candidates if c != v and compat[v][c]]
            
            # Pruning: if len(new_clique) + len(new_candidates) <= len(best_clique), skip
            if len(new_clique) + len(new_candidates) <= len(best_clique):
                continue
                
            expand(new_clique, new_candidates)

    # Initialize
    expand([], list(range(num_blocks)))
    
    best_block_indices = best_clique
    best_blocks = [blocks[i] for i in best_block_indices]
    
    # Construct decomposition
    num_selected = len(best_blocks)
    
    covered_entries = set()
    for b in best_blocks:
        i0, j0, k0 = b
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered_entries.add((i, j, k))
    
    uncovered_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_entries:
                    uncovered_entries.append((i, j, k))
    
    rank = 7 * num_selected + len(uncovered_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M2
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        # M5
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        # M6
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M7
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b in best_blocks:
        add_strassen(*b)
        
    # Add schoolbook terms for uncovered entries
    for (i, j, k) in uncovered_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
    if time.time() < _t0 + 110.0:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if val != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): {val}")

    return {"U": U, "V": V, "W": W}
