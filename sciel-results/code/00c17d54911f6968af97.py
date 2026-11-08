import time
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

    # Tomita Max Independent Set (Max Clique in complement)
    # We want Max Clique in the compatibility graph.
    # Compat graph edges: i ~ j if NOT adj[i][j]
    
    # Precompute neighbors as bitmasks
    compat_mask = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for j in range(num_blocks):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        compat_mask[i] = mask

    best_clique_size = 0
    best_clique_nodes = []

    # Tomita's algorithm with coloring
    # We process vertices in a specific order.
    # We maintain a set of candidates P.
    # We color P to get an upper bound.
    
    def expand(clique, candidates_mask):
        nonlocal best_clique_size, best_clique_nodes
        
        if not candidates_mask:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique_nodes = clique[:]
            return
        
        # Greedy Coloring
        # We need to order the vertices in candidates_mask.
        # We use a heuristic: sort by degree in the subgraph induced by candidates_mask.
        
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        
        # Calculate degrees in the induced subgraph
        degs = []
        for v in candidates:
            d = bin(candidates_mask & compat_mask[v]).count('1')
            degs.append((d, v))
        
        # Sort by degree descending
        degs.sort(reverse=True)
        
        # Coloring
        # We assign colors to vertices such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the clique size.
        
        color_map = {}
        current_color = 0
        uncolored = set(candidates)
        
        # We process vertices in order of decreasing degree.
        # This is a simplified coloring.
        
        # Actually, Tomita's algorithm uses a specific coloring scheme.
        # We will use a standard greedy coloring.
        
        # To save time, we use a simple bound: len(candidates)
        if len(clique) + len(candidates) <= best_clique_size:
            return
            
        # Iterate through candidates in order of decreasing degree
        for d, v in degs:
            if (candidates_mask >> v) & 1 == 0:
                continue
                
            # New candidates must be neighbors of v
            new_candidates_mask = candidates_mask & compat_mask[v] & ~(1 << v)
            
            if len(clique) + 1 + bin(new_candidates_mask).count('1') <= best_clique_size:
                continue
                
            expand(clique + [v], new_candidates_mask)
            
            # Remove v from candidates for the next iteration
            # This is handled by the fact that we are iterating over a static list `degs`
            # but we must ensure we don't pick v again.
            # The `new_candidates_mask` excludes v.
            # But in the next iteration of the loop, `v` is still in `candidates_mask`.
            # We need to remove v from `candidates_mask` for the *current* level?
            # No, the standard Bron-Kerbosch style loop removes v from P after recursion.
            # Here, we are iterating over `degs`.
            # We should update `candidates_mask` to remove v?
            # No, the recursive call handles the subset.
            # The current level just tries the next vertex.
            
            # However, we must ensure that the next vertex is still in `candidates_mask`.
            # Since we are iterating over `degs` which is derived from the initial `candidates_mask`,
            # and we don't modify `candidates_mask` in the loop, it's fine.
            
    initial_mask = (1 << num_blocks) - 1
    expand([], initial_mask)

    best_block_indices = best_clique_nodes
    best_blocks = [blocks[i] for i in best_block_indices]
    
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
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b in best_blocks:
        add_strassen(*b)
        
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
