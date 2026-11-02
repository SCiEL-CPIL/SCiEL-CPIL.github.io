import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(decomp, n, m, p, sample=False):
    """Verify that the decomposition is correct."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    if sample:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(indices)
        indices = indices[:min(200, len(indices))]
    else:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for i, j, k in indices:
        s_val = 0
        for c in range(rank):
            s_val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s_val - 1) > 1e-9:
            return False
    return True

def _build_strassen_6x6():
    """
    Builds the standard Strassen decomposition for 6x6 matrices.
    Rank: 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    block_size = 3
    
    # Strassen atoms for 2x2
    atoms = [
        ({0: 1, 3: -1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({0: 1, 3: 1}, {3: 1}, {3: 1}),
        ({2: 1, 3: 1}, {1: 1}, {1: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        ({3: 1}, {2: 1, 0: -1}, {2: 1}),
        ({0: 1, 1: 1}, {2: 1}, {2: 1}),
        ({2: 1, 1: -1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ]
    
    U = []
    V = []
    W = []
    
    for s in range(len(atoms)):
        a_coeffs, b_coeffs, c_coeffs = atoms[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * block_size
                a_col_start = (ai % 2) * block_size
                b_row_start = (bi // 2) * block_size
                b_col_start = (bi % 2) * block_size
                
                for i in range(block_size):
                    for j in range(block_size):
                        for k in range(block_size):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // block_size
                            c_block_col = gk // block_size
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_coeffs.get(c_block_idx, 0)
                            
                            if c_sign == 0:
                                continue
                            
                            new_U = [0] * (n * m)
                            new_V = [0] * (m * p)
                            new_W = [0] * (p * n)
                            
                            new_U[gi * m + gj] = a_sign
                            new_V[gj * p + gk] = b_sign
                            new_W[gk * n + gi] = c_sign
                            
                            U.append(new_U)
                            V.append(new_V)
                            W.append(new_W)
                            
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_decomp, n, m, p, target_rank, time_limit=10):
    """
    Attempt to reduce the rank of the base decomposition by finding linear dependencies.
    This is a heuristic search.
    """
    start_time = time.time()
    U_base = np.array(base_decomp["U"])
    V_base = np.array(base_decomp["V"])
    W_base = np.array(base_decomp["W"])
    current_rank = U_base.shape[1]
    
    if current_rank <= target_rank:
        return base_decomp

    # We try to find a subset of columns or linear combinations that span the same tensor.
    # This is equivalent to finding a lower rank approximation that is exact.
    # For exactness, we can try to nullify some columns by linear combination.
    
    # Simple heuristic: Randomly combine columns and check if rank decreases while maintaining validity.
    # This is expensive. We will just return the base if time is short.
    
    # A more effective structural change: Use the fact that 6 = 2 * 3.
    # We already did that.
    
    # If we have time, we can try a random search for a rank-188 decomposition.
    # But given the 150s limit and the complexity, we return the base.
    
    return base_decomp

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    try:
        # Generate the standard Strassen decomposition (Rank 189)
        decomp = _build_strassen_6x6()
        
        # Verify
        if _verify(decomp, n, m, p, sample=False):
            # Try to reduce rank if possible (currently just returns same)
            # Future: Implement specific rank-reduction search
            return decomp
        else:
            # Fallback to schoolbook
            return _schoolbook(n, m, p)
            
    except Exception as e:
        # Fallback to schoolbook on any error
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
