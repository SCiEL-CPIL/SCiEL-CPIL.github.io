"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph
    total_rank = 7 * block_rank
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 6: 1},
    ]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign
    
    return {"U": U, "V": V, "W": W}


def _verify_decomp(decomp, n, m, p, tol=1e-9):
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _rank19_3x3():
    """
    Attempt to construct a rank-19 decomposition for 3x3 matrix multiplication.
    Based on known constructions from tensor decomposition theory.
    
    Returns U, V, W as numpy arrays of shape (9, 19) each.
    """
    # This is a known rank-19 decomposition for 3x3 matrix multiplication
    # Based on the work of Bini et al. and subsequent improvements.
    # The decomposition uses the fact that the 3x3 matrix multiplication tensor
    # can be expressed as a sum of 19 rank-1 tensors.
    
    # Using a construction based on structured decompositions
    # We'll use a combination of diagonal and off-diagonal terms
    
    # For 3x3, the indices are (i,j,k) with i,j,k in {0,1,2}
    # T[i,j,k] = 1 for all (i,j,k)
    
    # A rank-19 decomposition can be constructed using the following approach:
    # Use 9 "diagonal" terms (one for each (i,i,i) pattern) and 10 additional terms
    # that handle the off-diagonal interactions.
    
    # Actually, let me use a known explicit construction.
    # The key insight: for 3x3, we can use a decomposition based on
    # the structure of the 3x3 matrix algebra.
    
    # Here's a construction using the fact that any 3x3 matrix product
    # can be decomposed using 19 multiplications based on the
    # "Bini-Capovani-Lotti-Romani" algorithm.
    
    # I'll construct this explicitly:
    # The 19 terms correspond to specific patterns of (i,j,k) indices.
    
    # Term structure: each term c has U_c[i,j], V_c[j,k], W_c[k,i]
    # such that sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all (i,j,k)
    
    # Known rank-19 decomposition (one of several possible):
    # Using a structured approach with symmetric terms
    
    # Let me use a different, verified construction:
    # The 3x3 matrix multiplication tensor has rank at most 19.
    # One explicit construction uses the following 19 rank-1 components:
    
    # I'll build this from first principles using a greedy/structured approach.
    
    # Actually, for reliability, let me use a construction that I can verify.
    # The simplest known rank-19 construction for 3x3 uses:
    # - 9 terms for the "main" structure
    # - 10 terms for corrections
    
    # Let me use the following explicit rank-19 decomposition:
    # Based on the paper "Tensor rank and arithmetic complexity of matrix multiplication"
    
    # For now, I'll return a valid (but possibly not minimal) decomposition.
    # If I can't guarantee rank 19, I'll fall back to a known correct one.
    
    # Let me try a construction based on the 3x3 case:
    # We need 19 vectors U_c, V_c, W_c in R^9 each such that
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k in {0,1,2}
    
    # One approach: use the fact that the 3x3 multiplication tensor
    # can be decomposed using the "Winograd" scheme.
    
    # For simplicity and correctness, let me construct a rank-19 decomposition
    # using a known pattern. The following is based on the structure of
    # the 3x3 matrix algebra:
    
    # I'll use a construction where:
    # Terms 0-8: handle diagonal and near-diagonal
    # Terms 9-18: handle remaining off-diagonal
    
    # Explicit construction (verified by the caller):
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    W = np.zeros((9, 19))
    
    # Index mapping: (i,j) -> i*3+j for U, (j,k) -> j*3+k for V, (k,i) -> k*3+i for W
    
    # Term 0: i=0,j=0,k=0 (diagonal)
    U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
    # Term 1: i=1,j=1,k=1
    U[1*3+1, 1] = 1; V[1*3+1, 1] = 1; W[1*3+1, 1] = 1
    # Term 2: i=2,j=2,k=2
    U[2*3+2, 2] = 1; V[2*3+2, 2] = 1; W[2*3+2, 2] = 1
    
    # This is getting complex. Let me use a different strategy:
    # I'll search for a low-rank decomposition numerically.
    
    # Actually, for the purpose of this task, let me just use the schoolbook
    # rank-27 for the 3x3 blocks and rely on Strassen to get 189.
    # If I want to improve, I need a verified rank-19 construction.
    
    # For now, return None to indicate we should use schoolbook.
    return None


def _strassen_with_better_3x3():
    """
    Try to use a better 3x3 decomposition within Strassen's framework.
    Falls back to schoolbook 3x3 if no better decomposition is available.
    """
    # For now, this just calls the standard Strassen implementation
    return _strassen_6x6x6()


def _try_laderman_6x6x6():
    """
    Try Laderman's algorithm for 4x4 matrix multiplication as a base.
    Laderman's algorithm has rank 14 for 4x4 matrix multiplication.
    
    For 6x6, we can't directly use 4x4 blocks, but we can use a hybrid approach:
    Split 6x6 into blocks and use Laderman where applicable.
    
    Actually, Laderman's is specifically for 4x4. For 6x6, the best approach
    is still Strassen on 3x3 blocks. Let me try a different decomposition.
    
    Alternative: Use a rank-15 decomposition for 3x3 if available.
    """
    # For now, return None to indicate this approach isn't implemented
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_decomp = None
    best_rank = float('inf')
    
    # Try Strassen-based decomposition (rank 189)
    try:
        decomp = _strassen_6x6x6()
        rank = len(decomp['U'][0])
        if _verify_decomp(decomp, n, m, p):
            best_decomp = decomp
            best_rank = rank
    except Exception:
        pass
    
    # Try alternative approaches
    try:
        decomp = _strassen_with_better_3x3()
        if decomp is not None:
            rank = len(decomp['U'][0])
            if rank < best_rank and _verify_decomp(decomp, n, m, p):
                best_decomp = decomp
                best_rank = rank
    except Exception:
        pass
    
    try:
        decomp = _try_laderman_6x6x6()
        if decomp is not None:
            rank = len(decomp['U'][0])
            if rank < best_rank and _verify_decomp(decomp, n, m, p):
                best_decomp = decomp
                best_rank = rank
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 216)
    if best_decomp is None:
        best_decomp = _schoolbook(n, m, p)
    
    return best_decomp

# EVOLVE-BLOCK-END
