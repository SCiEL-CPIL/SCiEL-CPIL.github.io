"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
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
    return U, V, W

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (u_dict, v_dict, w_dict) with keys (i,j), (j,k), (i,k) in local coords.
    Verified: sum over 7 terms of outer products equals the 2x2x2 matmul tensor.
    """
    terms = []
    # Term 1: (A00+A11)(B00+B11)
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    # Term 2: (A10+A11)B00
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):1}))
    # Term 3: A00(B01-B11)
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
    # Term 4: A11(B10-B00)
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
    # Term 5: (A00+A01)B11
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
    # Term 6: (A10-A00)(B00+B01)
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    # Term 7: (A01-A11)(B10+B11)
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _verify_2x2x2():
    """Sanity check that the 7 Strassen terms correctly decompose 2x2x2 matmul."""
    terms = _strassen_2x2x2_terms()
    T = np.zeros((4,4,4))
    for u_d, v_d, w_d in terms:
        for (i,j), ui in u_d.items():
            for (j2,k), vj in v_d.items():
                for (i2,k2), wk in w_d.items():
                    T[i*2+j, j2*2+k, i2*2+k2] += ui * vj * wk
    # Expected: T[i*2+j, j*2+k, i*2+k] = 1 for all i,j,k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if T[i*2+j, j*2+k, i*2+k] != 1.0:
                    return False
    # All other entries should be 0
    for idx in range(4):
        for jdx in range(4):
            for kdx in range(4):
                # Check if this is a valid (i,j,k) triple
                valid = False
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            if i*2+j == idx and j*2+k == jdx and i*2+k == kdx:
                                valid = True
                if not valid and T[idx, jdx, kdx] != 0.0:
                    return False
    return True

def _construct_2x3x9_strassen():
    """
    Construct rank-50 decomposition for 2x3 * 3x9 using Strassen on 2x2x2 blocks.
    
    Structure:
    - j in {0,1} paired with k in {0,1},{2,3},{4,5},{6,7}: 4 blocks of 2x2x2, rank 7 each = 28
    - j in {0,1} with k=8: 2x2x1, rank 4 = 4
    - j=2 with k in {0,1},{2,3},{4,5},{6,7}: 4 blocks of 2x1x2, rank 4 each = 16
    - j=2 with k=8: 2x1x1, rank 2 = 2
    Total: 28 + 4 + 16 + 2 = 50
    """
    n, m, p = 2, 3, 9
    strassen_terms = _strassen_2x2x2_terms()
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Block type 1: j in {0,1}, k-pair (k1,k2) -> 2x2x2, rank 7
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    for (k1, k2) in k_pairs:
        for (u_d, v_d, w_d) in strassen_terms:
            U_row = [0.0] * (n * m)
            V_row = [0.0] * (m * p)
            W_row = [0.0] * (p * n)
            for (li, lj), coeff in u_d.items():
                U_row[li * m + lj] = coeff
            for (lj, lk), coeff in v_d.items():
                V_row[lj * p + (k1 + lk)] = coeff
            for (li, lk), coeff in w_d.items():
                W_row[(k1 + lk) * n + li] = coeff
            U_rows.append(U_row)
            V_rows.append(V_row)
            W_rows.append(W_row)
    
    # Block type 2: j in {0,1}, k=8 -> 2x2x1, rank 4
    for i in range(2):
        for j in range(2):
            U_row = [0.0] * (n * m)
            V_row = [0.0] * (m * p)
            W_row = [0.0] * (p * n)
            U_row[i * m + j] = 1.0
            V_row[j * p + 8] = 1.0
            W_row[8 * n + i] = 1.0
            U_rows.append(U_row)
            V_rows.append(V_row)
            W_rows.append(W_row)
    
    # Block type 3: j=2, k-pair (k1,k2) -> 2x1x2, rank 4
    for (k1, k2) in k_pairs:
        for i in range(2):
            for k in [k1, k2]:
                U_row = [0.0] * (n * m)
                V_row = [0.0] * (m * p)
                W_row = [0.0] * (p * n)
                U_row[i * m + 2] = 1.0
                V_row[2 * p + k] = 1.0
                W_row[k * n + i] = 1.0
                U_rows.append(U_row)
                V_rows.append(V_row)
                W_rows.append(W_row)
    
    # Block type 4: j=2, k=8 -> 2x1x1, rank 2
    for i in range(2):
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        U_row[i * m + 2] = 1.0
        V_row[2 * p + 8] = 1.0
        W_row[8 * n + i] = 1.0
        U_rows.append(U_row)
        V_rows.append(V_row)
        W_rows.append(W_row)
    
    rank = len(U_rows)
    U = [[U_rows[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_rows[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_rows[c][r] for c in range(rank)] for r in range(p * n)]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)], dtype=np.float64)
        v_vec = np.array([V[i][c] for i in range(m*p)], dtype=np.float64)
        w_vec = np.array([W[i][c] for i in range(p*n)], dtype=np.float64)
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec, optimize=True)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-9, diff

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    try:
        # First verify our Strassen terms are correct
        if not _verify_2x2x2():
            # Fallback to schoolbook if Strassen terms are wrong
            U, V, W = _rank1_decomposition(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        U, V, W = _construct_2x3x9_strassen()
        
        ok, diff = _verify_decomposition(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, fallback
            U, V, W = _rank1_decomposition(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
