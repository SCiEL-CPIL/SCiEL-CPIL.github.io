"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def construct():
    """
    Construct the tensor decomposition for 4x4 * 4x6 -> 4x6.
    
    Strategy:
    Decompose the 4x4 * 4x6 multiplication into a Kronecker product of:
    1. An outer 2x2 * 2x3 multiplication (for the block structure).
    2. An inner 2x2 * 2x2 multiplication (for the block contents).
    
    The outer decomposition for 2x2 * 2x3 is formed by combining:
    - Rank-7 Strassen decomposition for the 2x2 * 2x2 part (cols 0,1).
    - Rank-4 schoolbook decomposition for the 2x2 * 1 part (col 2).
    Total outer rank = 11.
    
    The inner decomposition is the standard rank-7 Strassen for 2x2 * 2x2.
    Total rank = 11 * 7 = 77.
    
    Returns:
    {
        'U': list of lists, shape (16, 77)
        'V': list of lists, shape (24, 77)
        'W': list of lists, shape (24, 77)
    }
    """
    n, m, p = DIMENSIONS # 4, 4, 6
    
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    sU, sV, sW = _strassen_2x2()
    
    # Outer decomposition: 2x2 * 2x3 -> 2x3
    # We build a list of outer terms. Each term is a tuple (U_outer, V_outer, W_outer)
    # where U_outer is a vector of size 4 (2x2 matrix A), V_outer size 6 (2x3 matrix B), W_outer size 6 (2x3 matrix C).
    # We represent them as dicts {(row, col): coeff} for sparse handling.
    
    outer_terms = []
    
    # 1. Strassen terms for cols 0,1 (7 terms)
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        
        # U part: A is 2x2
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2 + b, t]
                if coeff != 0:
                    U_dict[(a, b)] = U_dict.get((a, b), 0) + coeff
                    
        # V part: B is 2x2 (cols 0,1). Indices: row b, col c (0..1)
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2 + c, t]
                if coeff != 0:
                    # Map to 2x3: row b, col c
                    V_dict[(b, c)] = V_dict.get((b, c), 0) + coeff
                    
        # W part: C is 2x2 (cols 0,1). Indices: row a, col c (0..1)
        # sW is indexed by [c*2 + a]
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2 + a, t]
                if coeff != 0:
                    W_dict[(a, c)] = W_dict.get((a, c), 0) + coeff
        
        if U_dict and V_dict and W_dict:
            outer_terms.append((U_dict, V_dict, W_dict))
            
    # 2. Schoolbook terms for col 2 (4 terms)
    # C[a, 2] = sum_b A[a, b] * B[b, 2]
    # Terms: (a, b)
    for a in range(2):
        for b in range(2):
            U_dict = {(a, b): 1}
            V_dict = {(b, 2): 1}
            W_dict = {(a, 2): 1}
            outer_terms.append((U_dict, V_dict, W_dict))
            
    total_rank = len(outer_terms) * 7
    
    # Initialize output matrices
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, (U_outer, V_outer, W_outer) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # Check time
            if time.time() > deadline:
                # Incomplete result, but return what we have? No, must be valid.
                # This block should be fast enough.
                pass
            
            # Combine outer and inner
            # U_final: 4x4 matrix. Index (i, j).
            # Outer A index (a, b) -> Block (a, b). Inner A index (ai2, bj2) -> Inside block.
            # i = 2*a + ai2, j = 2*b + bj2
            for (a, b), coeff_outer in U_outer.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            i = 2*a + ai2
                            j = 2*b + bj2
                            u_row = i * m + j
                            U[u_row, col] += coeff_outer * coeff_inner
                            
            # V_final: 4x6 matrix. Index (j, k).
            # Outer B index (b, c) -> Block (b, c). Inner B index (bj2, ck2) -> Inside block.
            # j = 2*b + bj2, k = 2*c + ck2
            for (b, c), coeff_outer in V_outer.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            j = 2*b + bj2
                            k = 2*c + ck2
                            v_row = j * p + k
                            V[v_row, col] += coeff_outer * coeff_inner
                            
            # W_final: 4x6 matrix. Index (i, k). Trace convention: row = k*n + i.
            # Outer C index (a, c) -> Block (a, c). Inner C index (ai2, ck2) -> Inside block.
            # i = 2*a + ai2, k = 2*c + ck2
            # Inner W index is ck2*2 + ai2
            for (a, c), coeff_outer in W_outer.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            i = 2*a + ai2
                            k = 2*c + ck2
                            w_row = k * n + i
                            W[w_row, col] += coeff_outer * coeff_inner
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 4, 4, 6
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify on random integer matrices
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    C_ref = A @ B
    
    # Compute C from decomposition
    # C[i, k] = sum_r U[i*m+j? No, U is flattened A] 
    # U row index: i*m + j. V row index: j*p + k. W row index: k*n + i.
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want C[i, k] = sum_j A[i, j] B[j, k]
    
    C_calc = np.zeros((n, p), dtype=int)
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                # Sum over r
                u_vec = U[i*m + j, :]
                v_vec = V[j*p + k, :]
                w_val = W[k*n + i, :]
                s += np.sum(u_vec * v_vec * w_val)
            C_calc[i, k] = s
            
    if np.array_equal(C_ref, C_calc):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("Ref:\n", C_ref)
        print("Calc:\n", C_calc)
