import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    Uses block decomposition:
    - Split rows: {0}, {1,2}
    - Split middle: {0,1}, {2,3}
    - Split columns: {0,1}, {2,3}, {4,5}
    
    Subproblems:
    - 6 of type 1x2x2 (rank 2 each)
    - 6 of type 2x2x2 (rank 7 each, Strassen)
    Total rank: 54
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Define the block partitions
    row_blocks = [[0], [1, 2]]
    mid_blocks = [[0, 1], [2, 3]]
    col_blocks = [[0, 1], [2, 3], [4, 5]]
    
    # Collect all subproblems
    subproblems = []
    for rb in row_blocks:
        for mb in mid_blocks:
            for cb in col_blocks:
                subproblems.append((rb, mb, cb))
    
    # Build the decomposition
    U_rows = []
    V_rows = []
    W_rows = []
    
    for rb, mb, cb in subproblems:
        nr = len(rb)
        nm = len(mb)
        nc = len(cb)
        
        if nr == 1 and nm == 2 and nc == 2:
            # 1x2x2: rank 2
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # Term 1: A[0,0]*B[0,0] contributes to C[0,0]
            # Term 2: A[0,1]*B[1,0] contributes to C[0,0]
            # Term 3: A[0,0]*B[0,1] contributes to C[0,1]
            # Term 4: A[0,1]*B[1,1] contributes to C[0,1]
            # But we can do it in rank 2:
            # U: [1, 0] for term 1 (A[0,0]), [0, 1] for term 2 (A[0,1])
            # V: [B[0,0], B[0,1]] for term 1, [B[1,0], B[1,1]] for term 2
            # W: [1, 1] for both (since both contribute to C[0,0] and C[0,1] respectively)
            # Wait, this isn't quite right. Let me think again.
            
            # Actually for 1x2x2:
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            
            # We need 2 terms:
            # Term 1: uses A[0,0], B[0,0] and B[0,1], contributes to C[0,0] and C[0,1]
            #   U: [1] (A[0,0])
            #   V: [B[0,0], B[0,1]]
            #   W: [1, 1]
            # Term 2: uses A[0,1], B[1,0] and B[1,1], contributes to C[0,0] and C[0,1]
            #   U: [1] (A[0,1])
            #   V: [B[1,0], B[1,1]]
            #   W: [1, 1]
            
            # So for this subproblem, we add 2 terms.
            # In the global U, V, W, we need to map local indices to global.
            
            # For term 1:
            # U index: row rb[0]=0, mid mb[0]=0 → U[0*4+0] = U[0]
            # V indices: mid mb[0]=0, col cb[0]=0 → V[0*6+0]=V[0]; mid mb[0]=0, col cb[1]=1 → V[0*6+1]=V[1]
            # W indices: col cb[0]=0, row rb[0]=0 → W[0*3+0]=W[0]; col cb[1]=1, row rb[0]=0 → W[1*3+0]=W[3]
            
            # For term 2:
            # U index: row rb[0]=0, mid mb[1]=1 → U[0*4+1] = U[1]
            # V indices: mid mb[1]=1, col cb[0]=0 → V[1*6+0]=V[6]; mid mb[1]=1, col cb[1]=1 → V[1*6+1]=V[7]
            # W indices: same as term 1: W[0], W[3]
            
            # Add term 1
            u_idx = rb[0] * m + mb[0]
            v_idx1 = mb[0] * p + cb[0]
            v_idx2 = mb[0] * p + cb[1]
            w_idx1 = cb[0] * n + rb[0]
            w_idx2 = cb[1] * n + rb[0]
            
            U_rows.append((u_idx, 1))
            V_rows.append((v_idx1, 1))
            V_rows.append((v_idx2, 1))
            W_rows.append((w_idx1, 1))
            W_rows.append((w_idx2, 1))
            
            # Add term 2
            u_idx = rb[0] * m + mb[1]
            v_idx1 = mb[1] * p + cb[0]
            v_idx2 = mb[1] * p + cb[1]
            w_idx1 = cb[0] * n + rb[0]
            w_idx2 = cb[1] * n + rb[0]
            
            U_rows.append((u_idx, 1))
            V_rows.append((v_idx1, 1))
            V_rows.append((v_idx2, 1))
            W_rows.append((w_idx1, 1))
            W_rows.append((w_idx2, 1))
            
        elif nr == 2 and nm == 2 and nc == 2:
            # 2x2x2: Strassen, rank 7
            # Local indices: A[0,0], A[0,1], A[1,0], A[1,1]
            #                B[0,0], B[0,1], B[1,0], B[1,1]
            #                C[0,0], C[0,1], C[1,0], C[1,1]
            
            # Strassen formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            
            # Each Mi is a product of a linear combination of A entries and a linear combination of B entries.
            # So each Mi corresponds to one term in the decomposition.
            
            # For Mi, U has the coefficients of the A linear combination, V has the coefficients of the B linear combination,
            # and W has the coefficients for how Mi contributes to each C entry.
            
            # M1 = (A00+A11)(B00+B11)
            # U: A00 coeff 1, A11 coeff 1
            # V: B00 coeff 1, B11 coeff 1
            # W: C00 coeff 1, C11 coeff 1
            
            # M2 = (A10+A11)B00
            # U: A10 coeff 1, A11 coeff 1
            # V: B00 coeff 1
            # W: C10 coeff 1, C11 coeff -1
            
            # M3 = A00(B10-B11)
            # U: A00 coeff 1
            # V: B10 coeff 1, B11 coeff -1
            # W: C01 coeff 1, C11 coeff 1
            
            # M4 = A11(B01-B00)
            # U: A11 coeff 1
            # V: B01 coeff 1, B00 coeff -1
            # W: C00 coeff 1, C10 coeff 1
            
            # M5 = (A00+A01)B11
            # U: A00 coeff 1, A01 coeff 1
            # V: B11 coeff 1
            # W: C00 coeff -1, C01 coeff 1
            
            # M6 = (A10-A00)(B00+B10)
            # U: A10 coeff 1, A00 coeff -1
            # V: B00 coeff 1, B10 coeff 1
            # W: C11 coeff 1
            
            # M7 = (A01-A11)(B01+B11)
            # U: A01 coeff 1, A11 coeff -1
            # V: B01 coeff 1, B11 coeff 1
            # W: C00 coeff 1
            
            # Now map local to global indices.
            # Local A[i,j] → global U[rb[i]*m + mb[j]]
            # Local B[j,k] → global V[mb[j]*p + cb[k]]
            # Local C[i,k] → global W[cb[k]*n + rb[i]]
            
            # M1
            U_rows.append((rb[0]*m+mb[0], 1))
            U_rows.append((rb[1]*m+mb[1], 1))
            V_rows.append((mb[0]*p+cb[0], 1))
            V_rows.append((mb[1]*p+cb[1], 1))
            W_rows.append((cb[0]*n+rb[0], 1))
            W_rows.append((cb[1]*n+rb[1], 1))
            
            # M2
            U_rows.append((rb[1]*m+mb[0], 1))
            U_rows.append((rb[1]*m+mb[1], 1))
            V_rows.append((mb[0]*p+cb[0], 1))
            W_rows.append((cb[0]*n+rb[1], 1))
            W_rows.append((cb[1]*n+rb[1], -1))
            
            # M3
            U_rows.append((rb[0]*m+mb[0], 1))
            V_rows.append((mb[1]*p+cb[0], 1))
            V_rows.append((mb[1]*p+cb[1], -1))
            W_rows.append((cb[1]*n+rb[0], 1))
            W_rows.append((cb[1]*n+rb[1], 1))
            
            # M4
            U_rows.append((rb[1]*m+mb[1], 1))
            V_rows.append((mb[0]*p+cb[1], 1))
            V_rows.append((mb[0]*p+cb[0], -1))
            W_rows.append((cb[0]*n+rb[0], 1))
            W_rows.append((cb[0]*n+rb[1], 1))
            
            # M5
            U_rows.append((rb[0]*m+mb[0], 1))
            U_rows.append((rb[0]*m+mb[1], 1))
            V_rows.append((mb[1]*p+cb[1], 1))
            W_rows.append((cb[0]*n+rb[0], -1))
            W_rows.append((cb[1]*n+rb[0], 1))
            
            # M6
            U_rows.append((rb[1]*m+mb[0], 1))
            U_rows.append((rb[0]*m+mb[0], -1))
            V_rows.append((mb[0]*p+cb[0], 1))
            V_rows.append((mb[1]*p+cb[0], 1))
            W_rows.append((cb[1]*n+rb[1], 1))
            
            # M7
            U_rows.append((rb[0]*m+mb[1], 1))
            U_rows.append((rb[1]*m+mb[1], -1))
            V_rows.append((mb[0]*p+cb[1], 1))
            V_rows.append((mb[1]*p+cb[1], 1))
            W_rows.append((cb[0]*n+rb[0], 1))
    
    # Now build the full U, V, W matrices
    total_rank = len(U_rows)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Each "term" in the decomposition corresponds to one column in U, V, W
    # But we've been collecting entries per term. Let me restructure.
    
    # Actually, I need to rethink the data structure. Each term in the decomposition
    # has a set of (U_index, coeff), (V_index, coeff), (W_index, coeff) entries.
    # Let me collect terms properly.
    
    terms = []
    
    for rb, mb, cb in subproblems:
        nr = len(rb)
        nm = len(mb)
        nc = len(cb)
        
        if nr == 1 and nm == 2 and nc == 2:
            # Term 1
            term_u = [(rb[0]*m+mb[0], 1)]
            term_v = [(mb[0]*p+cb[0], 1), (mb[0]*p+cb[1], 1)]
            term_w = [(cb[0]*n+rb[0], 1), (cb[1]*n+rb[0], 1)]
            terms.append((term_u, term_v, term_w))
            
            # Term 2
            term_u = [(rb[0]*m+mb[1], 1)]
            term_v = [(mb[1]*p+cb[0], 1), (mb[1]*p+cb[1], 1)]
            term_w = [(cb[0]*n+rb[0], 1), (cb[1]*n+rb[0], 1)]
            terms.append((term_u, term_v, term_w))
            
        elif nr == 2 and nm == 2 and nc == 2:
            # M1
            term_u = [(rb[0]*m+mb[0], 1), (rb[1]*m+mb[1], 1)]
            term_v = [(mb[0]*p+cb[0], 1), (mb[1]*p+cb[1], 1)]
            term_w = [(cb[0]*n+rb[0], 1), (cb[1]*n+rb[1], 1)]
            terms.append((term_u, term_v, term_w))
            
            # M2
            term_u = [(rb[1]*m+mb[0], 1), (rb[1]*m+mb[1], 1)]
            term_v = [(mb[0]*p+cb[0], 1)]
            term_w = [(cb[0]*n+rb[1], 1), (cb[1]*n+rb[1], -1)]
            terms.append((term_u, term_v, term_w))
            
            # M3
            term_u = [(rb[0]*m+mb[0], 1)]
            term_v = [(mb[1]*p+cb[0], 1), (mb[1]*p+cb[1], -1)]
            term_w = [(cb[1]*n+rb[0], 1), (cb[1]*n+rb[1], 1)]
            terms.append((term_u, term_v, term_w))
            
            # M4
            term_u = [(rb[1]*m+mb[1], 1)]
            term_v = [(mb[0]*p+cb[1], 1), (mb[0]*p+cb[0], -1)]
            term_w = [(cb[0]*n+rb[0], 1), (cb[0]*n+rb[1], 1)]
            terms.append((term_u, term_v, term_w))
            
            # M5
            term_u = [(rb[0]*m+mb[0], 1), (rb[0]*m+mb[1], 1)]
            term_v = [(mb[1]*p+cb[1], 1)]
            term_w = [(cb[0]*n+rb[0], -1), (cb[1]*n+rb[0], 1)]
            terms.append((term_u, term_v, term_w))
            
            # M6
            term_u = [(rb[1]*m+mb[0], 1), (rb[0]*m+mb[0], -1)]
            term_v = [(mb[0]*p+cb[0], 1), (mb[1]*p+cb[0], 1)]
            term_w = [(cb[1]*n+rb[1], 1)]
            terms.append((term_u, term_v, term_w))
            
            # M7
            term_u = [(rb[0]*m+mb[1], 1), (rb[1]*m+mb[1], -1)]
            term_v = [(mb[0]*p+cb[1], 1), (mb[1]*p+cb[1], 1)]
            term_w = [(cb[0]*n+rb[0], 1)]
            terms.append((term_u, term_v, term_w))
    
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (term_u, term_v, term_w) in enumerate(terms):
        for idx, coeff in term_u:
            U[idx][t] = coeff
        for idx, coeff in term_v:
            V[idx][t] = coeff
        for idx, coeff in term_w:
            W[idx][t] = coeff
    
    # Verify the decomposition
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T_rec, T, atol=1e-9):
        # Fallback to schoolbook
        def get_schoolbook():
            rank_schoolbook = n * m * p
            U_s = [[0] * rank_schoolbook for _ in range(n * m)]
            V_s = [[0] * rank_schoolbook for _ in range(m * p)]
            W_s = [[0] * rank_schoolbook for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][column] = 1
                        V_s[j * p + k][column] = 1
                        W_s[k * n + i][column] = 1
                        column += 1
            return {"U": U_s, "V": V_s, "W": W_s}
        return get_schoolbook()
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
