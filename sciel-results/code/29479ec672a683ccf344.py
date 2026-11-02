import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 56: 2 A-blocks * 4 B-blocks * 7 Strassen terms
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = [[c11, c12], [c21, c22]]
    # Indices in local 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U coeffs: (local_A_idx, coeff)
    # V coeffs: (local_B_idx, coeff)
    # W coeffs: (local_C_idx, coeff) -> global W index is k*n + i, so local C (i,k) maps to global offset
    
    strassen_terms = [
        # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a21+a22)b11 -> c21 += M2, c22 -= M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a11(b12-b22) -> c12 += M3, c22 += M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = a22(b21-b11) -> c11 += M4, c21 += M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a11+a12)b22 -> c11 -= M5, c12 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (a21-a11)(b11+b12) -> c22 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a12-a22)(b21+b22) -> c11 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Loop over A-blocks (2 blocks of width 2) and B-blocks (4 blocks of width 2)
    # A is 2x4. A-block u covers columns 2u, 2u+1.
    # B is 4x8. B-block (u, v) covers rows 2u, 2u+1 and cols 2v, 2v+1.
    # Wait, B is 4x8. The "u" index for A-blocks corresponds to the row block of B.
    # A is 2x4. We split A into 2 blocks of 2x2: A_u where u=0,1.
    # B is 4x8. We split B into 8 blocks of 2x2: B_{u,v} where u=0,1 (row block) and v=0,1,2,3 (col block).
    # C is 2x8. We split C into 4 blocks of 2x2: C_v where v=0,1,2,3.
    
    for u in range(2):  # A row block / B row block
        for v in range(4):  # B col block / C col block
            base_col = 7 * (4 * u + v)
            
            # Global indices for A block u
            # A is 2x4. u=0 -> cols 0,1. u=1 -> cols 2,3.
            # A[0, 2u], A[0, 2u+1]
            # A[1, 2u], A[1, 2u+1]
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1
            a_local_to_global = {0: a11, 1: a12, 2: a21, 3: a22}
            
            # Global indices for B block (u, v)
            # B is 4x8.
            # Rows: 2u, 2u+1. Cols: 2v, 2v+1.
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1
            b_local_to_global = {0: b11, 1: b12, 2: b21, 3: b22}
            
            # Global indices for C block v
            # C is 2x8.
            # Rows: 0, 1. Cols: 2v, 2v+1.
            # W index convention: W[k*n + i, rank]. k is col of C, i is row of C.
            # C[0, 2v] -> k=2v, i=0 -> W[2v*2 + 0]
            # C[0, 2v+1] -> k=2v+1, i=0 -> W[(2v+1)*2 + 0]
            # C[1, 2v] -> k=2v, i=1 -> W[2v*2 + 1]
            # C[1, 2v+1] -> k=2v+1, i=1 -> W[(2v+1)*2 + 1]
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1
            c_local_to_global = {0: c11, 1: c12, 2: c21, 3: c22}
            
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                col = base_col + t
                
                for local_idx, coeff in u_coeffs:
                    U[a_local_to_global[local_idx], col] += coeff
                for local_idx, coeff in v_coeffs:
                    V[b_local_to_global[local_idx], col] += coeff
                for local_idx, coeff in w_coeffs:
                    W[c_local_to_global[local_idx], col] += coeff
    
    # Quick verification to ensure correctness
    # We check a few random samples to avoid O(nmp * rank) cost if it's too high, 
    # but nmp=64, rank=56, so 64*56 ~ 3500 ops per check. 
    # Actually, we can verify the full tensor contraction efficiently using numpy.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r U[i*m+j, r] ... wait, U is (n*m, rank).
    # The tensor is T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Note: U index is i*m+j. V index is j*p+k. W index is k*n+i.
    
    # Construct A, B matrices for verification
    # We can just check if U, V, W satisfy the condition for all i,j,k
    # To be safe and fast, we check all 64 triples.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
