import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # --- Rank 50 baseline: 2x2 blocks ---
    def build_rank50():
        rank_50 = 50
        U50 = np.zeros((n*m, rank_50), dtype=int)
        V50 = np.zeros((m*p, rank_50), dtype=int)
        W50 = np.zeros((p*n, rank_50), dtype=int)
        
        col = 0
        # For each pair of A columns (j0 in {0,2}), process 7 B columns
        for j0 in [0, 2]:
            # 3 blocks of 2x2x2 (Strassen) for B cols (0,1), (2,3), (4,5)
            for k0 in [0, 2, 4]:
                # Strassen 2x2x2: 7 terms
                # A block: A[0:2, j0:j0+2], B block: B[j0:j0+2, k0:k0+2], C block: C[0:2, k0:k0+2]
                
                # Term 0: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
                U50[j0, col] += 1; U50[j0+1+m, col] += 1
                V50[k0*p+j0, col] += 1; V50[(k0+1)*p+j0+1, col] += 1
                W50[k0*n+0, col] += 1; W50[(k0+1)*n+1, col] += 1
                col += 1
                
                # Term 1: (a10+a11)b00 -> c10 += 1, c11 -= 1
                U50[j0+m, col] += 1; U50[j0+1+m, col] += 1
                V50[k0*p+j0, col] += 1
                W50[k0*n+1, col] += 1; W50[(k0+1)*n+1, col] -= 1
                col += 1
                
                # Term 2: a00(b10-b11) -> c01 += 1, c11 += 1
                U50[j0, col] += 1
                V50[(k0+1)*p+j0, col] += 1; V50[(k0+1)*p+j0+1, col] -= 1
                W50[k0*n+1, col] += 1; W50[(k0+1)*n+1, col] += 1
                col += 1
                
                # Term 3: a11(b01-b00) -> c00 += 1, c10 += 1
                U50[j0+1+m, col] += 1
                V50[k0*p+j0+1, col] += 1; V50[k0*p+j0, col] -= 1
                W50[k0*n+0, col] += 1; W50[k0*n+1, col] += 1
                col += 1
                
                # Term 4: (a00+a01)b11 -> c00 -= 1, c01 += 1
                U50[j0, col] += 1; U50[j0+1, col] += 1
                V50[(k0+1)*p+j0+1, col] += 1
                W50[k0*n+0, col] -= 1; W50[k0*n+1, col] += 1
                col += 1
                
                # Term 5: (a10-a00)(b00+b01) -> c11 += 1
                U50[j0+m, col] += 1; U50[j0, col] -= 1
                V50[k0*p+j0, col] += 1; V50[k0*p+j0+1, col] += 1
                W50[(k0+1)*n+1, col] += 1
                col += 1
                
                # Term 6: (a01-a11)(b10+b11) -> c00 += 1
                U50[j0+1, col] += 1; U50[j0+1+m, col] -= 1
                V50[(k0+1)*p+j0, col] += 1; V50[(k0+1)*p+j0+1, col] += 1
                W50[k0*n+0, col] += 1
                col += 1
            
            # 1 block of 2x2x1 (schoolbook) for B col 6
            for t in range(4):
                # Term t: a_{row, col} * b_{row, 6}
                # t=0: A[0,j0]*B[j0,6] -> C[0,6]
                # t=1: A[0,j0+1]*B[j0+1,6] -> C[0,6]
                # t=2: A[1,j0]*B[j0,6] -> C[1,6]
                # t=3: A[1,j0+1]*B[j0+1,6] -> C[1,6]
                a_row = t // 2
                a_col_off = t % 2
                b_row_off = t % 2
                U50[a_row*m + j0 + a_col_off, col] += 1
                V50[6*p + j0 + b_row_off, col] += 1
                W50[6*n + a_row, col] += 1
                col += 1
        
        return U50, V50, W50
    
    def verify(U, V, W, tol=1e-9):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        A_flat = A.flatten()  # index i*m + j
        B_flat = B.flatten()  # index i*p + j
        
        C_computed = np.zeros((n, p))
        for k in range(U.shape[1]):
            u_col = U[:, k]
            v_col = V[:, k]
            w_col = W[:, k]
            
            # For each nonzero in U, V, W
            nz_u = np.nonzero(u_col)[0]
            nz_v = np.nonzero(v_col)[0]
            nz_w = np.nonzero(w_col)[0]
            
            for a_idx in nz_u:
                i = a_idx // m
                j_a = a_idx % m
                for b_idx in nz_v:
                    j_b = b_idx // p
                    j = b_idx % p
                    for c_idx in nz_w:
                        c_i = c_idx % n
                        c_j = c_idx // n
                        if i == c_i and j_b == c_j and j_a == j_b:
                            C_computed[c_i, c_j] += u_col[a_idx] * v_col[b_idx] * w_col[c_idx]
        
        return np.max(np.abs(C_computed - C_expected)) < tol
    
    # Build and verify rank 50
    U50, V50, W50 = build_rank50()
    
    if verify(U50, V50, W50):
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    
    # Fallback
    return {"U": [], "V": [], "W": []}
