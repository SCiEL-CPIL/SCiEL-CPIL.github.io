import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = 4, 5, 6
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    a_row_blocks = [[0, 1], [2, 3]]
    mid_blocks = [[0, 1], [2, 3], [4]]
    b_col_blocks = [[0, 1], [2, 3], [4, 5]]
    
    col = 0
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ar = a_row_blocks[i]
                mid = mid_blocks[j]
                bc = b_col_blocks[k]
                
                # m1 = (a00+a11)(b00+b11)
                U[ar[0]*m+mid[0]][col] += 1
                U[ar[1]*m+mid[1]][col] += 1
                V[mid[0]*p+bc[0]][col] += 1
                V[mid[1]*p+bc[1]][col] += 1
                W[bc[0]*n+ar[0]][col] += 1
                W[bc[1]*n+ar[1]][col] += 1
                col += 1
                
                # m2 = (a10+a11)b00
                U[ar[1]*m+mid[0]][col] += 1
                U[ar[1]*m+mid[1]][col] += 1
                V[mid[0]*p+bc[0]][col] += 1
                W[bc[0]*n+ar[1]][col] += 1
                W[bc[1]*n+ar[1]][col] -= 1
                col += 1
                
                # m3 = a00(b01-b11)
                U[ar[0]*m+mid[0]][col] += 1
                V[mid[0]*p+bc[1]][col] += 1
                V[mid[1]*p+bc[1]][col] -= 1
                W[bc[1]*n+ar[0]][col] += 1
                W[bc[1]*n+ar[1]][col] += 1
                col += 1
                
                # m4 = a11(b10-b00)
                U[ar[1]*m+mid[1]][col] += 1
                V[mid[1]*p+bc[0]][col] += 1
                V[mid[0]*p+bc[0]][col] -= 1
                W[bc[0]*n+ar[0]][col] += 1
                W[bc[0]*n+ar[1]][col] += 1
                col += 1
                
                # m5 = (a00+a01)b11
                U[ar[0]*m+mid[0]][col] += 1
                U[ar[0]*m+mid[1]][col] += 1
                V[mid[1]*p+bc[1]][col] += 1
                W[bc[0]*n+ar[0]][col] -= 1
                W[bc[1]*n+ar[0]][col] += 1
                col += 1
                
                # m6 = (a10-a00)(b00+b01)
                U[ar[1]*m+mid[0]][col] += 1
                U[ar[0]*m+mid[0]][col] -= 1
                V[mid[0]*p+bc[0]][col] += 1
                V[mid[0]*p+bc[1]][col] += 1
                W[bc[1]*n+ar[1]][col] += 1
                col += 1
                
                # m7 = (a01-a11)(b10+b11)
                U[ar[0]*m+mid[1]][col] += 1
                U[ar[1]*m+mid[1]][col] -= 1
                V[mid[1]*p+bc[0]][col] += 1
                V[mid[1]*p+bc[1]][col] += 1
                W[bc[0]*n+ar[0]][col] += 1
                col += 1
    
    for i in range(2):
        for k in range(3):
            ar = a_row_blocks[i]
            mid = mid_blocks[2]
            bc = b_col_blocks[k]
            for r in range(2):
                for c in range(2):
                    U[ar[r]*m+mid[0]][col] += 1
                    V[mid[0]*p+bc[c]][col] += 1
                    W[bc[c]*n+ar[r]][col] += 1
                    col += 1
    
    assert col == rank
    
    # Quick verification with numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C via decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually: C = sum_r (U[:,r] reshaped) * (V[:,r] reshaped) ... 
    # The decomposition means: C[i,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over j? No.
    
    # The tensor decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Where T is the matrix multiplication tensor.
    # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product entry.
    
    # For verification: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] over j and r
    
    # Let me verify differently:
    # The decomposition says: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for non-product entries, it should be 0.
    
    # Actually, the standard verification:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that's not right either.
    
    # The correct interpretation: 
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Hmm, actually the standard form is:
    # C[i,k] = sum_r (U^T A_flat)[r] * (V^T B_flat)[r] * W[k*n+i, r]
    # where A_flat is A flattened in row-major order, etc.
    
    # Let me just verify: for each r, the contribution to C is:
    # C += W_col_r * outer(U_col_r reshaped, V_col_r reshaped)
    # where U_col_r is reshaped to (n,m) and V_col_r to (m,p)
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Ur = U_np[:, r].reshape(n, m)
        Vr = V_np[:, r].reshape(m, p)
        Wr = W_np[:, r]  # shape (p*n,)
        # C += Wr reshaped * (Ur * Vr elementwise? No...)
        # Actually: C[i,k] += sum over the tensor... 
        
    # Let me think about this more carefully.
    # The matrix multiplication tensor T has T[i,j,k] = 1 if this is a valid product term.
    # T = sum_r u_r ⊗ v_r ⊗ w_r where u_r ∈ R^{nm}, v_r ∈ R^{mp}, w_r ∈ R^{pn}
    # T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    
    # So verification: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for invalid (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # Quick check with a few entries:
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fall back to schoolbook
        rank_fb = n*m*p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c2] = 1
                    V_fb[j*p+k][c2] = 1
                    W_fb[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
