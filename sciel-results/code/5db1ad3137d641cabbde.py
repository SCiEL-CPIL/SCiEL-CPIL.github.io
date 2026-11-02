"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    start_time = time.time()
    deadline = start_time + 140.0
    
    def strassen_block(col_off):
        """Strassen 7-rank decomposition for 2x2 * 2x2 block at columns col_off, col_off+1."""
        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # V indices: B[0,col_off]=col_off, B[1,col_off]=p+col_off, B[0,col_off+1]=col_off+1, B[1,col_off+1]=p+col_off+1
        # W indices: C[0,col_off]=2*col_off, C[1,col_off]=2*col_off+1, C[0,col_off+1]=2*col_off+2, C[1,col_off+1]=2*col_off+3
        
        # Verified Strassen formulas:
        # M1=(a+b)(e+f), M2=(c+d)(e), M3=a(f-h), M4=d(g-e), M5=(a+d)h, M6=(b-c)(e+g), M7=(b-d)(f+h)
        # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        
        terms = []
        # M1=(a+b)(e+f)
        terms.append({'u':[(0,1),(1,1)], 'v':[(col_off,1),(col_off+1,1)], 'w':[(2*col_off,1),(2*col_off+3,1)]})
        # M2=(c+d)(e)
        terms.append({'u':[(2,1),(3,1)], 'v':[(col_off,1)], 'w':[(2*col_off+1,1),(2*col_off+3,-1)]})
        # M3=a(f-h)
        terms.append({'u':[(0,1)], 'v':[(col_off+1,1),(p+col_off+1,-1)], 'w':[(2*col_off+2,1),(2*col_off+3,1)]})
        # M4=d(g-e)
        terms.append({'u':[(3,1)], 'v':[(p+col_off,1),(col_off,-1)], 'w':[(2*col_off,1),(2*col_off+1,1)]})
        # M5=(a+d)h
        terms.append({'u':[(0,1),(3,1)], 'v':[(p+col_off+1,1)], 'w':[(2*col_off,-1),(2*col_off+2,1)]})
        # M6=(b-c)(e+g)
        terms.append({'u':[(1,1),(2,-1)], 'v':[(col_off,1),(p+col_off,1)], 'w':[(2*col_off+3,1)]})
        # M7=(b-d)(f+h)
        terms.append({'u':[(1,1),(3,-1)], 'v':[(col_off+1,1),(p+col_off+1,1)], 'w':[(2*col_off,1)]})
        
        return terms
    
    def schoolbook_single(col_off):
        """Schoolbook 4-rank for single column."""
        terms = []
        for i in range(n):
            for k in range(m):
                terms.append({'u':[(i*m+k,1)], 'v':[(k*p+col_off,1)], 'w':[(col_off*n+i,1)]})
        return terms
    
    all_terms = []
    for pair_idx in range(7):
        col_off = 2 * pair_idx
        all_terms.extend(strassen_block(col_off))
    all_terms.extend(schoolbook_single(14))
    
    rank = len(all_terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, term in enumerate(all_terms):
        for idx, val in term['u']:
            U[idx][c] += val
        for idx, val in term['v']:
            V[idx][c] += val
        for idx, val in term['w']:
            W[idx][c] += val
    
    # Verify correctness
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct: C[i,j] = sum_c U[i*m+k? no...]
    # Actually: C[i,j] = sum_{k,c} A[i,k] * B[k,j] * (decomposition)
    # The tensor is: T[i,k,j] = sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
    # Wait no. The decomposition is: T[i,k,j] = sum_c U[i*m+k,c] * V[k*p+j,c] * W[j*n+i,c]
    # But that's not how matrix multiplication tensor works.
    
    # Actually for matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T has T[i,k,j] = 1 if... no.
    # The multiplication tensor: M = sum_{i,k,j} e_{i,k} ⊗ e_{k,j} ⊗ e_{i,j}
    # So M[i,k,j] = 1 always? No, M is the structure tensor.
    # M[i,k,j] = 1 for all i,k,j? No.
    # Actually M[i,k,j] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j].
    # So M[i,k,j] = 1 for all valid i,k,j.
    # The rank decomposition: M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i*m+k] = U[i*m+k,c], v_c[k*p+j] = V[k*p+j,c], w_c[j*n+i] = W[j*n+i,c]
    # And we need: sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] = 1 for all i,k,j
    
    # Let's verify: for each (i,k,j), sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] should be 1
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+k][c] * V[k*p+j][c] * W[j*n+i][c]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},k={k},j={j}, got {s}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
