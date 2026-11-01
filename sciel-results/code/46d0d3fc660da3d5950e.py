"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    # Use Strassen for each 2x2 block of B, giving rank 4*7=28
    num_blocks = p // 2  # 4 blocks of 2x2
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen coefficients for 2x2 x 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    
    # For each block b (0..3), B block is B[:, 2b:2b+2]
    # A entries: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B block entries: (0,2b)=e, (0,2b+1)=f, (1,2b)=g, (1,2b+1)=h
    
    for b in range(num_blocks):
        base = b * 7
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1  # A[0,0]
        U[3][base + 0] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 0] += 1  # B[0,2b]
        V[1 * p + 2 * b + 1][base + 0] += 1  # B[1,2b+1]
        # W: C[0,2b] = M1 + M4 - M5 + M7, C[0,2b+1] = M3 + M5, C[1,2b] = M2 + M4, C[1,2b+1] = M1 - M2 + M3 + M6
        W[2 * b * n + 0][base + 0] += 1  # C[0,2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1  # A[1,0]
        U[3][base + 1] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 1] += 1  # B[0,2b]
        W[2 * b * n + 1][base + 1] += 1  # C[1,2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1  # A[0,0]
        V[0 * p + 2 * b + 1][base + 2] += 1  # B[0,2b+1]
        V[1 * p + 2 * b + 1][base + 2] -= 1  # B[1,2b+1]
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1  # A[1,1]
        V[1 * p + 2 * b][base + 3] += 1  # B[1,2b]
        V[0 * p + 2 * b][base + 3] -= 1  # B[0,2b]
        W[2 * b * n + 0][base + 3] += 1  # C[0,2b]
        W[2 * b * n + 1][base + 3] += 1  # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1  # A[0,0]
        U[1][base + 4] += 1  # A[0,1]
        V[1 * p + 2 * b + 1][base + 4] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[1,2b]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1  # A[1,0]
        U[0][base + 5] -= 1  # A[0,0]
        V[0 * p + 2 * b][base + 5] += 1  # B[0,2b]
        V[0 * p + 2 * b + 1][base + 5] += 1  # B[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1  # A[0,1]
        U[3][base + 6] -= 1  # A[1,1]
        V[1 * p + 2 * b][base + 6] += 1  # B[1,2b]
        V[1 * p + 2 * b + 1][base + 6] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 6] += 1  # C[0,2b]
    
    # Verify correctness
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: C[i,l] = sum_k U[i,k] * V[k,l] * W[l,k]
        # Actually the tensor decomposition is: T[i,j,k] = U[i,k] * V[j,k] * W[i,j,k]... 
        # No, for matrix multiplication: C = A*B, the tensor is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # where a indexes A, b indexes B, c indexes C
        # C[c] = sum_a sum_b T[a,b,c] * A[a] * B[b]
        # So T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # A has n*m = 4 entries, B has m*p = 16 entries, C has p*n = 16 entries
        # Indexing: A[i*m+k] for row i, col k of A (2x2)
        # B[k*p+l] for row k, col l of B (2x8)  
        # C[l*n+i] for row l, col i of C (2x8) -> W[k*n+i, r] per convention
        
        # Let me verify with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()  # [A[0,0], A[0,1], A[1,0], A[1,1]]
        B_flat = B_test.flatten()  # [B[0,0], B[0,1], ..., B[1,7]]
        
        # Compute C using decomposition
        C_computed = np.zeros(p * n)
        for r in range(rank):
            u_r = U_np[:, r]  # (n*m,)
            v_r = V_np[:, r]  # (m*p,)
            w_r = W_np[:, r]  # (p*n,)
            # C[c] += sum_a u_r[a] * A_flat[a] * sum_b v_r[b] * B_flat[b] * w_r[c]
            # Actually: C_computed[c] = sum_r w_r[c] * (sum_a U[a,r]*A_flat[a]) * (sum_b V[b,r]*B_flat[b])
            prod_u = np.dot(u_r, A_flat)
            prod_v = np.dot(v_r, B_flat)
            C_computed += w_r * prod_u * prod_v
        
        C_computed = C_computed.reshape(p, n).T
        
        # Check
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # Fallback to schoolbook if verification fails
            rank_schoolbook = n * m * p  # 32
            U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
            V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
            W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
            
            idx = 0
            for i in range(n):
                for k in range(m):
                    for l in range(p):
                        U_sb[i * m + k][idx] = 1
                        V_sb[k * p + l][idx] = 1
                        W_sb[l * n + i][idx] = 1
                        idx += 1
            
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
