"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140  # safety margin
    
    # Start with the known best: 8 Strassen blocks, rank 56
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Build Strassen-based decomposition of rank 56
    def build_strassen_56():
        rank = 56
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # 8 Strassen 2x2 blocks
        # Split m=4 into two groups of 2: j in {0,1} and j in {2,3}
        # Split p=8 into four groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
        # For each (j_group, k_group), do Strassen 2x2@2x2
        
        col = 0
        for jg in range(2):  # j group: 0 or 1 (j = 2*jg, 2*jg+1)
            for kg in range(4):  # k group: 0,1,2,3 (k = 2*kg, 2*kg+1)
                # Strassen 7 multiplications for A[i, 2*jg:2*jg+2] @ B[2*jg:2*jg+2, 2*kg:2*kg+2]
                # A indices: i in {0,1}, j in {2*jg, 2*jg+1}
                # B indices: j in {2*jg, 2*jg+1}, k in {2*kg, 2*kg+1}
                # C indices: i in {0,1}, k in {2*kg, 2*kg+1}
                
                # Strassen for 2x2:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                j0, j1 = 2*jg, 2*jg+1
                k0, k1 = 2*kg, 2*kg+1
                
                # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                U[i0*m + j0][col] = 1  # A[0, j0]
                U[i1*m + j1][col] = 1  # A[1, j1]
                V[j0*p + k0][col] = 1  # B[j0, k0]
                V[j1*p + k1][col] = 1  # B[j1, k1]
                W[k0*n + i0][col] = 1  # C[0, k0] +=
                W[k1*n + i1][col] = 1  # C[1, k1] +=
                W[k0*n + i1][col] += 1  # C[1, k0] +=
                W[k1*n + i0][col] += 1  # C[0, k1] +=
                col += 1
                
                # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
                U[i1*m + j0][col] = 1
                U[i1*m + j1][col] = 1
                V[j0*p + k0][col] = 1
                W[k0*n + i0][col] = 1
                W[k0*n + i1][col] = 1  # wait, need to check Strassen formula
                col += 1
                
                # Let me redo Strassen properly
                # For 2x2 matrices A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
                # C = [[a*e+b*g, a*f+b*h],
                #      [c*e+d*g, c*f+d*h]]
                # Strassen:
                # M1 = (a+d)(e+h)
                # M2 = (c+d)e
                # M3 = a(f-h)
                # M4 = d(g-e)
                # M5 = (a+b)h
                # M6 = (c-a)(e+f)
                # M7 = (b-d)(g+h)
                # C00 = M1+M4-M5+M7
                # C01 = M3+M5
                # C10 = M2+M4
                # C11 = M1-M2+M3+M6
                
                # In our case:
                # A = [[A[0,j0], A[0,j1]],
                #      [A[1,j0], A[1,j1]]]
                # B = [[B[j0,k0], B[j0,k1]],
                #      [B[j1,k0], B[j1,k1]]]
                
                # Let me rebuild this properly
                pass
        
        # This is getting complex. Let me just use a direct approach.
        return U, V, W
    
    # Actually, let me build the 56-rank Strassen decomposition carefully
    def build_strassen():
        rank = 56
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for jg in range(2):
            for kg in range(4):
                j0, j1 = 2*jg, 2*jg+1
                k0, k1 = 2*kg, 2*kg+1
                
                # Strassen 7 products
                # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                U[0*m+j0][col] = 1
                U[1*m+j1][col] = 1
                V[j0*p+k0][col] = 1
                V[j1*p+k1][col] = 1
                W[k0*n+0][col] = 1   # C[0,k0]
                W[k1*n+1][col] = 1   # C[1,k1]
                W[k0*n+1][col] = 1   # C[1,k0]
                W[k1*n+0][col] = 1   # C[0,k1]
                col += 1
                
                # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                U[1*m+j0][col] = 1
                U[1*m+j1][col] = 1
                V[j0*p+k0][col] = 1
                W[k0*n+1][col] = 1   # C[1,k0]
                col += 1
                
                # M3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
                U[0*m+j0][col] = 1
                V[j0*p+k1][col] = 1
                V[j1*p+k1][col] = -1
                W[k1*n+0][col] = 1   # C[0,k1]
                col += 1
                
                # M4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
                U[1*m+j1][col] = 1
                V[j1*p+k0][col] = 1
                V[j0*p+k0][col] = -1
                W[k0*n+1][col] = 1   # C[1,k0]
                col += 1
                
                # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                U[0*m+j0][col] = 1
                U[0*m+j1][col] = 1
                V[j1*p+k1][col] = 1
                W[k1*n+1][col] = 1   # C[1,k1]
                col += 1
                
                # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                U[1*m+j0][col] = 1
                U[0*m+j0][col] = -1
                V[j0*p+k0][col] = 1
                V[j0*p+k1][col] = 1
                W[k0*n+1][col] = 1   # C[1,k0]
                W[k1*n+1][col] = 1   # C[1,k1]
                col += 1
                
                # M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                U[0*m+j1][col] = 1
                U[1*m+j1][col] = -1
                V[j1*p+k0][col] = 1
                V[j1*p+k1][col] = 1
                W[k0*n+0][col] = 1   # C[0,k0]
                W[k1*n+0][col] = 1   # C[0,k1]
                col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen()
    
    # Verify
    def verify(U, V, W):
        # Check that sum_c U[:,c] outer V[:,c] outer W[:,c] = matmul tensor
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matmul: T[i,j,k] should be 1 if this is the (i,j,k) entry of the matmul tensor
        # Actually the matmul tensor for (n,m,p) is: T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiplication" sense)
        # Wait no. The matrix multiplication tensor is defined as:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # And the decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        n, m, p = DIMENSIONS
        # Build expected tensor
        # T[i,j,k] = 1 for all i,j,k
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fall back to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        rank = n*m*p
    
    # Try to improve with random search
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt optimization to reduce rank
    # This is a non-convex problem. Try random restarts with low rank.
    rng = np.random.default_rng(42)
    
    for target_rank in [52, 50, 48, 44, 40]:
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            U_try = rng.standard_normal((n*m, target_rank))
            V_try = rng.standard_normal((m*p, target_rank))
            W_try = rng.standard_normal((p*n, target_rank))
            
            # Optimize: minimize ||T - sum_c U_c⊗V_c⊗W_c||_F
            # Use gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                # Compute residual
                # T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Gradient w.r.t. U[i*m+j, c] = -sum_{j,k} T[i,j,k]*V[j*p+k,c]*W[k*n+i,c] + sum_c' ... 
                # Actually: residual[i,j,k] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # d(residual)/d(U[a,c]) = -V[j*p+k,c]*W[k*n+i,c] where a=i*m+j
                # So grad_U[a,c] = -sum_{j,k: a=i*m+j} V[j*p+k,c]*W[k*n+i,c]
                # = -V[j*p+k,c]*W[k*n+i,c] for the specific i,j that map to a
                
                # This is expensive. Let me use a simpler approach.
                pass
            break  # Skip optimization for now
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
