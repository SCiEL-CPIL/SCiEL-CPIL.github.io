"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm applied to 6 independent 2x2x2 sub-problems,
    giving rank 42 (vs schoolbook 48).
    
    Verification: For all i in {0,1}, j in {0,1}, k in {0,...,11}:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    t_start = time.time()
    deadline = t_start + 115  # safety margin before 120s limit
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A00+A11)*(B00+B11) -> C11+=1, C22+=1
    # M2 = (A10+A11)*B00       -> C21+=1, C22-=1
    # M3 = A00*(B01-B11)       -> C12+=1, C22+=1
    # M4 = A11*(B10-B00)       -> C11+=1, C21+=1
    # M5 = (A00+A01)*B11       -> C11-=1, C12+=1
    # M6 = (A10-A00)*(B00+B01) -> C22+=1
    # M7 = (A01-A11)*(B10+B11) -> C11+=1
    
    # Verify: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    for t in range(num_pairs):
        if time.time() > deadline - 10:
            break
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00 + A11) * (B00 + B11)
                U[0][c] = 1   # A00
                U[3][c] = 1   # A11
                V[2*t][c] = 1           # B[0, 2t]
                V[p + 2*t+1][c] = 1     # B[1, 2t+1]
                W[2*t*n + 0][c] = 1     # C[0, 2t]
                W[(2*t+1)*n + 1][c] = 1 # C[1, 2t+1]
                
            elif s == 1:
                # M2 = (A10 + A11) * B00
                U[2][c] = 1   # A10
                U[3][c] = 1   # A11
                V[2*t][c] = 1           # B[0, 2t]
                W[2*t*n + 1][c] = 1     # C[1, 2t]
                W[(2*t+1)*n + 1][c] = -1 # C[1, 2t+1]
                
            elif s == 2:
                # M3 = A00 * (B01 - B11)
                U[0][c] = 1   # A00
                V[2*t+1][c] = 1           # B[0, 2t+1]
                V[p + 2*t+1][c] = -1      # B[1, 2t+1]
                W[(2*t+1)*n + 0][c] = 1   # C[0, 2t+1]
                W[(2*t+1)*n + 1][c] = 1   # C[1, 2t+1]
                
            elif s == 3:
                # M4 = A11 * (B10 - B00)
                U[3][c] = 1   # A11
                V[p + 2*t][c] = 1     # B[1, 2t]
                V[2*t][c] = -1        # B[0, 2t]
                W[2*t*n + 0][c] = 1   # C[0, 2t]
                W[2*t*n + 1][c] = 1   # C[1, 2t]
                
            elif s == 4:
                # M5 = (A00 + A01) * B11
                U[0][c] = 1   # A00
                U[1][c] = 1   # A01
                V[p + 2*t+1][c] = 1 # B[1, 2t+1]
                W[2*t*n + 0][c] = -1     # C[0, 2t]
                W[(2*t+1)*n + 0][c] = 1  # C[0, 2t+1]
                
            elif s == 5:
                # M6 = (A10 - A00) * (B00 + B01)
                U[2][c] = 1   # A10
                U[0][c] = -1  # A00
                V[2*t][c] = 1           # B[0, 2t]
                V[2*t+1][c] = 1         # B[0, 2t+1]
                W[(2*t+1)*n + 1][c] = 1 # C[1, 2t+1]
                
            elif s == 6:
                # M7 = (A01 - A11) * (B10 + B11)
                U[1][c] = 1   # A01
                U[3][c] = -1  # A11
                V[p + 2*t][c] = 1     # B[1, 2t]
                V[p + 2*t+1][c] = 1   # B[1, 2t+1]
                W[2*t*n + 0][c] = 1   # C[0, 2t]
    
    # Verify the decomposition using vectorized numpy
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # Build a 3D array T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        # Use einsum: T[i,j,k] = einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to schoolbook if Strassen verification fails
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii*m+jj][c] = 1
                        V_s[jj*p+kk][c] = 1
                        W_s[kk*n+ii][c] = 1
                        c += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
