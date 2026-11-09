"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Partition the 4x5x5 tensor into 2x2x2 sub-tensors where possible
    and use Strassen's 7-term decomposition for those. Use schoolbook for the rest.
    
    Partitions:
    - I groups: {0,1}, {2,3}
    - J groups: {0,1}, {2,3}, {4}
    - K groups: {0,1}, {2,3}, {4}
    
    For each (Ig, Jg, Kg):
    - If all have size 2: use Strassen (7 terms)
    - Otherwise: use schoolbook (|Ig|*|Jg|*|Kg| terms)
    
    Total rank = 8*7 + 42 = 98 (vs 100 schoolbook)
    """
    deadline = time.time() + 140.0
    
    n, m, p = 4, 5, 5
    
    # Define partitions
    I_groups = [[0, 1], [2, 3]]
    J_groups = [[0, 1], [2, 3], [4]]
    K_groups = [[0, 1], [2, 3], [4]]
    
    # Build U, V, W
    U = [[0] * 100 for _ in range(n * m)]  # temporary size, we'll resize
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    col = 0
    
    for Ig in I_groups:
        for Jg in J_groups:
            for Kg in K_groups:
                if len(Ig) == 2 and len(Jg) == 2 and len(Kg) == 2:
                    # Strassen 2x2x2
                    i0, i1 = Ig[0], Ig[1]
                    j0, j1 = Jg[0], Jg[1]
                    k0, k1 = Kg[0], Kg[1]
                    
                    # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
                    U[i0*m+j0][col] += 1; U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1; V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1; W[k1*n+i1][col] += 1
                    col += 1
                    
                    # m2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
                    U[i1*m+j0][col] += 1; U[i1*m+j1][col] += 1
                    V[j0*p+k0][col] += 1
                    W[k0*n+i1][col] += 1
                    col += 1
                    
                    # m3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
                    U[i0*m+j0][col] += 1
                    V[j0*p+k1][col] += 1; V[j1*p+k1][col] -= 1
                    W[k1*n+i0][col] += 1
                    col += 1
                    
                    # m4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
                    U[i1*m+j1][col] += 1
                    V[j1*p+k0][col] += 1; V[j0*p+k0][col] -= 1
                    W[k0*n+i1][col] += 1
                    col += 1
                    
                    # m5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
                    U[i0*m+j0][col] += 1; U[i0*m+j1][col] += 1
                    V[j1*p+k1][col] += 1
                    W[k1*n+i0][col] += 1
                    col += 1
                    
                    # m6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
                    U[i1*m+j0][col] += 1; U[i0*m+j0][col] -= 1
                    V[j0*p+k0][col] += 1; V[j0*p+k1][col] += 1
                    W[k0*n+i1][col] += 1; W[k1*n+i1][col] += 1
                    col += 1
                    
                    # m7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
                    U[i0*m+j1][col] += 1; U[i1*m+j1][col] -= 1
                    V[j1*p+k0][col] += 1; V[j1*p+k1][col] += 1
                    W[k0*n+i0][col] += 1; W[k1*n+i0][col] += 1
                    col += 1
                else:
                    # Schoolbook
                    for i in Ig:
                        for j in Jg:
                            for k in Kg:
                                U[i*m+j][col] += 1
                                V[j*p+k][col] += 1
                                W[k*n+i][col] += 1
                                col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    rank = col
    
    # Verify correctness
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for all (i,j), (j',k), (k',i'), sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] should be
        # 1 if j==j' and i==i' and k==k', else 0.
        import random
        random.seed(42)
        ok = True
        for _ in range(1000):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            kp = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            
            expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
            actual = float(np.sum(U_arr[i*m+j] * V_arr[jp*p+k] * W_arr[kp*n+ip]))
            if abs(actual - expected) > 1e-9:
                ok = False
                break
        
        if not ok:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
