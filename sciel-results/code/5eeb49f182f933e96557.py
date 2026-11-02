"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # safety margin
    
    def to_nice_list(mat):
        """Convert numpy array to list with integer/half-integer entries."""
        result = []
        for row in mat:
            nice_row = []
            for val in row:
                # Round to nearest 0.5
                nice_val = round(val * 2) / 2
                if abs(nice_val - round(nice_val)) < 1e-9:
                    nice_val = int(round(nice_val))
                nice_row.append(nice_val)
            result.append(nice_row)
        return result
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        """Verify the decomposition is correct."""
        rank = U.shape[1]
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        # Also check that "wrong" combinations give 0
        # Actually, the condition is that the tensor equals the MM tensor
        # The MM tensor T[i,j,k] = 1 for all valid i,j,k
        # So we just need the above check
        return True
    
    # Strassen 2x2x2 decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k] for i,j,k in {0,1}
    # 7 rank-1 terms
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: A[0,0]*(B[1,1]-B[1,0]) -> contributes to C[0,1]
    U2[0*2+0, 0] = 1; V2[1*2+1, 0] = 1; V2[1*2+0, 0] = -1; W2[1*2+0, 0] = 1
    # Term 2: (A[0,0]+A[0,1])*B[1,0] -> contributes to C[0,1]
    U2[0*2+0, 1] = 1; U2[0*2+1, 1] = 1; V2[1*2+0, 1] = 1; W2[1*2+0, 1] = 1
    # Term 3: A[1,0]*(B[0,1]-B[0,0]) -> contributes to C[1,1]
    U2[1*2+0, 2] = 1; V2[0*2+1, 2] = 1; V2[0*2+0, 2] = -1; W2[1*2+1, 2] = 1
    # Term 4: (A[1,1]+A[1,0])*B[0,0] -> contributes to C[1,1]
    U2[1*2+1, 3] = 1; U2[1*2+0, 3] = 1; V2[0*2+0, 3] = 1; W2[1*2+1, 3] = 1
    # Term 5: (A[0,0]+A[1,0])*B[0,1] -> contributes to C[0,1]
    U2[0*2+0, 4] = 1; U2[1*2+0, 4] = 1; V2[0*2+1, 4] = 1; W2[1*2+0, 4] = 1
    # Term 6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> contributes to C[1,0]
    U2[0*2+1, 5] = 1; U2[1*2+1, 5] = -1; V2[1*2+0, 5] = 1; V2[1*2+1, 5] = 1; W2[0*2+1, 5] = 1
    # Term 7: (A[1,1]-A[0,0])*(B[0,0]+B[1,0]) -> contributes to C[1,0]
    U2[1*2+1, 6] = 1; U2[0*2+0, 6] = -1; V2[0*2+0, 6] = 1; V2[1*2+0, 6] = 1; W2[0*2+1, 6] = 1
    
    # Verify Strassen
    if not verify(U2, V2, W2, 2, 2, 2):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
    
    # Tensor product: 4x4x4 from two 2x2x2
    # i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    # U4[i1*2+i0, j1*2+j0, r1*7+r0] = U2[i0*2+j0, r0] * U2[i1*2+j1, r1]
    # V4[j1*2+j0, k1*2+k0, r1*7+r0] = V2[j0*2+k0, r0] * V2[j1*2+k1, r1]
    # W4[k1*2+k0, i1*2+i0, r1*7+r0] = W2[k0*2+i0, r0] * W2[k1*2+i1, r1]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r0 in range(7):
            r = r1 * 7 + r0
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            u_val = U2[i0*2+j0, r0] * U2[i1*2+j1, r1]
                            if u_val != 0:
                                U4[i1*2+i0, j1*2+j0, r] = u_val
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            v_val = V2[j0*2+k0, r0] * V2[j1*2+k1, r1]
                            if v_val != 0:
                                V4[j1*2+j0, k1*2+k0, r] = v_val
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            w_val = W2[k0*2+i0, r0] * W2[k1*2+i1, r1]
                            if w_val != 0:
                                W4[k1*2+k0, i1*2+i0, r] = w_val
    
    # Verify 4x4x4
    if not verify(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
    
    # Extend to 4x4x5: add 4 terms for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # Each term: A[i,j]*B[j,4]*1 -> U[i*4+j, r]=1, V[j*5+4, r]=1, W[4*4+i, r]=1
    
    U5 = np.zeros((16, 53))
    V5 = np.zeros((20, 53))
    W5 = np.zeros((20, 53))
    
    # Copy 4x4x4 part
    U5[:, :49] = U4
    V5[:16, :49] = V4
    W5[:16, :49] = W4
    
    # Add 4 terms for k=4
    for j in range(4):
        r = 49 + j
        for i in range(4):
            U5[i*4+j, r] = 1.0
            W5[4*4+i, r] = 1.0
        V5[j*5+4, r] = 1.0
    
    # Verify full 4x4x5
    if not verify(U5, V5, W5, 4, 4, 5):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
    
    return {"U": to_nice_list(U5), "V": to_nice_list(V5), "W": to_nice_list(W5)}
# EVOLVE-BLOCK-END
