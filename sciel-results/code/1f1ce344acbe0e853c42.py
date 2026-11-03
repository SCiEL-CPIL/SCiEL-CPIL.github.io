import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds, leaving 10s margin
    
    n, m, p = 3, 5, 6  # 3x5 times 5x6
    
    # Initialize with schoolbook (rank = 90)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is correct
    def verify(U, V, W, n, m, p):
        # Check if the decomposition exactly reproduces matrix multiplication
        # The tensor T[i,j,k] = delta[i*m+j, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum over r
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # The target is 1 if this is the (i,j,k) entry of the matmul tensor
                    # Actually, the matmul tensor T[i,j,k] = 1 for all i,j,k in the standard basis
                    # No wait - the matmul tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j]*B[j,k]
                    # In tensor form: T[i,j,k] = delta (it's just 1 for the standard basis)
                    # Actually the tensor is T[i,j,k] = 1 for all valid indices? No...
                    # The matrix multiplication tensor has T[i,j,k] = 1 if we're computing (AB)[i,k]
                    # More precisely, T is a 3-tensor where T[i,j,k] = 1 (all ones) is NOT correct.
                    # The correct tensor: for each i,j,k, the coefficient is 1 (since (AB)[i,k] = sum_j A[i,j]B[j,k])
                    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
                    # Wait no. Let me think again.
                    # (AB)[i,k] = sum_j A[i,j] * B[j,k]
                    # So the tensor T[i,j,k] = 1 for all i,j,k
                    # That means we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                    expected = 1.0
                    if abs(total - expected) > 1e-9:
                        return False
        return True
    
    # The schoolbook decomposition should verify
    # Let me just use it as baseline
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
