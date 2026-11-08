"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NMP = N * M * P

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    best_rank = NMP
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook decomposition
    U_school = np.zeros((N*M, NMP))
    V_school = np.zeros((M*P, NMP))
    W_school = np.zeros((P*N, NMP))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                idx = i*M + j
                vidx = j*P + k
                widx = k*N + i
                col = i*M*P + j*P + k
                U_school[idx, col] = 1.0
                V_school[vidx, col] = 1.0
                W_school[widx, col] = 1.0

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try to find a lower rank decomposition using tensor train decomposition
    # We construct the 3-mode tensor T[i, j, k] = 1 for all i, j, k
    # This is actually a rank-1 tensor! But wait, the tensor is not T[i,j,k]=1.
    # The matrix multiplication tensor has entries T[i,j,k] where:
    #   T[i, j, k] = 1 if we're looking at the (i,j) entry of A, (j,k) entry of B, 
    #   and we want C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Actually, the tensor is defined as T[i, j, k] = e_{i, j, k} which is 1 for all i, j, k
    # when we're in the standard basis. But that's not right either.
    #
    # Let me reconsider. The matrix multiplication tensor T is a 3-tensor of shape (N, M, P)
    # where T[i, j, k] = 1 if and only if... actually no.
    #
    # The tensor T satisfies: C[i, k] = sum_{j=1}^{M} A[i, j] * B[j, k]
    # The tensor T is defined such that when we take the inner product of T with A (flattened) 
    # and B (flattened), we get C (flattened).
    #
    # More precisely, T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
    # So T[i, j, k] = 1 for all i, j, k (since each product A[i,j]*B[j,k] appears exactly once in C[i,k]).
    #
    # Wait, that means T is the all-ones tensor! That has rank 1!
    # But that can't be right because the rank of matrix multiplication is much higher than 1.
    #
    # I'm confusing myself. Let me think more carefully.
    #
    # The tensor T is of shape (N*P, N*M, M*P) where:
    #   T[i*P+k, j*M+k', l] ... no that's not right.
    #
    # Let me use the standard formulation. The tensor T has indices (i, j, k) where:
    #   i ranges over rows of A and C (1..N)
    #   j ranges over columns of A / rows of B (1..M)
    #   k ranges over columns of B and C (1..P)
    #
    # T[i, j, k] = 1 for all i, j, k.
    #
    # But the tensor rank of the all-ones tensor of shape (N, M, P) is 1!
    #
    # This is clearly wrong. The issue is that the tensor is not simply the all-ones tensor.
    #
    # Let me reconsider the definition. The tensor T is defined by:
    #   T = sum_{i,j,k} e_{i,k} \otimes e_{i,j} \otimes e_{j,k}
    #
    # where e_{i,k} is a basis vector for C, e_{i,j} is a basis vector for A, and e_{j,k} is a basis vector for B.
    #
    # So T is a sum of N*M*P terms, each of which is a rank-1 tensor.
    # The tensor rank is the minimum number of rank-1 tensors needed to express T.
    #
    # For the schoolbook decomposition, we use N*M*P rank-1 tensors.
    #
    # The question is whether we can do better. For N=3, M=5, P=7, the schoolbook rank is 105.
    #
    # Without access to known optimal decompositions, I'll try a simple approach:
    # Use a random low-rank approximation and check if it's exact.
    
    # Try random rank-r decompositions for r < 105
    rng = np.random.RandomState(42)
    
    for r in range(30, NMP):
        if time.time() > deadline - 5.0:
            break
        
        # Construct random U, V, W
        U = rng.randn(N*M, r)
        V = rng.randn(M*P, r)
        W = rng.randn(P*N, r)
        
        # Check if this decomposes the tensor exactly
        # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
        # where delta is 1 for all valid (i,j,k) and 0 otherwise
        # Wait, that's not right either.
        
        # Let me reconsider. The tensor T satisfies:
        # T[i, j, k] = 1 for all i in [0,N), j in [0,M), k in [0,P)
        #
        # And we want: T[i,j,k] = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
        #
        # So we need: sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = 1 for all i, j, k
        
        # Check a few random entries
        valid = True
        for _ in range(100):
            i = rng.randint(0, N)
            j = rng.randint(0, M)
            k = rng.randint(0, P)
            val = np.sum(U[i*M+j] * V[j*P+k] * W[k*N+i])
            if abs(val - 1.0) > 1e-6:
                valid = False
                break
        
        if valid:
            best_rank = r
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break

    # Convert to lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
