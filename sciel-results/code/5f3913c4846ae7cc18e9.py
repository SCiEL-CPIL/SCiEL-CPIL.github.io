"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Non-zero entries of T:
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    nonzero_set = set(nonzero)
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the tensor."""
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    expected = 1 if (a, b, c) in nonzero_set else 0
                    actual = 0.0
                    for r in range(U.shape[1]):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - expected) > 1e-8:
                        return False
        return True
    
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    # Try rank 20 with a structured approach
    # Use the fact that we can express the computation using fewer terms
    # by exploiting the linear structure
    
    # Strategy: For each k (0..5), we compute:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Introduce intermediate values:
    # S0 = A[0,0] + A[0,1], D0 = A[0,0] - A[0,1]
    # S1 = A[1,0] + A[1,1], D1 = A[1,0] - A[1,1]
    #
    # Then: A[0,0] = (S0+D0)/2, A[0,1] = (S0-D0)/2
    #        A[1,0] = (S1+D1)/2, A[1,1] = (S1-D1)/2
    #
    # C[0,k] = (S0+D0)/2 * B[0,k] + (S0-D0)/2 * B[1,k]
    #         = S0*(B[0,k]+B[1,k])/2 + D0*(B[0,k]-B[1,k])/2
    #
    # Let P_k = B[0,k]+B[1,k], Q_k = B[0,k]-B[1,k]
    # C[0,k] = S0*P_k/2 + D0*Q_k/2
    # C[1,k] = S1*P_k/2 + D1*Q_k/2
    #
    # Now we have 12 outputs as linear combinations of:
    # S0*P_k, S0*Q_k, S1*P_k, S1*Q_k, D0*P_k, D0*Q_k, D1*P_k, D1*Q_k
    # That's 8 terms per k, 48 total. Worse!
    #
    # Let me try a different approach. Use the following rank-20 construction:
    # 24 schoolbook terms, but combine 4 pairs using clever cancellation.
    
    # Actually, let's try a direct numerical search for rank 20.
    # We'll use a structured initialization and then refine.
    
    import random
    np.random.seed(42)
    
    best_result = None
    
    for target_rank in [20, 21, 22, 23, 24]:
        found = False
        for attempt in range(50):
            # Initialize with the schoolbook pattern plus random perturbations
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            
            if target_rank == 24:
                # Schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            U[a, col] = 1
                            V[b, col] = 1
                            W[c, col] = 1
                            col += 1
                if verify(U, V, W):
                    best_result = (U, V, W)
                    found = True
                    break
            else:
                # For rank < 24, use a structured ansatz
                # We need to merge some of the 24 terms
                # Try: for each k, use 3 terms instead of 4
                # This gives 18 terms, leaving 2 extra for rank 20
                # 
                # For fixed k, entries are:
                # e1 = (0, k, 2k): A[0,0]*B[0,k]
                # e2 = (1, k, 2k+1): A[1,0]*B[0,k]
                # e3 = (2, 6+k, 2k): A[0,1]*B[1,k]
                # e4 = (3, 6+k, 2k+1): A[1,1]*B[1,k]
                #
                # Try to represent these 4 products with 3 rank-1 terms.
                # This is generally impossible for arbitrary values, but maybe
                # with the right structure across all k's it works.
                
                # Alternative: use rank 20 with 4 "shared" terms and 16 "individual" terms
                # The 4 shared terms handle cross-k interactions
                
                # Let's just try random initialization and check
                U = np.random.randn(dim_u, target_rank)
                V = np.random.randn(dim_v, target_rank)
                W = np.random.randn(dim_w, target_rank)
                
                # Scale to make it more likely to work
                # This is a heuristic; real solution needs structured approach
                
                if verify(U, V, W):
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: schoolbook rank 24
        U = np.zeros((dim_u, 24))
        V = np.zeros((dim_v, 24))
        W = np.zeros((dim_w, 24))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
