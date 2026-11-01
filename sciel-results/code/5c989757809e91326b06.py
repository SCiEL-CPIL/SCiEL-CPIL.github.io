import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Use several random test matrices for robustness
        for trial in range(3):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p))
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.allclose(C_check, C_expected, atol=1e-10):
                return False
        return True
    
    best = build_strassen_28()
    best_rank = 28
    
    # Verify the rank-28 solution
    if not verify_decomposition(best["U"], best["V"], best["W"], best_rank, n, m, p):
        # Fallback to schoolbook if Strassen fails
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    # Try to find rank 27 using exact rational arithmetic
    # The matmul tensor T has entries T[a,b,c] where:
    #   a = i*m + j (index into A, i in [0,n), j in [0,m))
    #   b = j'*p + k (index into B, j' in [0,m), k in [0,p))
    #   c = k'*n + i' (index into C in trace convention, k' in [0,p), i' in [0,n))
    # T[a,b,c] = 1 if i=i' and j=j' and k=k', else 0
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # For rank 27, we have 27 unknowns per matrix (4*27 + 16*27 + 16*27 = 1080 unknowns)
    # and 4*16*16 = 1024 equations
    # This is an overdetermined system of bilinear equations
    
    # Strategy: Fix U and V to the Strassen structure but allow W to be adjusted
    # to see if we can eliminate one rank term
    
    # Alternative: Try a structured search where we modify the last block
    # to share a term with the first block
    
    if time.time() < deadline - 30.0:
        # Try: can we merge two Strassen blocks by finding a linear dependency?
        # For 2x2x8, we have 4 blocks of 2x2x2, each with rank 7
        # Total rank = 28
        # To get rank 27, we need one term to be expressible as a combination of others
        
        # Try a targeted approach: use 3 full Strassen blocks (rank 21) 
        # and 1 block with rank 6 (if possible)
        # The 2x2x2 tensor has rank exactly 7, so we can't do rank 6 for a full 2x2x2
        
        # Instead, try a different decomposition:
        # Use the fact that 2x2x8 = 2x2x4 * 2 (two copies)
        # 2x2x4 has a known rank... let's think
        # Actually, let's try a direct rational search for rank 27
        
        # Build the tensor T explicitly
        # T[a,b,c] = 1 iff a corresponds to (i,j), b to (j,k), c to (k,i)
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # T[a,b,c] = 1 if i=i', j=j', k=k'
        
        # For rank r, we need U (4 x r), V (16 x r), W (16 x r) such that
        # for all a,b,c: sum_{t=0}^{r-1} U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
        
        # This is a system of bilinear equations. Let's try to solve it for r=27
        # by fixing most entries and solving for the rest
        
        # Approach: Start from the rank-28 Strassen solution and try to find
        # a linear relation that allows us to eliminate one column
        
        U28 = np.array(best["U"], dtype=Fraction)
        V28 = np.array(best["V"], dtype=Fraction)
        W28 = np.array(best["W"], dtype=Fraction)
        
        # Check if any column of the "tensor product" is linearly dependent
        # For each rank term t, the contribution to the tensor is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # We need to check if one such term is in the span of the others
        
        # This is hard to check directly. Instead, let's try a different approach:
        # Use the exact verification and try random modifications to see if rank 27 works
        
        # Try: can we find a rank-27 solution by modifying the W matrix?
        # Fix U and V to Strassen values, and see if we can solve for W with 27 columns
        
        # Actually, let's try a more systematic approach:
        # The tensor T for 2x2x8 can be written as a sum of rank-1 tensors
        # Each rank-1 tensor corresponds to a choice of (i,j,k) where the 
        # contraction index j is fixed
        
        # Let's try to find if there's a rank-27 solution by checking
        # if the 28 Strassen terms have a linear dependency in the tensor space
        
        # Flatten each rank-1 term into a vector of size 4*16*16 = 1024
        # and check if they are linearly independent
        
        if time.time() < deadline - 20.0:
            # Check linear independence of the 28 rank-1 tensor terms
            # Each term t gives a vector v_t of size 1024 where
            # v_t[a*256 + b*16 + c] = U[a,t] * V[b,t] * W[c,t]
            
            terms = []
            for t in range(28):
                vec = []
                for a in range(4):
                    for b in range(16):
                        for c in range(16):
                            vec.append(U28[a, t] * V28[b, t] * W28[c, t])
                terms.append(vec)
            
            # Check if the 28 vectors are linearly independent
            # Use rational Gaussian elimination
            mat = np.array(terms, dtype=object)  # 28 x 1024 matrix of Fractions
            
            # Do rank computation over Q
            rank = 0
            pivots = []
            for col in range(1024):
                # Find pivot in rows rank..27
                pivot_row = -1
                for row in range(rank, 28):
                    if mat[row, col] != 0:
                        pivot_row = row
                        break
                if pivot_row == -1:
                    continue
                # Swap
                mat[[rank, pivot_row]] = mat[[pivot_row, rank]]
                # Eliminate
                for row in range(28):
                    if row != rank and mat[row, col] != 0:
                        factor = mat[row, col] / mat[rank, col]
                        for col2 in range(col, 1024):
                            mat[row, col2] = mat[row, col2] - factor * mat[rank, col2]
                pivots.append(col)
                rank += 1
                if rank == 28:
                    break
            
            if rank < 28:
                # There's a linear dependency, so we might be able to reduce the rank
                # But this doesn't directly give us a rank-27 decomposition
                # We need to find the actual dependency and use it
                pass
            
            # For now, if no easy reduction is found, return the rank-28 solution
            # A full search for rank 27 would require solving the bilinear system,
            # which is computationally expensive
    
    return best
