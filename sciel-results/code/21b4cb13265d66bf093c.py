"""Matrix multiplication 3,4,7 - low-rank tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen-like decomposition for 2x2x2 blocks to reduce rank
    # Standard schoolbook rank = 3*4*7 = 84
    # We can use a hybrid approach:
    # - Split A into [3,2] x [2,2] blocks
    # - Split B into [2,2] x [2,5] blocks  
    # - Use Strassen on the 2x2x2 core, linear for the rest
    
    # Actually, let's use a simpler approach: direct optimization with random restarts
    # to find a low-rank decomposition.
    
    # Start with a known low-rank construction.
    # For 3x4x7, the minimum rank is known to be at most 22 (by various results).
    # Let's try to construct a rank-22 or lower decomposition.
    
    # Strategy: use the fact that matrix multiplication tensor is sum of rank-1 tensors.
    # We'll use a greedy/random approach to find a good decomposition.
    
    import numpy as np
    
    # Build the full multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # Actually, the tensor is defined by: C = A @ B where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index)
    
    # For a rank-r decomposition, we need matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,k: sum_j U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] summed over c
    # More precisely: sum_c U[i*m+j, c] * V[j*p+k, c] = delta(i,k) * ... 
    
    # Actually, the exact condition is:
    # For all i, k: C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) = sum_c W[k*n+i, c]
    # And we need this to equal sum_j A[i,j] * B[j,k]
    
    # So the tensor entries are: T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] * (some structure)
    
    # Let me think more carefully. The multiplication tensor is a 3-tensor M with:
    # M[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k.
    
    # A rank-1 decomposition is: U[i*m+j, c] * V[j*p+k, c] = M[i,j,k] for a single c
    # But that's not quite right. The decomposition is:
    # M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c]
    # And W[k*n+i, c] is just the output coefficient, which should be 1 for the "diagonal" terms.
    
    # Wait, I need to re-read the problem. The output factor W[k*n+i, r] follows the trace convention.
    # Let me assume the standard formulation:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    # No, that's not right either.
    
    # Standard tensor decomposition for matrix mult:
    # C = A @ B, where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 (scalar)
    # A rank-r decomposition gives: T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Where U is n x m x r, V is m x p x r, W is p x n x r (or similar)
    
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # The condition should be: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] * (something with W)
    
    # Actually, I think the correct formulation is:
    # The multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_k B[j,k] * V[j*p+k, c]) * W[k*n+i, c]
    # No, that doesn't make sense dimensionally.
    
    # Let me just use the standard approach: the tensor T[i,j,k] = 1
    # Rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # With U flattened to (n*m, r), V to (m*p, r), W to (p*n, r)
    
    # So: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k
    # And we want to minimize r.
    
    # This is a hard problem. Let me try a simple approach: use a known good decomposition
    # or do random search.
    
    # For now, let's use the schoolbook (rank 84) as a baseline and try to improve.
    
    # Actually, let me try a different approach: use random initialization and iterative refinement.
    
    start_time = time.time()
    deadline = 140  # leave 10s margin
    
    # Try various ranks, starting from a low estimate
    # Known: for 3x4x7, the rank is likely around 20-30
    
    import numpy as np
    from fractions import Fraction
    
    def try_rank(r, max_iter=100):
        """Try to find a rank-r decomposition."""
        if r < 1:
            return None
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # We need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a non-convex optimization problem.
        
        # Let's use a simple gradient-free approach: random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Normalize
            for c in range(r):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    U[:, c] /= norm_u
                    V[:, c] /= norm_v
                    W[:, c] /= norm_w
            
            # Check quality
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0) ** 2
                return err ** 0.5
            
            err = compute_error(U, V, W)
            if err < 1e-6:
                return U, V, W
        
        return None
    
    # Try ranks from 20 to 40
    for r in range(20, 45):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to fractions for exact representation
            U_frac = [[Fraction(int(round(x * 1000000)), 1000000) if abs(x * 1000000 - round(x * 1000000)) < 1e-6 else Fraction(int(x), 1) for x in U[i]] for i in range(n*m)]
            V_frac = [[Fraction(int(round(x * 1000000)), 1000000) if abs(x * 1000000 - round(x * 1000000)) < 1e-6 else Fraction(int(x), 1) for x in V[i]] for i in range(m*p)]
            W_frac = [[Fraction(int(round(x * 1000000)), 1000000) if abs(x * 1000000 - round(x * 1000000)) < 1e-6 else Fraction(int(x), 1) for x in W[i]] for i in range(p*n)]
            
            # Convert to list of lists with float or int
            def to_list(mat):
                return [[float(x) if isinstance(x, Fraction) else x for x in row] for row in mat]
            
            return {"U": to_list(U_frac), "V": to_list(V_frac), "W": to_list(W_frac)}
    
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
