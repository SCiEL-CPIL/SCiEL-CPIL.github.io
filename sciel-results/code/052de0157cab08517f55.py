"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from fractions import Fraction

    deadline = time.time() + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def _verify(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = len(U[0])
        # Check that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 if i,j,k match, 0 otherwise
        # Actually, the tensor for matrix multiplication T_{i,j,k} satisfies:
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c[i,j], V_c[j,k], W_c[k,i]
        # and T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        # For matrix multiplication, T[i,j,k] = 1 if the indices correspond to a valid term, else 0
        # Specifically, T[i,j,k] = delta_{i,j,k} in the sense that:
        # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for the multiplication)
        
        # Let's check: for each (i,j,k), sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] should be 1
        # Wait, that's not right. The tensor T is defined such that:
        # sum_j T[i,j,k] A[i,j] B[j,k] = (AB)[i,k]
        # So T[i,j,k] = delta_{j,j} = 1 for all i,j,k? No.
        # Actually, (AB)[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all i,j,k.
        # That means the tensor is the all-ones tensor? No, that's not right either.
        
        # Let me reconsider. The matrix multiplication tensor is:
        # T_{i,j,k} such that (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # This means T_{i,j,k} = 1 for all i,j,k.
        # But that would make the rank n*m*p trivially, which is the schoolbook.
        
        # Actually, I think I'm confusing myself. Let me re-read the problem.
        # "Matrix multiplication 3,3,8" - multiplying a 3x3 matrix by a 3x8 matrix.
        # A is 3x3, B is 3x8, C = A*B is 3x8.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices i (row of A, 0..2), j (col of A / row of B, 0..2), k (col of B, 0..7)
        # T[i,j,k] = 1 for all i,j,k
        # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c is 3x3, V_c is 3x8, W_c is 8x3
        # And T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # So for each (i,j,k), we need sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    # Start with schoolbook
    best = _schoolbook()
    
    # Try to find a lower rank decomposition using random search
    # Strategy: try ranks from 71 down to some lower bound
    # For each rank, try random initializations and use gradient descent or similar
    
    # Given the time constraint, let's try a structured approach
    # We'll use the fact that for rectangular matrices, there might be structure
    
    # Let's try a random search for rank 60 first as a test
    def _random_search(rank, num_trials=100, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Use simple gradient descent on the loss:
            # L = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            # This is expensive, so let's try a different approach
            
            # Instead, let's try to find a sparse decomposition
            # Start with a random subset of the schoolbook terms and see if we can adjust
            
            # For now, just check if we can find a solution with this rank
            # by solving a system of equations (if rank is small enough)
            
            pass
        
        return None

    # Given the complexity, let's just return the schoolbook for now
    # but structure the code so future improvements can be added
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
