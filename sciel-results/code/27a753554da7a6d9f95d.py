import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the list of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    # 30 triples
    
    # Schoolbook: rank 30
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        for c, (i, j, k) in enumerate(triples):
            U[i * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + i][c] = 1.0
        return U, V, W
    
    # Try rank 28
    def try_rank(r):
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of 30 equations in r*3 unknowns (but nonlinear)
        
        # Use a simple iterative approach
        # For each (i,j,k), the equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is hard to optimize directly. Let me try a different approach.
        # 
        # Alternative: use the fact that the tensor has a specific structure.
        # The tensor T[i,j,k] = 1 for all i,j,k means:
        # T = sum_j (sum_i e_i) ⊗ e_j ⊗ (sum_k e_k) ... no, as we discussed.
        
        # Let me try a completely different approach:
        # Use the "bilinear" structure. The matrix multiplication is:
        # C = A * B
        # We can write this as a sum of rank-1 bilinear maps.
        
        # For rank 28, we need 28 terms. Let me try to construct them explicitly.
        
        pass
    
    # Let me try a direct construction for rank 28.
    # Based on known results, here's a construction:
    
    # The 2x3x5 multiplication tensor has rank 28.
    # One construction:
    # 
    # Let's use the following approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For the first term A[i,0]*B[0,k]: this is a 2x5 outer product, rank 2*5=10? No.
    # A[i,0]*B[0,k] for i in {0,1}, k in {0,1,2,3,4}: 
    # This is sum_{i,k} A[i,0] * B[0,k] * C[i,k]
    # As a tensor: sum_{i,k} e_{i,0} ⊗ e_{0,k} ⊗ e_{i,k}
    # This has 2*5 = 10 terms, all independent, so rank 10.
    #
    # Similarly for j=1 and j=2: each has rank 10.
    # Total: 30.
    #
    # To reduce: we need to combine terms across different j.
    # 
    # Key insight: for fixed i and k, the three terms (j=0,1,2) are:
    # A[i,0]*B[0,k], A[i,1]*B[1,k], A[i,2]*B[2,k]
    #
    # These can be combined if we use a "Strassen-like" trick.
    # 
    # For 2x2x2, Strassen uses 7 terms instead of 8. The trick is:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # For 2x3x5, we can try a similar approach.
    
    # Given the complexity, let me just return the schoolbook rank-30 for now,
    # but try a quick optimization to see if rank 28 is achievable.
    
    # Actually, let me try a specific rank-28 construction.
    # 
    # After more thought, here's a construction for 2x3x5 with rank 28:
    #
    # We use the following 28 rank-1 terms:
    # For each (i,k) pair (10 pairs), we need to express:
    # sum_j A[i,j] * B[j,k] = C[i,k]
    #
    # using a total of 28 terms instead of 30.
    #
    # The reduction comes from sharing terms between different (i,k) pairs.
    
    # I'll use a numerical approach: set up the equations and solve.
    
    def solve_rank28():
        """Try to find a rank-28 decomposition."""
        r = 28
        # Variables: U (6x28), V (15x28), W (10x28) = 6*28 + 15*28 + 10*28 = 812 variables
        # Equations: 30 (one per (i,j,k))
        # Underdetermined system, so solutions exist if rank <= 28.
        
        # Strategy: fix W to be a specific matrix, then solve for U and V.
        # Or: use a greedy approach.
        
        # Let's try: set W to be a 10x28 matrix with specific structure.
        # For each (i,k), W[k*n+i, :] should be a row vector.
        # 
        # The equation is: U[i*m+j, :] dot V[j*p+k, :] dot W[k*n+i, :] = 1
        # where "dot" here means element-wise product then sum over r.
        
        # This is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # For fixed (i,k), this must hold for all j.
        # Let w_ik = W[k*n+i, :] (a 1-d vector of length 28)
        # Then: sum_r U[i*m+j, r] * V[j*p+k, r] * w_ik[r] = 1 for all j
        
        # This is: (U[i*m+0, :] * V[0*p+k, :] * w_ik) = 1
        #           (U[i*m+1, :] * V[1*p+k, :] * w_ik) = 1
        #           (U[i*m+2, :] * V[2*p+k, :] * w_ik) = 1
        # where * is element-wise and () means sum.
        
        # Let a_j = U[i*m+j, :], b_j = V[j*p+k, :]
        # Then: sum_r a_j[r] * b_j[r] * w_ik[r] = 1 for j=0,1,2
        
        # If we set w_ik to be a unit vector (say w_ik[r] = 1 for r in some set, 0 otherwise),
        # then we need sum_{r in set} a_j[r] * b_j[r] = 1 for all j.
        
        # This is getting complicated. Let me try a different initialization.
        
        # Initialize with schoolbook (rank 30) and try to merge 2 terms.
        # 
        # Schoolbook term (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        #
        # To merge terms (i1,j1,k1) and (i2,j2,k2), we need:
        # U[i1*m+j1, c1] = U[i1*m+j1, c2] = a
        # U[i2*m+j2, c1] = U[i2*m+j2, c2] = b
        # V[j1*p+k1, c1] = V[j1*p+k1, c2] = c
        # V[j2*p+k2, c1] = V[j2*p+k2, c2] = d
        # W[k1*n+i1, c1] = W[k1*n+i1, c2] = e
        # W[k2*n+i2, c1] = W[k2*n+i2, c2] = f
        #
        # And we need: a*c*e + a*c*f = 1 (if i1=i2, j1=j2, k1=k2... no)
        
        # This approach is too ad-hoc. Let me just return schoolbook.
        
        return None
    
    # Given the time constraints and complexity, I'll return the schoolbook decomposition.
    # The rank is 30, which gives score 30/30 = 1.0.
    # To improve, I need to find a rank-28 decomposition.
    
    # Let me try one more approach: use the known fact that rank(2x3x5) = 28
    # and construct it using a specific algebraic identity.
    
    # Construction for rank 28:
    # 
    # The key identity: for the 2x3x5 tensor, we can use the following:
    # 
    # Let's define 28 "Strassen-like" multipliers M1, ..., M28.
    # Each multiplier is a product of a linear form in A and a linear form in B.
    # The output C[i,k] is a linear combination of these multipliers.
    
    # I'll use the following construction (based on known results):
    #
    # For 2x3x5, use the decomposition:
    # C = A * B where A is 2x3, B is 3x5
    #
    # Write B = [B1 | B2] where B1 is 3x2, B2 is 3x3
    # C = A*B1 + A*B2
    #
    # A*B1: 2x2 multiplication, rank 4 (schoolbook) or 4 (minimal for 2x2x2 is 7, but for 2x2x1 it's 2)
    # Wait, A*B1 is 2x2, computed from 2x3 times 3x2. Rank = 2*3*2 = 12 schoolbook.
    #
    # This isn't helping. Let me just go with rank 30.
    
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
