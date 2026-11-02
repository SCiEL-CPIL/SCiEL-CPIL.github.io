import numpy as np
from itertools import product, combinations
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # Deadline guard: 150s wall-clock limit, keep 10s safety margin
    t_start = time.time()
    deadline = t_start + 140.0

    # The tensor T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We want rank-1 decompositions: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use the "subtraction of linear dependencies" method.
    # Start with rank-18 (schoolbook) and try to find a rank-15 decomposition.
    # The border rank of 2x3x3 matrix multiplication is 15.
    
    # A known rank-15 decomposition for 2x3x3 can be constructed using
    # the border rank construction. Let me try to find it via algebraic search.
    
    # Approach: 
    # The tensor has 2*3*3 = 18 entries.
    # A rank-r decomposition gives r triples (u_r, v_r, w_r) where:
    #   u_r is a vector of length n*m = 6
    #   v_r is a vector of length m*p = 9  
    #   w_r is a vector of length p*n = 6
    # and sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all (i,j,k)
    
    # Key insight: For 2x3x3, we can use the fact that the border rank is 15.
    # One construction: use the identity that allows "folding" 3 terms into 2
    # by introducing a parameter that cancels in the limit.
    
    # Let me try a direct algebraic approach.
    # For 2x3x3, the tensor is:
    # T[i,j,k] = 1 for i=0,1; j=0,1,2; k=0,1,2
    
    # A rank-15 decomposition can be found by:
    # 1. Start with 18 standard rank-1 terms
    # 2. Find 3 linear dependencies among the mode-2 unfoldings
    # 3. Use those to eliminate 3 terms
    
    # Actually, let me try a more direct approach: search for a rank-15 solution
    # with a specific structure.
    
    # Known result: The 2x3x3 matrix multiplication tensor has rank 15.
    # One explicit construction uses the following idea:
    # Group the 18 terms and use polynomial identities to reduce.
    
    # Let me try to construct it explicitly.
    # For 2x3x3, we can write:
    # (A00 A01 A02)   (B00 B01 B02)   (C00 C01 C02)
    # (A10 A11 A12) * (B10 B11 B12) = (C00 C01 C02)
    #                         (B20 B21 B22)   (C10 C11 C12)
    
    # The 18 products are Aij*Bjk for i=0,1; j=0,1,2; k=0,1,2.
    
    # Rank 15 construction (from literature on small matrix multiplication):
    # We can use the following trick: introduce 3 auxiliary variables that
    # allow us to combine 3 pairs of terms.
    
    # Let me try a randomized algebraic search for rank 15.
    # We'll use the fact that we need to solve:
    # sum_{r=1}^{15} U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (a,b,c)
    
    # This is a system of 18 equations in 15*(6+9+6) = 225 unknowns.
    # It's highly underdetermined, so solutions should exist.
    
    # Let me try a structured approach:
    # Fix some entries of U, V, W to simplify the system.
    
    # Strategy: Use the "Landsberg-Strassen" type construction.
    # For 2x3x3, we can use:
    # - 12 "diagonal" terms (one for each (i,j) pair with specific k)
    # - 3 "off-diagonal" correction terms
    
    # Actually, let me just try to find a rank-15 solution by solving a linear system
    # after fixing the structure of U, V, W.
    
    # Fix: Let U have a specific sparse structure.
    # For each rank-1 term r, let U[a,r] be non-zero only for a in some subset.
    
    # Let me try: each u_r has support of size 2 (one from i=0 row, one from i=1 row)
    # or size 1.
    
    # Actually, the simplest approach: try to find rank 15 by solving the
    # system with a specific ansatz.
    
    # Ansatz: 
    # For r = 0..14:
    #   u_r has exactly 2 non-zero entries
    #   v_r has exactly 3 non-zero entries  
    #   w_r has exactly 2 non-zero entries
    # Total non-zeros: 15*(2+3+2) = 105
    
    # This is still complex. Let me try a different approach.
    
    # Use the known rank-15 construction from the literature.
    # The 2x3x3 tensor can be decomposed as follows:
    # 
    # Let me try to hard-code a known rank-15 decomposition if I can find one.
    # 
    # Alternative: Use scipy to solve the system.
    
    # Let me set up the problem as a linear system after fixing V and W.
    # If we fix V and W, then for each (a,b,c):
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # This is linear in U.
    
    # But we need to choose V and W too. This is bilinear.
    
    # Let me try a simpler approach: 
    # Start with rank 18 and try to find linear dependencies.
    
    # The mode-2 unfolding of the tensor is a (n*p) x (m) x (n*p) tensor... 
    # Actually, let me think about this differently.
    
    # For matrix multiplication, the tensor T has T[i,j,k] = 1.
    # The "support" of each rank-1 term in a decomposition is a triple (S_U, S_V, S_W)
    # where S_U subset of {0..5}, S_V subset of {0..8}, S_W subset of {0..5}.
    
    # A term with support (S_U, S_V, S_W) contributes to T[i,j,k] iff
    # i*m+j in S_U, j*p+k in S_V, k*n+i in S_W.
    
    # For the standard decomposition, each term has singleton supports.
    
    # To get rank 15, we need 3 terms to "cover" 6 of the 18 entries (on average 2 each).
    
    # Let me try: use 15 terms where 12 have singleton supports and 3 have
    # larger supports that each cover 2 entries.
    
    # Actually, let me just try to solve this numerically.
    
    # Set up: We want U (6x15), V (9x15), W (6x15) such that
    # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me fix W to be a specific matrix and solve for U, V.
    
    # Fix W to be the identity-like structure:
    # W[k*n+i, r] = 1 if r == k*n+i, else 0.
    # Then W is 6x15 with the first 6 columns being identity.
    
    # Then the equation becomes:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * delta_{r, k*n+i} = 1
    # = U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all (i,j,k)
    
    # So U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # This means U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i]
    
    # But this only uses 6 of the 15 columns of U and V.
    # The other 9 columns must satisfy:
    # sum_{r=6}^{14} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0 for all (i,j,k)
    
    # If W has zeros in columns 6..14, this is automatically satisfied.
    # But then we're just using rank 6, which doesn't work.
    
    # Let me try a different fix for W.
    
    # Actually, let me try a completely different approach:
    # Use the known fact that rank(2x3x3) = 15 and construct it via
    # the "border rank" method.
    
    # The border rank construction for 2x3x3:
    # We can write the tensor as a limit of rank-15 tensors.
    # One explicit construction:
    
    # Let me try to find a rank-15 decomposition by solving a system of equations
    # using numpy's linear algebra.
    
    # Approach: 
    # 1. Choose a random V (9x15) with integer entries
    # 2. For each (i,j,k), the equation is:
    #    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 3. This is linear in U and W given V.
    # 4. But it's bilinear in U and W.
    
    # Let me fix U and solve for W, or vice versa.
    
    # Fix U to be a specific matrix, then the equation is linear in W:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let A[i,j,k] = U[i*m+j, r] * V[j*p+k, r] (depends on r)
    # Then sum_r A_r[i,j,k] * W[k*n+i, r] = 1
    
    # This is linear in W for fixed U, V.
    
    # Let me try: fix U and V to be random integer matrices, then solve for W.
    # If the system is consistent, we have a decomposition.
    
    # The system for W:
    # For each (i,j,k): sum_{r=0}^{14} (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    # This is 18 equations in 90 unknowns (W is 6x15).
    # It should be solvable if the U*V matrix has full rank.
    
    # Let me try this.
    
    def try_rank15():
        """Try to find a rank-15 decomposition."""
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline:
                return None
            
            # Random integer U, V with small entries
            U = np.random.randint(-2, 3, size=(n*m, 15))
            V = np.random.randint(-2, 3, size=(m*p, 15))
            
            # Build the linear system for W
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # W has shape (p*n, 15) = (6, 15)
            # Let W_flat be a vector of length 90
            # The equation for (i,j,k) involves W[k*n+i, :] which is 15 entries
            
            # Actually, W[k*n+i, r] for fixed (k,i) and varying r is a row of W.
            # Let me index W as W[c, r] where c = k*n+i, r = 0..14.
            
            # For each (i,j,k), the equation is:
            # sum_{r=0}^{14} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a linear equation in the 15 entries of W[k*n+i, :].
            # But different (i,j,k) may involve different rows of W.
            
            # Let me collect all equations.
            # There are 18 equations, each involving 15 variables (a row of W).
            # But the rows of W are shared across equations.
            
            # W has 6 rows (for k*n+i where k=0,1,2 and i=0,1).
            # Row k*n+i is involved in equations where (k,i) matches.
            
            # For fixed (k,i), the equations involving W[k*n+i, :] are:
            # For all j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # So for each (k,i), we have m=3 equations in 15 unknowns.
            # 6*(3) = 18 equations total, 6*15 = 90 unknowns.
            
            # Let me solve for each row of W independently.
            W = np.zeros((p*n, 15))
            valid = True
            for k in range(p):
                for i in range(n):
                    # Build the 3x15 system
                    A = np.zeros((m, 15))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    
                    # Solve A * W_row = b
                    # A is 3x15, b is 3x1
                    # Use least squares
                    try:
                        W_row, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        valid = False
                        break
                    
                    W[k*n+i, :] = W_row
                    # Check if the solution is exact
                    if not np.allclose(A @ W_row, b, atol=1e-10):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(15))
                        if abs(val - 1) > 1e-10:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert to lists of Fractions for exact representation
                U_frac = [[Fraction(int(U[a, r])) for r in range(15)] for a in range(n*m)]
                V_frac = [[Fraction(int(V[b, r])) for r in range(15)] for b in range(m*p)]
                W_frac = [[Fraction(int(W[c, r])) for r in range(15)] for c in range(p*n)]
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    result = try_rank15()
    if result is not None:
        return result
    
    return fallback
# EVOLVE-BLOCK-END
