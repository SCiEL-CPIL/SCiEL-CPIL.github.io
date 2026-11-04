"""Tensor completion approach for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: try to find a low-rank decomposition
    # Known bounds: for 4x4*4x7, schoolbook is 112
    # We'll try ranks from a reasonable lower bound up to schoolbook
    
    import numpy as np
    from fractions import Fraction
    import sys
    
    # The tensor T[i,j,k] = delta[i+k] (standard matmul tensor)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i+k]
    
    # Strategy: Fix U and V to random integer matrices of rank r,
    # then solve for W via the constraint that the tensor must match.
    # Actually, we need: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta[i+k]
    # This is a linear system in W entries if U,V are fixed.
    
    # Let's try various ranks. Start with known good values.
    # For 4x4*4x7, the minimal rank is at most 112 (schoolbook).
    # Let's try to find something better or equal.
    
    # Actually, let's use a systematic approach:
    # Try rank r from some lower bound to 112
    # For each r, randomly initialize U and V with small integers,
    # then check if a valid W exists (solve linear system)
    
    # The constraint is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i+k)
    # Let's denote A[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c]
    # Then for each (i,j,k): sum_c A[i,j,k,c] * W[k*n+i, c] = delta(i+k)
    
    # This is a system of (n*m*p) equations in (p*n*r) unknowns (W entries).
    # But W entries are W[k*n+i, c] for k in range(p), i in range(n), c in range(r).
    # So unknowns are indexed by (k, i, c) -> p*n*r unknowns.
    # Equations are indexed by (i, j, k) -> n*m*p equations.
    
    # For a given (i,j,k), the equation involves W[k*n+i, c] for all c.
    # So for fixed (k,i), we have equations from all j: 
    #   sum_c A[i,j,k,c] * W[k*n+i, c] = delta(i+k)  for each j
    
    # This means for each (k,i), we have m equations in r unknowns (W[k*n+i, c]).
    # The coefficient matrix for these m equations is M_{k,i}[j,c] = A[i,j,k,c] = U[i*m+j,c]*V[j*p+k,c]
    
    # So we need: for each (k,i), M_{k,i} @ W_col = b where b[j] = delta(i+k)
    # M_{k,i} is m x r, W_col is r x 1, b is m x 1
    
    # For this to have a solution, b must be in the column space of M_{k,i}.
    # If m <= r, we can try to solve. If the system is consistent, we get W.
    
    # Let's try to construct this properly.
    
    def try_rank(r, seed_val=42):
        np.random.seed(seed_val)
        
        # Initialize U and V with small random integers
        U = np.random.randint(-2, 3, size=(n*m, r))
        V = np.random.randint(-2, 3, size=(m*p, r))
        
        # For each (k, i), solve M_{k,i} @ W_col = b
        # M_{k,i}[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # b[j] = 1 if i+k == 0 else 0 ... wait, delta(i+k) means 1 if i==k? 
        # No: in matrix multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T[i,j,k] = 1 if... no, the multiplication tensor has
        # T[i,j,k] such that (A*B)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k? No that's not right either.
        
        # Actually, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k, and the output is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        
        # Wait, I think I'm confusing myself. Let me re-read the problem.
        # "Matrix multiplication 4,4,7" - multiply 4x4 by 4x7 to get 4x7.
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor decomposition: C[i,k] = sum_c U[i*m+j, c]... no.
        
        # Standard: C[i,k] = sum_c sum_j U[i,j,c] V[j,k,c] W[c]... 
        # Actually the standard Strassen-style: C[i,k] = sum_c U_i,c * V_k,c * ... 
        # No. The tensor is: C[i,k] = sum_{j,c} (U[i,j,c] A[i,j]) (V[j,k,c] B[j,k]) 
        # Hmm, that's not standard either.
        
        # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # So the product is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
        # And this should equal C[i,k] = sum_j A[i,j] * B[j,k]
        
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta... 
        # Wait, for the identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give
        # a tensor such that sum_j [that] * A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]
        
        # This means: for each (i,k), sum_j [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
        
        # By linearity, this requires: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k? 
        # No, that would give sum_j A[i,j]B[j,k] * 1 = sum_j A[i,j]B[j,k]. Yes!
        
        # Wait, so we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for ALL i,j,k?
        # That doesn't seem right for rank minimization. Let me reconsider.
        
        # Actually no. The standard matrix multiplication tensor is:
        # T[i,j,k] where the multiplication is C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all valid i,j,k, and the formula is
        # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
        
        # A rank-1 decomposition of this tensor would be:
        # T[i,j,k] = sum_c u[i,j,c] v[j,k,c] w[k,i,c]... 
        # Hmm, the indices don't quite work out. Let me think again.
        
        # The way the problem states it: U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
        
        # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        # This is a system of n*m*p = 4*4*7 = 112 equations.
        # Unknowns: U has n*m*r = 16r entries, V has m*p*r = 28r entries, W has p*n*r = 28r entries.
        # Total unknowns: 72r.
        
        # For r = 2, 72*2 = 144 > 112, so potentially solvable.
        # But it's a nonlinear system (trilinear).
        
        # The approach: fix U and V, then the equations become linear in W.
        # For each (i,j,k): sum_c [U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i, c] = 1
        
        # Group by (k,i): for fixed k,i, and varying j:
        # sum_c M[j,c] * W[k*n+i, c] = 1 where M[j,c] = U[i*m+j,c]*V[j*p+k,c]
        # This is m equations in r unknowns (W[k*n+i, c] for c=0..r-1).
        # Coefficient matrix M is m x r = 4 x r.
        # RHS is all ones (length m=4).
        
        # For a solution to exist, the all-ones vector must be in the column space of M.
        # If r >= 4, M can be full column rank (if generic), and we can solve.
        # If r < 4, we need special structure.
        
        # Let's try r from small values upward.
        
        for r in range(2, 113):
            # Try multiple random seeds
            for seed_val in range(20):
                np.random.seed(seed_val)
                U = np.random.randint(-3, 4, size=(n*m, r))
                V = np.random.randint(-3, 4, size=(m*p, r))
                
                W = np.zeros((p*n, r), dtype=float)
                valid = True
                
                for k in range(p):
                    for i in range(n):
                        # Build M: m x r
                        M = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        
                        b = np.ones(m)
                        
                        # Solve M @ W_col = b
                        # Use least squares
                        try:
                            sol, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
                            # Check if solution is exact
                            if np.allclose(M @ sol, b, atol=1e-10):
                                W[k*n+i, :] = sol
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Verify the full decomposition
                    # Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(r):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                err = abs(val - 1)
                                max_err = max(max_err, err)
                    
                    if max_err < 1e-8:
                        # Convert to fractions for exactness
                        U_frac = [[Fraction(int(U[i, c])) for c in range(r)] for i in range(n*m)]
                        V_frac = [[Fraction(int(V[i, c])) for c in range(r)] for i in range(m*p)]
                        W_frac = [[Fraction(W[i, c]).limit_denominator(1000000) for c in range(r)] for i in range(p*n)]
                        
                        # Verify with fractions
                        max_err_frac = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = Fraction(0)
                                    for c in range(r):
                                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                                    if val != 1:
                                        max_err_frac = 1
                        if max_err_frac == 0:
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # If we didn't find anything, fall back to schoolbook
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
    
    result = try_rank(112)
    return result
# EVOLVE-BLOCK-END
