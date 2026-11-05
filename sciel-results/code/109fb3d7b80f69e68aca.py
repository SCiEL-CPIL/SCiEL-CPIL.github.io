import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition using integer arithmetic to avoid floating point issues."""
        # Convert to fractions for exact arithmetic
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if val != Fraction(1):
                        return False
        return True

    # Structural approach: Use a systematic construction
    # Key insight: The problem has 72 constraints (one for each (i,j,k) triple)
    # Each rank-1 term contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Try a block-structured decomposition:
    # Partition k into groups and use Strassen-like methods within each group
    
    # For now, try a smarter random search with constraint-aware initialization
    # Use the fact that we need sum_c U_ik V_jk W_ki = 1 for all (i,j,k)
    
    np.random.seed(42)
    
    # Try a structured approach: initialize with schoolbook and try to merge terms
    # Or: use a greedy algorithm to build the decomposition
    
    # New structural approach: Use a "basis" of rank-1 tensors and solve for coefficients
    # Each rank-1 tensor is determined by (U_vec, V_vec, W_vec)
    # We need: sum_c alpha_c * T_c = T_target where T_target[i,j,k] = 1
    
    # Try random rank-1 tensors with small integer entries and solve the linear system
    # for the coefficients (but coefficients must be 1 for each term in the sum)
    
    # Actually, the decomposition is: T = sum_c U_c ⊗ V_c ⊗ W_c where each is a rank-1 tensor
    # So we need to find vectors u_c, v_c, w_c such that sum_c u_c ⊗ v_c ⊗ w_c = T
    
    # This is a nonlinear problem. Let's try a different structural approach.
    
    # Use the fact that for fixed (i,j), we need: sum_c U_ij,c * V_jk,c * W_ki,c = 1 for all k
    # This means: for fixed (i,j), the vector (sum_c U_ij,c * V_jk,c * W_ki,c)_{k=0}^{7} = (1,1,...,1)
    
    # Try a systematic search over small coefficient sets
    # Use constraint propagation: for each (i,j), track what's needed
    
    # Given time, let's try a more efficient random search with better verification
    # and a smaller search space
    
    # Use half-integers: multiply everything by 2 to work with integers
    # Then U', V', W' are integers and the constraint becomes:
    # sum_c U'_ij,c * V'_jk,c * W'_ki,c = 8 (since each product is divided by 8)
    # Wait, if U = U'/2, V = V'/2, W = W'/2, then product = U'V'W'/8
    # So sum_c U'V'W'/8 = 1 means sum_c U'V'W' = 8
    
    # Let's work with integers directly
    coeff_set = [-2, -1, 0, 1, 2]  # integer coefficients (will divide by 2 at the end)
    
    # Try ranks from 71 down, with more attempts for lower ranks
    for rank in range(71, 50, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 5 if rank < 65 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random integer coefficients
            U = np.random.choice(coeff_set, (n * m, rank))
            V = np.random.choice(coeff_set, (m * p, rank))
            W = np.random.choice(coeff_set, (p * n, rank))
            
            # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 8
            # (because we'll divide by 2 later, so 8/8 = 1)
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if val != 8:
                            valid = False
                            break
            
            if valid:
                # Convert back to half-integers
                U_half = U.astype(float) / 2.0
                V_half = V.astype(float) / 2.0
                W_half = W.astype(float) / 2.0
                
                best_rank = rank
                best_result = {
                    "U": [[float(x) for x in row] for row in U_half],
                    "V": [[float(x) for x in row] for row in V_half],
                    "W": [[float(x) for x in row] for row in W_half]
                }
                break
    
    return best_result
