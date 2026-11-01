"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    # 2 Strassen blocks (rank 7 each) for first 4 cols, 1 Schoolbook (rank 4) for last col
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank, tol=1e-10):
        """Verify that U,V,W with given rank exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m + j][c] * V[jp*p + k][c] * W[k*n + i][c]
                        if abs(val - target) > tol:
                            return False
        return True

    def try_algebraic_elimination(rank, max_attempts=20):
        """
        Try to find a rank-r decomposition by treating it as a sparse linear system
        and using rational Gaussian elimination with randomized sparsity patterns.
        
        The tensor equation is:
          sum_c U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c] = delta_{j,jp}
        
        For each term c, we have a rank-1 contribution. We parameterize with
        structured sparsity that breaks block-diagonal symmetry.
        """
        if time.time() > DEADLINE - 5:
            return None
        
        # Total number of equations: n * m * m * p = 2*2*2*5 = 40
        # Number of unknowns per factor entry: rank
        # U has n*m = 4 entries per column, V has m*p = 10, W has p*n = 10
        # Total unknowns = rank * (4 + 10 + 10) = 24 * rank
        
        # For rank 17: 408 unknowns, 40 equations -> very underdetermined
        # For rank 16: 384 unknowns, 40 equations
        
        # Strategy: fix a sparse pattern, solve linearly where possible,
        # then verify. Use randomization to explore different patterns.
        
        import random
        
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                return None
            
            random.seed(attempt * 7919 + rank * 104729)
            
            # Generate a sparse pattern: for each (factor, index, column), decide if it's zero
            # Use a structured approach: each column c has a "type" that determines sparsity
            
            U = [[Fraction(0)] * rank for _ in range(n * m)]
            V = [[Fraction(0)] * rank for _ in range(m * p)]
            W = [[Fraction(0)] * rank for _ in range(p * n)]
            
            # Assign random values to a subset of entries to create a "seed" pattern
            # Then try to complete the rest via linear constraints
            
            # For each column, pick which entries are non-zero
            # Use a pattern where each column touches a random subset of U,V,W indices
            
            for c in range(rank):
                # Each column: pick 1-2 entries in U, 1-3 in V, 1-3 in W
                n_u = random.randint(1, 2)
                n_v = random.randint(1, 3)
                n_w = random.randint(1, 3)
                
                # Pick indices
                u_indices = random.sample(range(n*m), min(n_u, n*m))
                v_indices = random.sample(range(m*p), min(n_v, m*p))
                w_indices = random.sample(range(p*n), min(n_w, p*n))
                
                # Assign random small rational values
                for idx in u_indices:
                    val = random.choice([1, -1, Fraction(1,2), Fraction(-1,2), Fraction(1,3), Fraction(2,3)])
                    U[idx][c] = val
                for idx in v_indices:
                    val = random.choice([1, -1, Fraction(1,2), Fraction(-1,2), Fraction(1,3), Fraction(2,3)])
                    V[idx][c] = val
                for idx in w_indices:
                    val = random.choice([1, -1, Fraction(1,2), Fraction(-1,2), Fraction(1,3), Fraction(2,3)])
                    W[idx][c] = val
            
            # Now we have a partially filled decomposition. Check if it's exact.
            # If not, we could try to adjust, but for now just verify.
            
            # Convert to float for quick check
            U_f = [[float(x) for x in row] for row in U]
            V_f = [[float(x) for x in row] for row in V]
            W_f = [[float(x) for x in row] for row in W]
            
            if verify_decomposition(U_f, V_f, W_f, rank, tol=1e-10):
                # Found a valid decomposition! Convert back to list of ints/halves
                U_out = [[int(x.numerator) // int(x.denominator) if x.denominator == 1 else float(x) for x in row] for row in U]
                V_out = [[int(x.numerator) // int(x.denominator) if x.denominator == 1 else float(x) for x in row] for row in V]
                W_out = [[int(x.numerator) // int(x.denominator) if x.denominator == 1 else float(x) for x in row] for row in W]
                
                # Ensure all are integer or half-integer
                def snap_val(x):
                    if isinstance(x, Fraction):
                        if x.denominator == 1:
                            return int(x.numerator)
                        elif x.denominator == 2:
                            return float(x)
                        else:
                            return float(x)
                    return x
                
                U_out = [[snap_val(x) for x in row] for row in U]
                V_out = [[snap_val(x) for x in row] for row in V]
                W_out = [[snap_val(x) for x in row] for row in W]
                
                return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    def try_systematic_search(rank, max_time=10.0):
        """
        More systematic search: use the fact that for matrix multiplication,
        we can fix U to be identity-like and solve for V and W.
        
        Actually, let's use a different approach: exploit the structure.
        For 2x2 * 2x5, the tensor is T[i,j,jp,k] = delta_{j,jp}.
        
        We can group by j: for each j, the sub-tensor over (i, jp, k) with jp=j
        gives us n*p = 10 entries that should be 1, and the rest 0.
        
        Let's try: fix U to have specific structure, then solve linearly for V,W.
        """
        if time.time() > DEADLINE - 5:
            return None
        
        import random
        start_t = time.time()
        
        for attempt in range(50):
            if time.time() - start_t > max_time or time.time() > DEADLINE - 5:
                break
            
            random.seed(attempt * 104729 + rank)
            
            # Strategy: For each column c, the product U[:,c] * V[:,c] * W[:,c]
            # contributes a rank-1 tensor. We need sum of these = target.
            
            # Fix U to be "generic" sparse, then for each equation, we get a linear
            # constraint on V and W entries.
            
            # Simpler: try to find a decomposition where each column c has
            # exactly one non-zero in U, one in V, one in W (completely sparse).
            # Then each column contributes U_val * V_val * W_val to exactly one
            # tensor entry. We need to cover all 40 non-zero entries (actually 20:
            # for each j, there are n*p=10 entries with jp=j that are 1).
            
            # Wait, the target tensor has T[i,j,jp,k] = 1 if j==jp else 0.
            # Non-zero entries: for each (i,j,k) with jp=j, T=1. That's 2*2*5 = 20 entries.
            
            # If each column contributes to exactly one tensor entry, we need at least 20 columns.
            # But we want rank < 18, so we need columns that contribute to multiple entries.
            
            # Let's try: each column has 2 non-zeros in U, 2 in V, 2 in W.
            # Then it can contribute to up to 2*2*2 = 8 tensor entries.
            
            # For rank 17: 17 * 8 = 136 > 20, so it's feasible in principle.
            
            # Let's do a more direct approach: randomly initialize and use
            # coordinate descent or just check random sparse patterns.
            
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            # Random sparse initialization
            for c in range(rank):
                # Pick 2 entries in U
                for _ in range(2):
                    idx = random.randint(0, n*m-1)
                    U[idx][c] = random.choice([1.0, -1.0, 0.5, -0.5])
                # Pick 2 entries in V
                for _ in range(2):
                    idx = random.randint(0, m*p-1)
                    V[idx][c] = random.choice([1.0, -1.0, 0.5, -0.5])
                # Pick 2 entries in W
                for _ in range(2):
                    idx = random.randint(0, p*n-1)
                    W[idx][c] = random.choice([1.0, -1.0, 0.5, -0.5])
            
            if verify_decomposition(U, V, W, rank, tol=1e-10):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try rank 17 first
    if time.time() < DEADLINE - 15:
        result = try_systematic_search(17, max_time=10.0)
        if result is not None:
            return result
    
    # Try rank 16
    if time.time() < DEADLINE - 15:
        result = try_systematic_search(16, max_time=10.0)
        if result is not None:
            return result
    
    # Try algebraic elimination for rank 17
    if time.time() < DEADLINE - 10:
        result = try_algebraic_elimination(17, max_attempts=30)
        if result is not None:
            return result
    
    # Fallback to rank 18
    return get_rank_18_solution()

# EVOLVE-BLOCK-END
