"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    def verify_exact(U, V, W):
        """Verify exactness using rational arithmetic."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        rank = len(U_frac[0]) if U_frac else 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_numeric(U, V, W):
        """Verify with random matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    def try_milp_decomposition(target_rank, max_coeff=3, time_limit=30):
        """
        Try to find a rank-target_rank decomposition using MILP.
        Binary variables select which (i,j,k) each rank term contributes to.
        Coefficient variables are bounded integers.
        """
        if time.time() - start_time > deadline:
            return None
        
        import pulp
        
        prob = pulp.LpProblem("tensor_decomp", pulp.LpMinimize)
        
        # Coefficient domains: -max_coeff to max_coeff
        coeff_vals = list(range(-max_coeff, max_coeff + 1))
        
        # Variables: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # For each rank term r, we need coefficients for all (i,j,k) triples
        # But many will be zero. We use binary selection + integer coeff.
        
        # Simpler approach: directly use integer variables bounded by [-max_coeff, max_coeff]
        # with the constraint that for each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # This is still nonlinear (product of three variables). 
        # Instead, use a different formulation.
        
        # Alternative: Fix W to small values and solve for U, V.
        # Or use the fact that we can enumerate W patterns.
        
        # Let's try a structured approach:
        # For 3x3x5, try to use Laderman-like structure for the 3x3x3 part
        # and extend to 3x3x5.
        
        # Laderman's 3x3x3 decomposition has rank 23.
        # For 3x3x5, we can think of it as 3x3x3 + 3x3x2.
        # The 3x3x2 part needs at least 3*3*2 = 18 terms in schoolbook,
        # but can share with the 3x3x3 part.
        
        # Let's implement Laderman's algorithm for 3x3x3 and extend.
        
        # Laderman's 3x3x3:
        # The 23 terms are defined by specific patterns.
        # Let me hardcode a known rank-23 decomposition for 3x3x3.
        
        # Actually, let's try a direct search with structured initialization.
        
        # Strategy: For each rank r from target_rank down, try to find
        # a valid decomposition using random restarts with verification.
        
        if time.time() - start_time > deadline:
            return None
        
        best_result = None
        
        # Try different coefficient ranges
        for coeff_range in [2, 3, 4]:
            if time.time() - start_time > deadline:
                break
            
            # Random search with structured initialization
            for seed in range(50):
                if time.time() - start_time > deadline:
                    break
                
                rng = np.random.RandomState(seed * 1000 + target_rank)
                
                U = np.zeros((n * m, target_rank), dtype=int)
                V = np.zeros((m * p, target_rank), dtype=int)
                W = np.zeros((p * n, target_rank), dtype=int)
                
                # Initialize with random small integers
                for r in range(target_rank):
                    # Each rank term should contribute to a specific (i,j,k) pattern
                    # Randomly assign which (i,j,k) this term affects
                    for _ in range(rng.randint(1, 4)):
                        i = rng.randint(0, n)
                        j = rng.randint(0, m)
                        k = rng.randint(0, p)
                        u_val = rng.randint(-coeff_range, coeff_range + 1)
                        v_val = rng.randint(-coeff_range, coeff_range + 1)
                        w_val = rng.randint(-coeff_range, coeff_range + 1)
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            U[i * m + j, r] += u_val
                            V[j * p + k, r] += v_val
                            W[k * n + i, r] += w_val
                
                # Verify
                if verify_exact(U.tolist(), V.tolist(), W.tolist()):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
        
        return best_result
    
    def try_laderman_extension(target_rank):
        """
        Try to extend Laderman's 3x3x3 rank-23 decomposition to 3x3x5.
        """
        if time.time() - start_time > deadline:
            return None
        
        # Laderman's 3x3x3 decomposition (rank 23)
        # I'll use a known construction.
        # The tensor for 3x3x3 matrix mult has 27 entries, rank 23.
        
        # For 3x3x5, we have 3*3*5 = 45 entries.
        # We can view this as two blocks: columns 0-2 and columns 3-4 of B.
        # C[:, 0:3] = A @ B[:, 0:3]  (3x3x3 problem, rank 23)
        # C[:, 3:5] = A @ B[:, 3:5]  (3x3x2 problem)
        
        # For the 3x3x2 part, schoolbook needs 18 terms.
        # But we can potentially share some structure.
        
        # Total rank: 23 + 18 = 41 in the simplest extension.
        # But maybe we can do better.
        
        # Let's try to find a decomposition with rank < 41.
        
        # Approach: Use the 23 Laderman terms for the first 3 columns,
        # then add terms for the remaining 2 columns.
        
        # For the remaining 2 columns (k=3,4), we need:
        # sum_r U[i*m+j, r] * V[j*p+3, r] * W[3*n+i, r] = 1 for all i,j
        # sum_r U[i*m+j, r] * V[j*p+4, r] * W[4*n+i, r] = 1 for all i,j
        
        # If we reuse the same U values from Laderman, we need to find V and W
        # for the new columns such that the products sum to 1.
        
        # This is a system of 9 equations (for i,j) for each new column.
        
        # Let's try a simpler approach: just do random search for the full problem.
        
        if time.time() - start_time > deadline:
            return None
        
        best_result = None
        
        for seed in range(100):
            if time.time() - start_time > deadline:
                break
            
            rng = np.random.RandomState(seed + 10000)
            
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Structured initialization:
            # First 23 terms: try to mimic Laderman structure
            # Remaining terms: for the extra columns
            
            for r in range(target_rank):
                if r < 23:
                    # Laderman-like pattern
                    i = r % 3
                    j = (r // 3) % 3
                    k = (r // 9) % 3
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                else:
                    # Extra terms for columns 3,4
                    k = 3 + (r - 23) % 2
                    i = (r - 23) // 2 % 3
                    j = (r - 23) // 6 % 3
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
            
            # Add some random perturbation
            if seed > 0:
                for r in range(target_rank):
                    if rng.random() < 0.3:
                        i = rng.randint(0, n)
                        j = rng.randint(0, m)
                        k = rng.randint(0, p)
                        U[i * m + j, r] += rng.randint(-2, 3)
                        V[j * p + k, r] += rng.randint(-2, 3)
                        W[k * n + i, r] += rng.randint(-2, 3)
            
            if verify_exact(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
        
        return best_result
    
    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known: 3x3x3 rank 23, so 3x3x5 should be at most 23 + 18 = 41
    # Try ranks from 30 to 44
    
    for rank in range(30, 45):
        if time.time() - start_time > deadline:
            break
        
        # Try structured Laderman extension first
        result = try_laderman_extension(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try MILP-like random search
        if time.time() - start_time > deadline:
            break
        
        result = try_milp_decomposition(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # Final verification
    if best_rank < n * m * p:
        if not verify_numeric(best_result["U"], best_result["V"], best_result["W"]):
            best_result = schoolbook()
            best_rank = n * m * p
    
    return best_result
# EVOLVE-BLOCK-END
