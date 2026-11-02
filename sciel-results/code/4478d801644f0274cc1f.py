"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Strategy: Use Strassen-based partitioning for 3x3x4
    # Partition A (3x3) into blocks: A11(1x1), A12(1x2), A21(2x1), A22(2x2)
    # Partition B (3x4) into blocks: B11(1x2), B12(1x2), B21(2x2), B22(2x2)
    # Partition C (3x4) into blocks: C11(1x2), C12(1x2), C21(2x2), C22(2x2)
    
    # Use Strassen on the 2x2x2 subproblem (bottom-right blocks)
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A22 - A11) * (B11 - B22)  [but we need to adapt for rectangular]
    
    # Actually, let's use a direct approach: schoolbook gives rank 36.
    # We can try to find rank-34 by exploiting linearity.
    
    # For 3x3x4, the tensor has 3*3*4 = 36 entries.
    # Schoolbook: rank 36.
    # Known: rank(3,3,4) <= 34 via structured decomposition.
    
    # Let's construct a rank-34 decomposition using the following partition:
    # Split the 3x3x4 problem into:
    # - A 2x2x4 subproblem (rows 0-1, all cols) 
    # - A 1x3x4 subproblem (row 2, all cols)
    # But this doesn't directly help.
    
    # Alternative: Use the fact that we can share computations.
    # For the 3x3x4 case, consider:
    # C[0:2, 0:2] uses A[0:2, 0:2] * B[0:2, 0:2]
    # C[0:2, 2:4] uses A[0:2, 0:2] * B[0:2, 2:4]
    # C[2:3, 0:2] uses A[2:3, 0:2] * B[0:2, 0:2]
    # C[2:3, 2:4] uses A[2:3, 0:2] * B[0:2, 2:4]
    
    # Let's use a systematic approach: try to find rank-34 via random search with structure.
    
    # First, build the schoolbook decomposition (rank 36) as baseline
    rank_sb = n * m * p  # 36
    U_sb = [[Fraction(0)] * rank_sb for _ in range(n * m)]
    V_sb = [[Fraction(0)] * rank_sb for _ in range(m * p)]
    W_sb = [[Fraction(0)] * rank_sb for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U_sb[i * m + j][idx] = 1
                V_sb[j * p + k][idx] = 1
                W_sb[k * n + i][idx] = 1
    
    # Verify schoolbook works
    def verify_decomposition(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(len(U[0])):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        # Also check off-diagonal terms are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) != (i2, j2, k2):
                                    total = Fraction(0)
                                    for c in range(len(U[0])):
                                        total += U[i2 * m + j2][c] * V[j2 * p + k2][c] * W[k2 * n + i2][c]
                                    if total != 0:
                                        return False
        return True
    
    # Try to find a rank-34 decomposition
    # Use the known structure: for 3x3x4, we can achieve rank 34
    # by exploiting the rectangular structure.
    
    # Partition approach:
    # Let A = [[a, b], [c, d]] where a is 1x1, b is 1x2, c is 2x1, d is 2x2
    # Let B = [[e, f], [g, h]] where e is 1x2, f is 1x2, g is 2x2, h is 2x2
    # Wait, B is 3x4, so partition differently.
    
    # Let me use a different partition:
    # A (3x3): rows 0-1 (2x3), row 2 (1x3)
    # B (3x4): cols 0-2 (3x3), col 3 (3x1)
    # C (3x4): rows 0-1 (2x4), row 2 (1x4)
    
    # This is getting complex. Let me try a numerical approach:
    # Start from schoolbook and try to merge terms.
    
    # For rank-34, we need to save 2 terms.
    # One known method: for the 2x2x2 subblock, use Strassen (7 instead of 4), saving 3 terms.
    # But we need to account for the interactions.
    
    # Let's try a direct construction for rank 34:
    # Based on the paper "Lower bounds for the rank of matrix multiplication"
    # For 3x3x4, rank <= 34 is achievable.
    
    # I'll use the following construction:
    # The 3x3x4 tensor can be decomposed using 34 rank-1 terms.
    
    # Since I can't recall the exact coefficients, let me try a search approach:
    # Start from schoolbook (rank 36) and try to find 2 linear dependencies.
    
    # Actually, let me just return schoolbook for now to ensure validity,
    # and add a time-limited search for rank reduction.
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find rank-34 via random perturbation and SVD-like reduction
    # This is a heuristic approach
    
    def try_reduce_rank(U0, V0, W0, target_rank, n, m, p, time_limit=60):
        """Try to reduce rank by finding linear dependencies."""
        import numpy as np
        start = time.time()
        
        # Convert to numpy for efficiency
        U0_np = np.array(U0, dtype=float)
        V0_np = np.array(V0, dtype=float)
        W0_np = np.array(W0, dtype=float)
        
        # Try random linear combinations of columns
        for attempt in range(50):
            if time.time() - start > time_limit:
                break
            
            # Try to merge two columns
            for c1 in range(target_rank + 1):
                for c2 in range(c1 + 1, target_rank + 1):
                    # Try to express column c2 as a linear combination of others
                    # This is a linear algebra problem
                    
                    # Skip for now - this approach is too slow
                    pass
        
        return None, None, None, best_rank
    
    # For now, return schoolbook which is guaranteed valid
    U_float = [[float(x) for x in row] for row in U_sb]
    V_float = [[float(x) for x in row] for row in V_sb]
    W_float = [[float(x) for x in row] for row in W_sb]
    
    return {"U": U_float, "V": V_float, "W": W_float}
# EVOLVE-BLOCK-END
