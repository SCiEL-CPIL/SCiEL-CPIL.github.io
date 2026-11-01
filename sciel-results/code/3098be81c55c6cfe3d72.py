"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook decomposition
    U = [[Fraction(0)] * schoolbook_rank for _ in range(n * m)]
    V = [[Fraction(0)] * schoolbook_rank for _ in range(m * p)]
    W = [[Fraction(0)] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Helper: verify decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition exactly matches matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Helper: compute tensor residual for given U, V, W
    def compute_residual(U_mat, V_mat, W_mat, rank):
        """Compute the residual tensor: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1"""
        residual = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    residual[(i, j, k)] = total - 1
        return residual
    
    # Helper: check if all residuals are zero
    def is_valid(residual):
        for val in residual.values():
            if val != 0:
                return False
        return True
    
    # Strategy: Try to find low-rank decompositions using structured patterns
    # For each target rank, try different construction strategies
    
    # Strategy 1: Use the fact that we can "share" terms across different (i,j,k)
    # A rank-1 term (u, v, w) contributes to all (i,j,k) where u[i,j]*v[j,k]*w[k,i] != 0
    
    # Let's try a systematic search with small value sets
    value_set = [Fraction(0), Fraction(1), Fraction(-1), Fraction(2), Fraction(-2), 
                 Fraction(1,2), Fraction(-1,2), Fraction(3), Fraction(-3)]
    
    # Precompute all possible column patterns for small dimensions
    # U has n*m = 6 rows, V has m*p = 12 rows, W has p*n = 8 rows
    
    # For efficiency, let's use a different approach:
    # Build decompositions incrementally, ensuring validity at each step
    
    # Try target ranks from low to high
    for target_rank in range(18, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break
        
        # Method A: Random search with structured initialization
        max_attempts = 100 if target_rank <= 20 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random columns with bias toward small values
            U_try = []
            for _ in range(n * m):
                row = []
                for _ in range(target_rank):
                    # Use small values with higher probability
                    val = random.choice(value_set[:5])  # Prefer 0, ±1, ±2
                    row.append(val)
                U_try.append(row)
            
            V_try = []
            for _ in range(m * p):
                row = []
                for _ in range(target_rank):
                    val = random.choice(value_set[:5])
                    row.append(val)
                V_try.append(row)
            
            W_try = []
            for _ in range(p * n):
                row = []
                for _ in range(target_rank):
                    val = random.choice(value_set[:5])
                    row.append(val)
                W_try.append(row)
            
            residual = compute_residual(U_try, V_try, W_try, target_rank)
            if is_valid(residual):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Method B: Local search - try to fix residuals by adjusting columns
    if best_rank < schoolbook_rank and time.time() < deadline - 20:
        # Try to improve by local adjustments
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            if best_rank <= 18:
                break
            
            # Pick a random column to modify
            col_idx = random.randint(0, best_rank - 1)
            
            # Try small perturbations
            improved = False
            for _ in range(20):
                U_new = [row[:] for row in best_U]
                V_new = [row[:] for row in best_V]
                W_new = [row[:] for row in best_W]
                
                # Perturb one entry in each matrix for this column
                u_row = random.randint(0, n * m - 1)
                v_row = random.randint(0, m * p - 1)
                w_row = random.randint(0, p * n - 1)
                
                U_new[u_row][col_idx] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                V_new[v_row][col_idx] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                W_new[w_row][col_idx] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                
                # Ensure entries stay in reasonable range
                if abs(U_new[u_row][col_idx]) > 5 or abs(V_new[v_row][col_idx]) > 5 or abs(W_new[w_row][col_idx]) > 5:
                    continue
                
                residual = compute_residual(U_new, V_new, W_new, best_rank)
                if is_valid(residual):
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    improved = True
                    break
            
            if improved:
                # Try to reduce rank by checking if any column can be removed
                for c in range(best_rank):
                    if time.time() > deadline - 10:
                        break
                    # Check if column c is "redundant" - i.e., can be expressed as combination of others
                    # Simple check: try zeroing out column c and see if we can fix with small adjustments
                    U_test = [row[:] for row in best_U]
                    V_test = [row[:] for row in best_V]
                    W_test = [row[:] for row in best_W]
                    
                    # Zero out column c
                    for i in range(n * m):
                        U_test[i][c] = Fraction(0)
                    for j in range(m * p):
                        V_test[j][c] = Fraction(0)
                    for k in range(p * n):
                        W_test[k][c] = Fraction(0)
                    
                    residual = compute_residual(U_test, V_test, W_test, best_rank)
                    
                    # If residual is small, try to fix it
                    if not is_valid(residual):
                        # Count non-zero residuals
                        non_zero = sum(1 for v in residual.values() if v != 0)
                        if non_zero <= 3:
                            # Try to fix by adjusting other columns
                            fixed = False
                            for fix_attempt in range(10):
                                U_fix = [row[:] for row in U_test]
                                V_fix = [row[:] for row in V_test]
                                W_fix = [row[:] for row in W_test]
                                
                                # Try to distribute the residual across other columns
                                for r in range(best_rank):
                                    if r == c:
                                        continue
                                    # Small adjustment
                                    if random.random() < 0.3:
                                        u_row = random.randint(0, n * m - 1)
                                        U_fix[u_row][r] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                                    if random.random() < 0.3:
                                        v_row = random.randint(0, m * p - 1)
                                        V_fix[v_row][r] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                                    if random.random() < 0.3:
                                        w_row = random.randint(0, p * n - 1)
                                        W_fix[w_row][r] += random.choice([Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)])
                                
                                residual_fix = compute_residual(U_fix, V_fix, W_fix, best_rank)
                                if is_valid(residual_fix):
                                    # Compress: remove column c
                                    active_cols = [r for r in range(best_rank) if r != c]
                                    U_comp = [[U_fix[i][r] for r in active_cols] for i in range(n * m)]
                                    V_comp = [[V_fix[j][r] for r in active_cols] for j in range(m * p)]
                                    W_comp = [[W_fix[k][r] for r in active_cols] for k in range(p * n)]
                                    
                                    if verify(U_comp, V_comp, W_comp, best_rank - 1):
                                        best_U = U_comp
                                        best_V = V_comp
                                        best_W = W_comp
                                        best_rank -= 1
                                        fixed = True
                                        break
                            
                            if fixed:
                                break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_U = [[Fraction(0)] * schoolbook_rank for _ in range(n * m)]
        best_V = [[Fraction(0)] * schoolbook_rank for _ in range(m * p)]
        best_W = [[Fraction(0)] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = Fraction(1)
                    best_V[j * p + k][col] = Fraction(1)
                    best_W[k * n + i][col] = Fraction(1)
                    col += 1
        best_rank = schoolbook_rank
    
    # Convert to list of lists of floats
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
