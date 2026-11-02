import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def is_nice(w_vec, tol=1e-8):
        """Check if w_vec entries are close to integers or half-integers."""
        w2 = [2 * x for x in w_vec]
        for x in w2:
            if abs(x - round(x)) > tol:
                return False
        return True
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0
    
    def solve_for_w(U, V, r):
        """
        Given U (nm x r) and V (mp x r), solve for W (pn x r).
        For each (i,k) pair, solve A * w = 1 where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
        Returns W or None if no nice solution found.
        """
        W = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                # Build system: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j][c] * V[j * p + k][c]
                
                # Solve A @ w_vec = ones(m)
                try:
                    # Try least squares first
                    w_sol, residuals, rank_val, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check if solution is nice (integer or half-integer)
                    if is_nice(w_sol, tol=1e-7):
                        w_rounded = [round_to_half(x) for x in w_sol]
                        # Verify the rounded solution works
                        A_np = np.array(A, dtype=float)
                        w_np = np.array(w_rounded, dtype=float)
                        if np.allclose(A_np @ w_np, np.ones(m), atol=1e-6):
                            for c in range(r):
                                W[k * n + i][c] = w_rounded[c]
                        else:
                            # Try exact rational solve
                            from fractions import Fraction
                            A_frac = [[Fraction(U[i*m+j][c]).limit_denominator(1000) * Fraction(V[j*p+k][c]).limit_denominator(1000) for c in range(r)] for j in range(m)]
                            b_frac = [Fraction(1) for _ in range(m)]
                            
                            # Gaussian elimination with fractions
                            try:
                                aug = [row[:] + [b_frac[j]] for j, row in enumerate(A_frac)]
                                n_eq = m
                                n_unk = r
                                
                                # Forward elimination
                                for col in range(min(n_eq, n_unk)):
                                    # Find pivot
                                    pivot_row = None
                                    for row in range(col, n_eq):
                                        if aug[row][col] != 0:
                                            pivot_row = row
                                            break
                                    if pivot_row is None:
                                        continue
                                    # Swap
                                    aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                                    # Eliminate
                                    pivot_val = aug[col][col]
                                    for row in range(n_eq):
                                        if row != col and aug[row][col] != 0:
                                            factor = aug[row][col] / pivot_val
                                            for c2 in range(col, n_unk + 1):
                                                aug[row][c2] -= factor * aug[col][c2]
                                
                                # Back substitution / extract solution
                                w_frac = [Fraction(0) for _ in range(n_unk)]
                                for col in range(min(n_eq, n_unk)):
                                    if aug[col][col] != 0:
                                        w_frac[col] = aug[col][n_unk] / aug[col][col]
                                    else:
                                        # Check if consistent
                                        if aug[col][n_unk] != 0:
                                            return None
                                
                                # Convert to float and check if nice
                                w_float = [float(x) for x in w_frac]
                                if is_nice(w_float, tol=1e-7):
                                    w_rounded = [round_to_half(x) for x in w_float]
                                    for c in range(r):
                                        W[k * n + i][c] = w_rounded[c]
                                else:
                                    return None
                            except Exception:
                                return None
                    else:
                        # Try with slightly different tolerance
                        if is_nice(w_sol, tol=1e-5):
                            w_rounded = [round_to_half(x) for x in w_sol]
                            A_np = np.array(A, dtype=float)
                            w_np = np.array(w_rounded, dtype=float)
                            if np.allclose(A_np @ w_np, np.ones(m), atol=1e-4):
                                for c in range(r):
                                    W[k * n + i][c] = w_rounded[c]
                            else:
                                return None
                        else:
                            return None
                except Exception:
                    return None
        
        return W
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True
    
    def generate_random_UV(r, density=0.25, entry_range=[-1, 1]):
        """Generate random sparse U and V matrices."""
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        
        for i in range(n * m):
            for c in range(r):
                if random.random() < density:
                    U[i][c] = random.choice(entry_range)
        
        for i in range(m * p):
            for c in range(r):
                if random.random() < density:
                    V[i][c] = random.choice(entry_range)
        
        return U, V
    
    def generate_block_UV(r, block_size):
        """Generate block-structured U and V."""
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        
        # Divide columns into blocks
        n_blocks = r // block_size + (1 if r % block_size > 0 else 0)
        
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, r)
            block_width = end - start
            
            # Randomly assign rows to this block
            for i in range(n * m):
                if random.random() < 0.5:
                    for c in range(start, end):
                        if random.random() < 0.7:
                            U[i][c] = random.choice([-1, 1])
            
            for i in range(m * p):
                if random.random() < 0.5:
                    for c in range(start, end):
                        if random.random() < 0.7:
                            V[i][c] = random.choice([-1, 1])
        
        return U, V
    
    # Search strategy: try ranks from lower to higher, with multiple approaches
    # Known lower bounds for 3x4x5 matrix multiplication:
    # Schoolbook: 60
    # We'll search from around 45 down to 35
    
    search_ranks = list(range(45, 34, -1))  # 45, 44, ..., 35
    
    for r in search_ranks:
        if time.time() > deadline:
            break
        
        # Multiple search approaches for each rank
        approaches = [
            ('sparse', 0.2, [-1, 1]),
            ('sparse', 0.3, [-1, 1]),
            ('sparse', 0.25, [-2, -1, 1, 2]),
            ('block', 4, None),
            ('block', 5, None),
            ('block', 6, None),
        ]
        
        for approach in approaches:
            if time.time() > deadline:
                break
            
            if approach[0] == 'sparse':
                density = approach[1]
                entries = approach[2]
                n_trials = 200
            else:
                density = None
                entries = None
                n_trials = 100
            
            for trial in range(n_trials):
                if time.time() > deadline:
                    break
                
                if approach[0] == 'sparse':
                    U, V = generate_random_UV(r, density, entries)
                else:
                    U, V = generate_block_UV(r, approach[1])
                
                W = solve_for_w(U, V, r)
                
                if W is not None:
                    if verify_decomposition(U, V, W, r):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = r
                        # Found a solution at this rank, try to find lower
                        break
            
            if best_rank == r:
                break
    
    # If we haven't found anything better, try a few more aggressive searches
    if best_rank == rank_sb:
        for r in [42, 40, 38]:
            if time.time() > deadline:
                break
            
            # More trials with varied parameters
            for density in [0.15, 0.2, 0.25, 0.3, 0.35]:
                if time.time() > deadline:
                    break
                
                for trial in range(300):
                    if time.time() > deadline:
                        break
                    
                    U, V = generate_random_UV(r, density, [-1, 1])
                    W = solve_for_w(U, V, r)
                    
                    if W is not None:
                        if verify_decomposition(U, V, W, r):
                            best_result = {"U": U, "V": V, "W": W}
                            best_rank = r
                            break
    
    return best_result
# EVOLVE-BLOCK-END
