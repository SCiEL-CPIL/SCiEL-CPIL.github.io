import time
import random
import numpy as np
from fractions import Fraction
import scipy.linalg

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 24
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def is_valid_solution(U, V, W, r):
        """Check if U, V, W exactly decompose the matrix multiplication tensor.
        
        The tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k,
        and T[a,b,c] = 0 otherwise.
        """
        # Check valid entries equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for c_idx in range(r):
                        s = s + U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                    if s != 1:
                        return False
        # Check invalid entries equal 0
        # For each pair (a, b), determine which c values should be nonzero
        for a in range(n * m):
            i_a, j_a = divmod(a, m)
            for b in range(m * p):
                j_b, k_b = divmod(b, p)
                if j_a != j_b:
                    # All c should be 0
                    for c in range(p * n):
                        s = 0
                        for c_idx in range(r):
                            s = s + U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                        if s != 0:
                            return False
                else:
                    # Only c = k_b * n + i_a should be 1, others 0
                    valid_c = k_b * n + i_a
                    for c in range(p * n):
                        s = 0
                        for c_idx in range(r):
                            s = s + U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                        if c == valid_c:
                            if s != 1:
                                return False
                        else:
                            if s != 0:
                                return False
        return True

    def solve_for_W(U, V, r):
        """Solve for W given U, V using linear algebra.
        
        Variables: W[c, c_idx] for c in range(p*n), c_idx in range(r)
        Equations: For each valid (i,j,k): sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] = 1
                  For each invalid (a,b,c): sum_c U[a][c]*V[b][c]*W[c][c] = 0
        """
        num_vars = p * n * r
        # Build equations
        A_rows = []
        b_vals = []
        
        # Valid equations: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    row = [0.0] * num_vars
                    for c_idx in range(r):
                        var_idx = c * r + c_idx
                        row[var_idx] = float(U[a][c_idx] * V[b][c_idx])
                    A_rows.append(row)
                    b_vals.append(1.0)
        
        # Invalid equations: T[a,b,c] = 0 for invalid combinations
        for a in range(n * m):
            i_a, j_a = divmod(a, m)
            for b in range(m * p):
                j_b, k_b = divmod(b, p)
                if j_a != j_b:
                    for c in range(p * n):
                        row = [0.0] * num_vars
                        for c_idx in range(r):
                            var_idx = c * r + c_idx
                            row[var_idx] = float(U[a][c_idx] * V[b][c_idx])
                        A_rows.append(row)
                        b_vals.append(0.0)
                else:
                    valid_c = k_b * n + i_a
                    for c in range(p * n):
                        if c == valid_c:
                            continue
                        row = [0.0] * num_vars
                        for c_idx in range(r):
                            var_idx = c * r + c_idx
                            row[var_idx] = float(U[a][c_idx] * V[b][c_idx])
                        A_rows.append(row)
                        b_vals.append(0.0)
        
        if not A_rows:
            return None
            
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        try:
            x, residuals, rank, sv = scipy.linalg.lstsq(A, b, cond=1e-12)
            if residuals is None or residuals[0] > 1e-8:
                return None
            return x
        except Exception:
            return None

    def try_rank(r, max_trials=500):
        """Try to find a rank-r decomposition."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Generate sparse random U, V with entries in {-1, 0, 1}
            # Bias towards sparsity
            sparsity = 0.3
            U = [[random.choice([-1, 0, 1]) if random.random() < sparsity else 0 for _ in range(r)] 
                 for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) if random.random() < sparsity else 0 for _ in range(r)] 
                 for _ in range(m * p)]
            
            x = solve_for_W(U, V, r)
            if x is None:
                continue
            
            # Reshape x to W
            W = [[0.0] * r for _ in range(p * n)]
            for c in range(p * n):
                for c_idx in range(r):
                    W[c][c_idx] = x[c * r + c_idx]
            
            # Snap to nearest half-integer
            W_snapped = [[round(val * 2) / 2 for val in row] for row in W]
            
            # Convert to Fractions for exact verification
            U_frac = [[Fraction(val) for val in row] for row in U]
            V_frac = [[Fraction(val) for val in row] for row in V]
            W_frac = [[Fraction(val).limit_denominator(100) for val in row] for row in W_snapped]
            
            if is_valid_solution(U_frac, V_frac, W_frac, r):
                return (U_frac, V_frac, W_frac, r)
        
        return None

    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But actual lower bound for 2x3x4 is at least 12
    for r in range(12, rank_school):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_trials=300)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            # Found a valid decomposition; try to find lower rank
            # But since we're going low to high, first found is minimal
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
