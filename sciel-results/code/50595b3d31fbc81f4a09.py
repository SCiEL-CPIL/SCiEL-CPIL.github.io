"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DEADLINE = 140.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _tensor_residual(params, n, m, p, rank):
    """Compute the tensor reconstruction error given flattened parameters."""
    # Parameters: U (n*m*rank), V (m*p*rank), W (p*n*rank)
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n*m, rank)
    V = params[U_size:U_size+V_size].reshape(m*p, rank)
    W = params[U_size+V_size:U_size+V_size+W_size].reshape(p*n, rank)
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                residuals.append(val - 1.0)
    return np.array(residuals)

def _try_numeric_optimization(n, m, p, target_rank, time_budget):
    """Try to find a decomposition of given rank using numerical optimization."""
    start = time.time()
    U_size = n * m * target_rank
    V_size = m * p * target_rank
    W_size = p * n * target_rank
    total_params = U_size + V_size + W_size
    
    # Initialize with random values
    np.random.seed(42)
    x0 = np.random.randn(total_params) * 0.5
    
    # Try multiple restarts
    best_solution = None
    best_residual = np.inf
    
    for restart in range(20):
        if time.time() - start > time_budget:
            break
        
        if restart == 0:
            x_init = x0
        else:
            x_init = np.random.randn(total_params) * 0.5
        
        try:
            result = least_squares(
                _tensor_residual,
                x_init,
                args=(n, m, p, target_rank),
                method='lm',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10
            )
            
            if result.cost < best_residual:
                best_residual = result.cost
                best_solution = result.x
                
            if best_residual < 1e-12:
                break
                
        except Exception:
            continue
    
    if best_solution is None or best_residual > 1e-8:
        return None
    
    # Extract and round to rationals
    U = best_solution[:U_size].reshape(n*m, target_rank)
    V = best_solution[U_size:U_size+V_size].reshape(m*p, target_rank)
    W = best_solution[U_size+V_size:U_size+V_size+W_size].reshape(p*n, target_rank)
    
    # Try to snap to simple rationals
    def snap_to_rational(x, max_denom=8):
        if abs(x) < 1e-10:
            return 0
        best_frac = Fraction(x).limit_denominator(max_denom)
        if abs(float(best_frac) - x) < 1e-8:
            return best_frac
        # Try half-integers
        half = Fraction(int(round(x * 2)), 2)
        if abs(float(half) - x) < 1e-8:
            return half
        # Try integers
        int_val = int(round(x))
        if abs(float(int_val) - x) < 1e-8:
            return int_val
        return None
    
    U_int = [[0]*target_rank for _ in range(n*m)]
    V_int = [[0]*target_rank for _ in range(m*p)]
    W_int = [[0]*target_rank for _ in range(p*n)]
    
    all_snapped = True
    for i in range(n*m):
        for c in range(target_rank):
            s = snap_to_rational(U[i][c])
            if s is None:
                all_snapped = False
                break
            U_int[i][c] = s
        if not all_snapped:
            break
    
    if not all_snapped:
        for j in range(m*p):
            for c in range(target_rank):
                s = snap_to_rational(V[j][c])
                if s is None:
                    all_snapped = False
                    break
                V_int[j][c] = s
            if not all_snapped:
                break
    
    if not all_snapped:
        for k in range(p*n):
            for c in range(target_rank):
                s = snap_to_rational(W[k][c])
                if s is None:
                    all_snapped = False
                    break
                W_int[k][c] = s
            if not all_snapped:
                break
    
    if not all_snapped:
        return None
    
    # Verify
    if _verify(U_int, V_int, W_int, n, m, p):
        return U_int, V_int, W_int
    
    return None

def _try_structured_search(n, m, p, target_rank, time_budget):
    """Try structured parameterization exploiting n=2 symmetry."""
    start = time.time()
    
    # For n=2, structure: U has two blocks of size m x rank
    # U_block0[i,j] for output row 0, U_block1[i,j] for output row 1
    # We can try to find rank-r decompositions where the factors have structure
    
    # Approach: fix W to be identity-like and solve for U, V
    # Or: use the fact that for each (i,k), we need sum_j A[i,j]*B[j,k]
    
    # Try: parameterize with rank r, where we set up the system
    # For each column c, we have vectors u_c in R^{m}, v_c in R^{p*2}, w_c in R^{p*2}
    # Actually let's think differently.
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # For n=2: W has 2*p = 10 rows.
    # Let's try a different parameterization:
    # Fix the W matrix to be a specific structured form
    # and solve for U and V.
    
    # Simple structured approach: 
    # Let W[k*n+i, c] = delta_{k, c%p} * something
    # This is getting complex. Let's just try numerical optimization with better init.
    
    # Better initialization: start from schoolbook and randomly perturb
    if target_rank >= n * m * p:
        return _schoolbook(n, m, p)
    
    # Try numerical optimization with structured initialization
    np.random.seed(123)
    
    U_size = n * m * target_rank
    V_size = m * p * target_rank  
    W_size = p * n * target_rank
    total_params = U_size + V_size + W_size
    
    # Initialize: use schoolbook-like structure but compressed
    # For each (i,j,k) triple, we have one term in schoolbook
    # We need to map n*m*p terms into target_rank terms
    
    # Strategy: randomly assign each (i,j,k) to one of target_rank columns
    assignment = np.random.randint(0, target_rank, size=(n, m, p))
    
    x0 = np.zeros(total_params)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = assignment[i, j, k]
                U_idx = (i*m + j) * target_rank + c
                V_idx = U_size + (j*p + k) * target_rank + c
                W_idx = U_size + V_size + (k*n + i) * target_rank + c
                x0[U_idx] = 1.0
                x0[V_idx] = 1.0
                x0[W_idx] = 1.0
    
    best_solution = None
    best_residual = np.inf
    
    for restart in range(15):
        if time.time() - start > time_budget:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the initial solution
            x_init = x0.copy()
            x_init += np.random.randn(total_params) * 0.3
        
        try:
            result = least_squares(
                _tensor_residual,
                x_init,
                args=(n, m, p, target_rank),
                method='lm',
                max_nfev=3000,
                xtol=1e-12,
                ftol=1e-12
            )
            
            if result.cost < best_residual:
                best_residual = result.cost
                best_solution = result.x
            
            if best_residual < 1e-14:
                break
                
        except Exception:
            continue
    
    if best_solution is None or best_residual > 1e-10:
        return None
    
    U = best_solution[:U_size].reshape(n*m, target_rank)
    V = best_solution[U_size:U_size+V_size].reshape(m*p, target_rank)
    W = best_solution[U_size+V_size:U_size+V_size+W_size].reshape(p*n, target_rank)
    
    # Snap to rationals
    def snap(x, max_denom=16):
        if abs(x) < 1e-10:
            return 0
        for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
            num = round(x * d)
            if abs(num / d - x) < 1e-8:
                return Fraction(num, d)
        return None
    
    U_int = [[0]*target_rank for _ in range(n*m)]
    V_int = [[0]*target_rank for _ in range(m*p)]
    W_int = [[0]*target_rank for _ in range(p*n)]
    
    all_ok = True
    for mat, int_mat in [(U, U_int), (V, V_int), (W, W_int)]:
        for i in range(len(mat)):
            for c in range(target_rank):
                s = snap(mat[i][c])
                if s is None:
                    all_ok = False
                    break
                int_mat[i][c] = s
            if not all_ok:
                break
        if not all_ok:
            break
    
    if not all_ok:
        return None
    
    if _verify(U_int, V_int, W_int, n, m, p):
        return U_int, V_int, W_int
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try ranks from 50 down, but be strategic
    # Known: rank(M_{n,m,p}) >= max(n*m, m*p, n*p) for some cases
    # For M_{2,5,5}: lower bound is at least 10 (from n*m=10 or m*p=25... actually m*p=25)
    # Wait, the lower bound is max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But actually the Strassen-style bounds give better estimates
    # Let's try from 50 down
    
    # Time budget allocation: spend more time on higher ranks (more likely to succeed)
    ranks_to_try = list(range(schoolbook_rank - 1, 9, -1))
    
    for target_rank in ranks_to_try:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5  # 5s safety margin
        
        if remaining < 2:
            break
        
        # Allocate time based on rank: higher ranks get less time (easier), lower get more
        if target_rank >= 40:
            time_budget = min(remaining, 3)
        elif target_rank >= 30:
            time_budget = min(remaining, 5)
        else:
            time_budget = min(remaining, 10)
        
        result = _try_structured_search(n, m, p, target_rank, time_budget)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # Don't break - keep trying for lower rank
        
        if time.time() > DEADLINE - 10:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
