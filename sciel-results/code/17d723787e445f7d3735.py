import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS
R_MAX = 60 # Schoolbook rank as upper bound

def _validate(U, V, W):
    """Check if U, V, W form a valid rank-R tensor decomposition."""
    if not U or not V or not W:
        return False
    
    rank = len(V[0]) if V else 0
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        return den in [1, 2]
    elif isinstance(x, int):
        return True
    elif isinstance(x, float):
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    return False

def _build_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return U, V, W

def _build_block_55():
    """
    Construct the Rank 55 decomposition using 5 blocks of (2x3x2).
    Each block uses Rank 11 (7 for 2x2x2 + 4 for 2x1x2).
    """
    n, m, p = DIMENSIONS
    R = 55
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for g in range(5):
        base_rank = 11 * g
        
        # --- Part 1: A1 * B1 (2x2 * 2x2) -> Rank 7 (Strassen) ---
        c = base_rank
        U[0][c] += 1   # A11
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] += 1   # C11
        W[4*g+3][c] += 1 # C22
        
        c = base_rank + 1
        U[3][c] += 1   # A21
        U[4][c] += 1   # A22
        V[2*g][c] += 1   # B11
        W[4*g+2][c] += 1 # C21
        
        c = base_rank + 2
        U[0][c] += 1   # A11
        V[10+2*g][c] += 1   # B21
        V[10+2*g+1][c] -= 1 # B22
        W[4*g+1][c] += 1 # C12
        W[4*g+3][c] += 1 # C22
        
        c = base_rank + 3
        U[4][c] += 1   # A22
        V[2*g+1][c] += 1   # B12
        V[2*g][c] -= 1   # B11
        W[4*g][c] += 1   # C11
        W[4*g+2][c] += 1 # C21
        
        c = base_rank + 4
        U[0][c] += 1   # A11
        U[1][c] += 1   # A12
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] -= 1   # C11
        W[4*g+1][c] += 1 # C12
        
        c = base_rank + 5
        U[3][c] += 1   # A21
        U[0][c] -= 1   # A11
        V[2*g][c] += 1   # B11
        V[2*g+1][c] += 1 # B12
        W[4*g+3][c] += 1 # C22
        
        c = base_rank + 6
        U[1][c] += 1   # A12
        U[4][c] -= 1   # A22
        V[10+2*g][c] += 1 # B21
        V[10+2*g+1][c] += 1 # B22
        W[4*g][c] += 1   # C11
        
        # --- Part 2: A2 * B2 (2x1 * 1x2) -> Rank 4 ---
        c = base_rank + 7
        U[2][c] += 1
        V[20+2*g][c] += 1
        W[4*g][c] += 1
        
        c = base_rank + 8
        U[2][c] += 1
        V[20+2*g+1][c] += 1
        W[4*g+1][c] += 1
        
        c = base_rank + 9
        U[5][c] += 1
        V[20+2*g][c] += 1
        W[4*g+2][c] += 1
        
        c = base_rank + 10
        U[5][c] += 1
        V[20+2*g+1][c] += 1
        W[4*g+3][c] += 1

    return U, V, W

def _try_numeric_optimization(start_rank, deadline):
    """
    Attempt to find a decomposition of a given rank using numerical optimization.
    We minimize the squared error of the tensor reconstruction.
    """
    # This is a heuristic search. We parameterize U, V, W directly.
    # To make it efficient, we fix the scale of one matrix (e.g., V) to avoid scaling invariance issues,
    # or we simply minimize the error and hope it converges to 0.
    
    # We will try to optimize for a specific rank.
    # We use a simple random restart strategy with Nelder-Mead if available, 
    # or a custom gradient-free step if not.
    
    # Since scipy is available, we can use minimize.
    from scipy.optimize import minimize
    
    # Flatten parameters
    size_U = n * m * start_rank
    size_V = m * p * start_rank
    size_W = p * n * start_rank
    total_size = size_U + size_V + size_W
    
    def objective(params):
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U_mat = U_flat.reshape(n*m, start_rank)
        V_mat = V_flat.reshape(m*p, start_rank)
        W_mat = W_flat.reshape(p*n, start_rank)
        
        # Compute tensor T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # Target is 1 if (i,j,k) corresponds to a valid multiplication term?
        # No, the target tensor for matrix multiplication is:
        # T_{(i,j), (j,k), (k,i)} = 1 if indices match, else 0.
        # Specifically, the tensor T has non-zero entries only at (i, j, k) where the 
        # "flow" of indices matches.
        # Wait, the standard tensor for matmul is T_{a, b, c} = delta_{a, b} delta_{b, c} ... no.
        # The tensor T such that C = A B is T_{i, j, k} = 1 if we are computing C_{i,k} from A_{i,j} B_{j,k}.
        # The indices are (i, j, k) -> C_{i,k}.
        # The tensor is defined on I x J x K where I=rows of A, J=cols of A/rows of B, K=cols of B.
        # T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # In our decomposition:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Decomposition: C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_k V_{j,k,c} B_{j,k}) ... No.
        # The decomposition is: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
        # Here U index is (i,j), V index is (j,k), W index is (k,i).
        # The target T_{i,j,k} is 1 if the indices are consistent for the matmul.
        # Actually, for the specific tensor of matmul, T_{i,j,k} is 1 for all i,j,k?
        # No. The tensor is sparse.
        # T_{i, j, k} = 1 if we are summing over j.
        # Wait, the tensor T is defined such that <T, A \otimes B> = C.
        # T_{i, j, k} = 1 if the indices allow the multiplication.
        # In the standard basis, T_{i, j, k} = 1 for all i, j, k?
        # No, T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # So T_{i, j, k} = 1 for all valid i, j, k.
        # There are n*m*p such terms.
        # So the target tensor is all ones for the valid indices?
        # Yes, T_{i, j, k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Let's verify:
        # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} 1 * A_{i,j} B_{j,k} = sum_i (sum_j A_{i,j} B_{j,k}) ...
        # Wait, the sum over k is missing in the LHS if we just sum.
        # The contraction is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}.
        # If T_{i,j,k} = 1, this is sum_{i,j,k} A_{i,j} B_{j,k}.
        # This sums over all i, j, k.
        # But C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor product A \otimes B has components A_{i,j} B_{j,k}.
        # The inner product <T, A \otimes B> should yield the vector C.
        # Usually, the tensor is defined such that T_{(i,k), j} ... no.
        
        # Let's look at the provided _validate function.
        # It checks: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # for all i, j, k.
        # This implies the target tensor T_{i,j,k} is 1 for ALL i, j, k.
        # This is correct for the "multiplication tensor" in this specific indexing convention.
        
        # So we want to minimize sum_{i,j,k} (sum_c U_{ij,c} V_{jk,c} W_{ki,c} - 1)^2
        
        error = 0.0
        # Efficient computation using einsum?
        # T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        # error = np.sum((T - 1.0)**2)
        
        # To speed up, we can compute the error incrementally or use vectorized ops.
        # For small dimensions, einsum is fine.
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        error = np.sum((T - 1.0)**2)
        return error

    # Initial guess: random small values
    # We try multiple random restarts
    best_result = None
    best_error = float('inf')
    
    # We have limited time. Let's do a few quick restarts.
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10: break
        
        # Initialize with random values
        params = np.random.randn(total_size) * 0.1
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500, 'maxfev': 500})
            if res.success or res.fun < best_error:
                best_error = res.fun
                if res.fun < 1e-6:
                    best_result = res.x
                    break
        except Exception:
            pass
            
    if best_result is None:
        return None
        
    # Extract U, V, W
    U_flat = best_result[:size_U]
    V_flat = best_result[size_U:size_U+size_V]
    W_flat = best_result[size_U+size_V:]
    
    U_mat = U_flat.reshape(n*m, start_rank)
    V_mat = V_flat.reshape(m*p, start_rank)
    W_mat = W_flat.reshape(p*n, start_rank)
    
    # Snap to half-integers
    # The problem requires integer or half-integer entries.
    # We round to the nearest 0.5.
    U_snap = np.round(U_mat * 2) / 2
    V_snap = np.round(V_mat * 2) / 2
    W_snap = np.round(W_mat * 2) / 2
    
    # Convert to list of Fractions for exact validation
    U_list = [[Fraction(int(x*2), 2) for x in row] for row in U_snap]
    V_list = [[Fraction(int(x*2), 2) for x in row] for row in V_snap]
    W_list = [[Fraction(int(x*2), 2) for x in row] for row in W_snap]
    
    if _validate(U_list, V_list, W_list):
        return U_list, V_list, W_list
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Return known best (55)
    U, V, W = _build_block_55()
    if _validate(U, V, W):
        # Check if we have time to try for 54
        if time.time() < deadline - 60:
            # Try to find rank 54
            res = _try_numeric_optimization(54, deadline)
            if res:
                U, V, W = res
        
        return {"U": U, "V": W, "W": W} # Wait, return dict keys U, V, W
        
    # Fallback
    U_fb, V_fb, W_fb = _build_schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
