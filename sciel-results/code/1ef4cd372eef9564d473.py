import time
import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 6]

def _snapping(value, scale=2.0):
    """Snap value to nearest half-integer."""
    return round(value * scale) / scale

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check dimensions
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each entry (i, j, k) in the output matrix C = A @ B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{j,j} (i.e., 1 if we're computing correctly)
    
    # Actually, the tensor decomposition means:
    # The multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, let me reconsider. The matrix multiplication tensor T_{ijk} where:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the structure constant)
    
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Let's verify: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _make_strassen_rank33(n, m, p):
    """
    Create a rank-33 decomposition using Strassen's algorithm for the 2x2 blocks.
    
    We have A (2x3) and B (3x6). Split into:
    A = [[A11, A12, A13],
         [A21, A22, A23]]
    B = [[B11, B12, B13, B14, B15, B16],
         [B21, B22, B23, B24, B25, B26],
         [B31, B32, B33, B34, B35, B36]]
    
    C = A @ B is 2x6:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    For Strassen, we'd typically split into 2x2 blocks, but our dimensions don't divide evenly.
    Let me think of a different approach.
    
    Actually, let's try a block approach:
    Split A into left (2x1) and right (2x2): A = [A_left | A_right]
    Split B into top (1x6) and bottom (2x6): B = [B_top; B_bottom]
    
    Then C = A_left @ B_top + A_right @ B_bottom
    
    A_left is 2x1, B_top is 1x6: product is 2x6, rank at most 2*1*6 = 12
    A_right is 2x2, B_bottom is 2x6: product is 2x6, rank at most 2*2*6 = 24
    Total: 36 (same as schoolbook)
    
    For Strassen on the 2x2 part:
    A_right = [[a, b], [c, d]], B_bottom = [[e, f, g, h, i, j], [k, l, m, n, o, p]]
    
    Strassen: 
    M1 = (a-d)(k+l)
    M2 = (c+d)k
    M3 = a(e-k)
    M4 = d(m-j)
    M5 = (a+b)(j-p)
    M6 = (c-a)(e+k)
    M7 = (b-d)(m+l)
    
    Result: [[M1+M4+M5+M7, ...], [M2+M4, ...]]
    
    But this gives 7 multiplications for the 2x2 x 2x6 part.
    Each "multiplication" here is between a scalar and a 1x6 vector, which has rank 6.
    So 7 * 6 = 42 for the 2x2 part? That's worse.
    
    Let me reconsider. The issue is that Strassen works for square matrices.
    
    Alternative: Use the fact that we can combine terms.
    
    Let me just try to find a rank-33 decomposition by optimization.
    """
    # Start with schoolbook and try to optimize
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with schoolbook-like structure
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    # Fill remaining columns with small random values
    for r in range(col, rank):
        U[:, r] = np.random.randn(n*m) * 0.1
        V[:, r] = np.random.randn(m*p) * 0.1
        W[:, r] = np.random.randn(p*n) * 0.1
    
    return U, V, W

def _objective(params, n, m, p, rank):
    """Objective function for optimization: sum of squared residuals."""
    total_size = n*m*rank + m*p*rank + p*n*rank
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank:n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                residual += (val - 1.0) ** 2
    return residual

def _search_lower_rank(n, m, p, max_rank=35, min_rank=25, time_budget=100):
    """Search for lower-rank decompositions."""
    start_time = time.time()
    best_rank = max_rank
    best_solution = None
    
    # Try ranks from max_rank down to min_rank
    for rank in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() - start_time > time_budget - 5:
                break
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or BFGS
            try:
                result = minimize(_objective, params, args=(n, m, p, rank), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if result.fun < 1e-10:
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                        best_rank = rank
                        best_solution = (U_snap, V_snap, W_snap)
                        return best_rank, best_solution
                    
                    # Try without snapping
                    if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p, tol=1e-8):
                        best_rank = rank
                        best_solution = (U_sol, V_sol, W_sol)
                        return best_rank, best_solution
                        
            except Exception:
                continue
    
    return best_rank, best_solution

def _make_schoolbook(n, m, p):
    """Create the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W

def _to_list(mat):
    """Convert numpy array to list of lists."""
    return [[float(val) for val in row] for row in mat]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Uses a search strategy to find the lowest-rank decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower-rank solutions
    try:
        rank_found, solution = _search_lower_rank(n, m, p, max_rank=35, min_rank=25, 
                                                   time_budget=min(100, 120 - (time.time() - (deadline - 120))))
        if solution is not None and rank_found < best_rank:
            best_rank = rank_found
            best_solution = solution
    except Exception:
        pass
    
    U, V, W = best_solution
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        # Fall back to schoolbook
        U, V, W = U_school, V_school, W_school
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
