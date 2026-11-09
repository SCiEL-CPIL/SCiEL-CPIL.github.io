"""Program for matrix multiplication 3,3,8 tensor decomposition search."""
import time
import numpy as np
from itertools import product
from scipy.optimize import minimize, linear_sum_assignment

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    total = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_r[:, :, c]  # (n, m)
        v_c = V_r[:, :, c]  # (m, p)
        w_c = W_r[:, :, c].T  # (n, p)
        total += u_c[:, :, None] * v_c[None, :, :] * w_c[:, None, :]
    
    target = np.ones((n, m, p))
    return np.allclose(total, target, atol=tol, rtol=0)


def _round_to_half_int(arr, tol=1e-6):
    """Round array to nearest half-integer, return None if not close enough."""
    rounded = np.round(arr * 2) / 2
    if np.max(np.abs(arr - rounded)) > tol:
        return None
    return rounded


def _try_polish(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """
    Try to polish a near-exact decomposition using alternating least squares.
    Returns polished (U, V, W) or None if it fails.
    """
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    
    n, m, p = 3, 3, 8
    rank = len(U_arr[0])
    
    # Reshape for computation
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    for iteration in range(max_iters):
        # Solve for W given U, V
        # For each (i, j, k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        # Group by (i, k): sum_c (sum_j U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
        # Actually, let's think about this differently.
        
        # The constraint is: for all i, j, k:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        
        # Let's fix U and V, solve for W.
        # For fixed (i, k), we need: sum_c W[k,i,c] * (sum_j U[i,j,c] * V[j,k,c]) = 1
        # Wait, that's not quite right either. Let me re-derive.
        
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
        
        # Let A[i,j,c] = U[i,j,c], B[j,k,c] = V[j,k,c], C[k,i,c] = W[k,i,c]
        # sum_c A[i,j,c] * B[j,k,c] * C[k,i,c] = 1
        
        # If we fix A and B, for each (k, i), we have a linear equation in C[k,i,:]:
        # sum_c C[k,i,c] * (sum_j A[i,j,c] * B[j,k,c]) = 1
        # Let D[k,i,c] = sum_j A[i,j,c] * B[j,k,c]
        # Then: D[k,i,:] . C[k,i,:] = 1
        
        # This is one equation with rank unknowns for each (k,i) pair.
        # There are p*n = 24 such pairs, each with rank unknowns and 1 constraint.
        # This is underdetermined if rank > 1, so we need more constraints.
        
        # Actually, the constraint must hold for ALL i, j, k simultaneously.
        # Let me think of it as a linear system.
        
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
        # That's p*n*rank = 24*rank unknowns.
        
        # Constraints: for each (i, j, k), we have:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # That's n*m*p = 72 constraints.
        
        # So we have 72 equations and 24*rank unknowns.
        # For this to be solvable, we need 24*rank >= 72, i.e., rank >= 3.
        
        # Let's set up the linear system.
        # Unknown vector: x of length p*n*rank
        # x[k*n*rank + i*rank + c] = W[k*n+i, c]
        
        # Equation for (i, j, k):
        # sum_c U[i*m+j, c] * V[j*p+k, c] * x[k*n*rank + i*rank + c] = 1
        
        # Build the matrix A_eq of shape (72, 24*rank)
        # And vector b of shape (72,)
        
        if rank < 3:
            continue
            
        num_constraints = n * m * p  # 72
        num_unknowns = p * n * rank  # 24 * rank
        
        if num_unknowns < num_constraints:
            continue
        
        A_eq = np.zeros((num_constraints, num_unknowns))
        b = np.ones(num_constraints)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U_arr[i*m+j, c] * V_arr[j*p+k, c]
                        A_eq[idx, k*n*rank + i*rank + c] = coeff
                    idx += 1
        
        # Solve the least squares problem
        try:
            x_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_eq, b, rcond=None)
        except:
            return None
        
        # Check residual
        if residuals and residuals[0] > 1e-6:
            continue
        
        # Reshape solution to W
        W_new = x_sol.reshape(p, n, rank).transpose(1, 0, 2)  # (n, p, rank) -> but we need (p, n, rank)
        # Wait, x_sol is indexed as [k*n*rank + i*rank + c], so reshape to (p, n, rank)
        W_new = x_sol.reshape(p, n, rank)
        
        # Update W
        W_arr = W_new.reshape(p*n, rank)
        W_r = W_new.copy()
        
        # Now solve for V given U, W
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed (j, k): sum_c V[j*p+k, c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1
        # Let E[j,k,c] = sum_i U[i*m+j, c] * W[k*n+i, c]
        # Then: E[j,k,:] . V[j*p+k, :] = 1
        
        num_unknowns_v = m * p * rank  # 24 * rank
        A_eq_v = np.zeros((num_constraints, num_unknowns_v))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U_arr[i*m+j, c] * W_arr[k*n+i, c]
                        A_eq_v[idx, j*p*rank + k*rank + c] = coeff
                    idx += 1
        
        try:
            x_sol_v, residuals_v, _, _ = np.linalg.lstsq(A_eq_v, b, rcond=None)
        except:
            continue
        
        if residuals_v and residuals_v[0] > 1e-6:
            continue
        
        V_new = x_sol_v.reshape(m, p, rank)
        V_arr = V_new.reshape(m*p, rank)
        V_r = V_new.copy()
        
        # Now solve for U given V, W
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed (i, j): sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
        # Let F[i,j,c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        # Then: F[i,j,:] . U[i*m+j, :] = 1
        
        num_unknowns_u = n * m * rank  # 9 * rank
        A_eq_u = np.zeros((num_constraints, num_unknowns_u))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        A_eq_u[idx, i*m*rank + j*rank + c] = coeff
                    idx += 1
        
        try:
            x_sol_u, residuals_u, _, _ = np.linalg.lstsq(A_eq_u, b, rcond=None)
        except:
            continue
        
        if residuals_u and residuals_u[0] > 1e-6:
            continue
        
        U_new = x_sol_u.reshape(n, m, rank)
        U_arr = U_new.reshape(n*m, rank)
        U_r = U_new.copy()
        
        # Check if we've converged
        total = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U_r[:, :, c]
            v_c = V_r[:, :, c]
            w_c = W_r[:, :, c].T
            total += u_c[:, :, None] * v_c[None, :, :] * w_c[:, None, :]
        
        if np.max(np.abs(total - 1.0)) < tol:
            # Try to round to half-integers
            U_rounded = _round_to_half_int(U_arr, tol=1e-4)
            V_rounded = _round_to_half_int(V_arr, tol=1e-4)
            W_rounded = _round_to_half_int(W_arr, tol=1e-4)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                # Verify the rounded solution
                if _validate(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    return (U_rounded, V_rounded, W_rounded)
            
            # Return the exact floating point solution as a fallback
            return (U_arr, V_arr, W_arr)
    
    return None


def _construct_schoolbook(n, m, p):
    """Construct the schoolbook decomposition."""
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
    return U, V, W


def _try_structured_decomposition(n, m, p, max_rank=72, deadline=None):
    """
    Try to find a structured low-rank decomposition using Winograd-like ideas.
    """
    n, m, p = 3, 3, 8
    
    # Strategy: Try various rank values from low to high
    # For each rank, try to find a decomposition using optimization
    
    best_result = None
    best_rank = max_rank + 1
    
    for rank in range(30, max_rank + 1):
        if deadline and time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if deadline and time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(42 + rank * 100 + trial)
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            # Try to polish
            result = _try_polish(U_init, V_init, W_init, n, m, p, rank, max_iters=50)
            
            if result is not None:
                U_pol, V_pol, W_pol = result
                
                # Check if it's valid
                if _validate(U_pol, V_pol, W_pol, n, m, p, tol=1e-6):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_pol, V_pol, W_pol)
                        break  # Found a valid decomposition at this rank
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're going from low to high)
        if best_result is not None and best_rank == rank:
            break
    
    return best_result, best_rank


def _try_explicit_construction(n, m, p):
    """
    Try explicit constructions based on known algebraic identities.
    """
    n, m, p = 3, 3, 8
    
    # One approach: use the fact that we can share computations
    # across the 8 output columns.
    
    # For each row i of A (i = 0, 1, 2), we compute:
    # C[i, :] = A[i, 0] * B[0, :] + A[i, 1] * B[1, :] + A[i, 2] * B[2, :]
    
    # This is a sum of 3 scaled rows of B.
    # If we could compute these with shared multiplications, we'd save.
    
    # Let me try a different approach: use the tensor structure directly.
    
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
    # We want to decompose it as a sum of rank-1 tensors.
    
    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i].
    
    # Let me try to find a decomposition with rank < 72.
    
    # One idea: use the fact that the tensor is symmetric in some sense.
    # Actually, it's not symmetric, but it has a regular structure.
    
    # Let me try a greedy approach:
    # 1. Start with the full tensor
    # 2. At each step, find the best rank-1 approximation
    # 3. Subtract it and continue
    
    # This is essentially the CP decomposition algorithm.
    
    # For exact decomposition, this might not work well.
    # Let me try a different approach.
    
    # Actually, let me just return None and rely on the schoolbook.
    return None, 72


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 72
    
    # Try to find a better decomposition
    if time.time() < deadline:
        result, rank = _try_structured_decomposition(n, m, p, max_rank=71, deadline=deadline)
        
        if result is not None and rank < best_rank:
            U_pol, V_pol, W_pol = result
            best_result = {"U": U_pol, "V": V_pol, "W": W_pol}
            best_rank = rank
    
    if time.time() < deadline:
        result, rank = _try_explicit_construction(n, m, p)
        
        if result is not None and rank < best_rank:
            U_pol, V_pol, W_pol = result
            best_result = {"U": U_pol, "V": V_pol, "W": W_pol}
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
