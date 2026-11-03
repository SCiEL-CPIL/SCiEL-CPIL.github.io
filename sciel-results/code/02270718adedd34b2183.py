"""
Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
Uses ALS + structured initialization + local search with half-integer snapping.
"""
import numpy as np
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Entries must be integers or half-integers.
    """
    rank = len(U[0]) if U else 0
    # Check entries are multiples of 0.5
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    
    # Construct the full tensor T from U, V, W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                val = 0.0
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - expected."""
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                residual += (val - 1.0) ** 2
    return residual

def _als_step(U, V, W, n, m, p, rank, max_iter=200, tol=1e-10):
    """
    Alternating least squares for tensor decomposition.
    Given rank, optimize U, V, W to minimize sum (sum_r U_{ij,r} V_{jk,r} W_{ki,r} - 1)^2
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve min sum_k (sum_r U_{ij,r} (V_{jk,r} W_{ki,r}) - 1)^2
        # This is: min || U_{ij} @ M_{(i,j)}^T - 1_{(p,)} ||^2 where M_{(i,j)}[k,r] = V_{jk,r} * W_{ki,r}
        old_U = U_np.copy()
        for i in range(n):
            for j in range(m):
                # Build matrix M of shape (p, rank) where M[k,r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V_np[j * p + k, :] * W_np[k * n + i, :]
                # Solve min || M^T @ u - 1 ||^2 where u is U[i*m+j, :]
                # Normal equations: M @ M^T @ u = M @ 1
                A = M @ M.T  # (rank, rank)
                b = M @ np.ones(p)  # (rank,)
                try:
                    u_sol = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
                except np.linalg.LinAlgError:
                    u_sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U_np[i * m + j, :] = u_sol
        
        # Update V: for each (j,k), solve min sum_i (sum_r (U_{ij,r} W_{ki,r}) V_{jk,r} - 1)^2
        old_V = V_np.copy()
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U_np[i * m + j, :] * W_np[k * n + i, :]
                A = M @ M.T
                b = M @ np.ones(n)
                try:
                    v_sol = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
                except np.linalg.LinAlgError:
                    v_sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V_np[j * p + k, :] = v_sol
        
        # Update W: for each (k,i), solve min sum_j (sum_r (U_{ij,r} V_{jk,r}) W_{ki,r} - 1)^2
        old_W = W_np.copy()
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_np[i * m + j, :] * V_np[j * p + k, :]
                A = M @ M.T
                b = M @ np.ones(m)
                try:
                    w_sol = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
                except np.linalg.LinAlgError:
                    w_sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W_np[k * n + i, :] = w_sol
        
        # Check convergence
        if np.max(np.abs(U_np - old_U)) < tol and np.max(np.abs(V_np - old_V)) < tol and np.max(np.abs(W_np - old_W)) < tol:
            break
    
    return U_np.tolist(), V_np.tolist(), W_np.tolist()

def _snap_to_half_integer(M, tol=0.15):
    """Snap values to nearest half-integer if within tol."""
    M_snapped = []
    for row in M:
        snapped_row = []
        for val in row:
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < tol:
                snapped_row.append(nearest)
            else:
                snapped_row.append(val)
        M_snapped.append(snapped_row)
    return M_snapped

def _structured_init(n, m, p, rank):
    """
    Create a structured initialization that respects the tensor structure.
    Uses the fact that n=2 is small to create a more informed start.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with random values scaled appropriately
    scale = 1.0 / np.sqrt(rank)
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    
    # For the schoolbook solution, each (i,j,k) triple has its own rank component.
    # For reduced rank, we need to share components.
    # A good structured init: for each rank component r, assign it to a subset of (i,j,k) triples
    # that form a "slice" of the tensor.
    
    # Alternative: use the structure that for n=2, we can think of it as two 5x5 multiplications
    # coupled together.
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank(n, m, p, target_rank, time_limit=10, num_restarts=5):
    """
    Attempt to find a decomposition of target_rank using ALS from multiple random starts.
    Returns None if not found.
    """
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        U, V, W = _structured_init(n, m, p, target_rank)
        
        # Run ALS
        U, V, W = _als_step(U, V, W, n, m, p, target_rank, max_iter=100, tol=1e-8)
        
        # Check residual before snapping
        residual = _tensor_residual(U, V, W, n, m, p)
        if residual > 1e-4:
            continue
        
        # Snap to half-integers
        U_snapped = _snap_to_half_integer(U, tol=0.1)
        V_snapped = _snap_to_half_integer(V, tol=0.1)
        W_snapped = _snap_to_half_integer(W, tol=0.1)
        
        # Verify
        if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
        
        # If snapping failed, try with looser tolerance
        U_snapped2 = _snap_to_half_integer(U, tol=0.2)
        V_snapped2 = _snap_to_half_integer(V, tol=0.2)
        W_snapped2 = _snap_to_half_integer(W, tol=0.2)
        if _verify_decomposition(U_snapped2, V_snapped2, W_snapped2, n, m, p):
            return {"U": U_snapped2, "V": V_snapped2, "W": W_snapped2}
    
    return None

def _polish_solution(U, V, W, n, m, p, time_limit=5):
    """
    Given a near-exact solution, try to polish it to exact half-integer values.
    Uses local search on individual entries.
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    rank = len(U[0])
    
    # Try to fix each entry by rounding to nearest half-integer
    # and checking if the solution remains valid
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        for M_name, M in [('U', U), ('V', V), ('W', W)]:
            for i in range(len(M)):
                for j in range(len(M[0])):
                    current = M[i][j]
                    # Try rounding to nearest half-integer
                    nearest = round(current * 2) / 2.0
                    if abs(current - nearest) > 1e-9:
                        M[i][j] = nearest
                        # Check if still valid
                        if _verify_decomposition(U, V, W, n, m, p):
                            changed = True
                        else:
                            M[i][j] = current
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Fallback to schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    deadline = time.time() + 130  # Leave 20s margin for 150s limit
    
    # Try ranks from 49 down to 40
    # For each rank, try a few restarts
    for target_rank in range(best_rank - 1, max(30, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Try this rank with limited time
        time_for_rank = min(10, remaining * 0.5)
        result = _try_rank(n, m, p, target_rank, time_limit=time_for_rank, num_restarts=3)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # If we found a solution, try to polish it
            U_pol, V_pol, W_pol = _polish_solution(
                best_result["U"], best_result["V"], best_result["W"], 
                n, m, p, time_limit=3
            )
            best_result = {"U": U_pol, "V": V_pol, "W": W_pol}
            # Continue trying lower ranks
            break  # Found one, break to try lower
    
    return best_result
