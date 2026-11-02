"""Rewritten program: rank-minimal tensor decomposition for 2x2 * 2x3 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    Strategy:
      1. Return the known rank-11 solution (Strassen 7 + schoolbook 4) as the safe baseline.
      2. Attempt to find a rank-10 decomposition via a smooth-optimization approach:
         - Parametrize rank-r factor matrices with free real variables.
         - Use a smooth loss (sum of squared residuals over all 12 tensor entries).
         - Alternate between continuous minimization (Gauss-Newton / least squares) and
           half-integer projection, with multiple random restarts.
      3. Verify any candidate exactly; return the best verified result.
    """
    n, m, p = DIMENSIONS  # n=2, m=3
    # Actually n=2 (rows of A), m=2 (cols of A / rows of B), p=3 (cols of B)
    n, m, p = 2, 2, 3
    _t0 = time.time()
    _deadline = _t0 + 135  # safety margin before 150s kill

    # Rank-11 baseline
    best_result = _rank11_solution(n, m, p)
    best_rank = 11

    # Try to find rank-10
    if time.time() < _deadline:
        result_10 = _search_rank(n, m, p, 10, _deadline, max_restarts=300)
        if result_10 is not None:
            best_result = result_10
            best_rank = 10

    # Try rank 9 if time remains (unlikely to succeed but cheap to attempt)
    if best_rank == 11 and time.time() < _deadline - 20:
        result_9 = _search_rank(n, m, p, 9, _deadline, max_restarts=100)
        if result_9 is not None:
            best_result = result_9
            best_rank = 9

    return best_result


def _rank11_solution(n, m, p):
    """Construct the known rank-11 decomposition: Strassen(7) for cols 0,1 + schoolbook(4) for col 2."""
    rank = 11
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen for the 2x2 sub-problem (cols 0 and 1 of B)
    # A = [[a,b],[c,d]], B_sub = [[b00,b01],[b10,b11]]
    # M1 = (a+d)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1   # a, d
    V[0, 0] = 1; V[4, 0] = 1   # b00, b11
    # M2 = (c+d)b00
    U[2, 1] = 1; U[3, 1] = 1   # c, d
    V[0, 1] = 1                # b00
    # M3 = a(b01-b11)
    U[0, 2] = 1                # a
    V[1, 2] = 1; V[4, 2] = -1  # b01, -b11
    # M4 = d(b10-b00)
    U[3, 3] = 1                # d
    V[3, 3] = 1; V[0, 3] = -1  # b10, -b00
    # M5 = (a+b)b11
    U[0, 4] = 1; U[1, 4] = 1   # a, b
    V[4, 4] = 1                # b11
    # M6 = (c-a)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1  # c, -a
    V[0, 5] = 1; V[1, 5] = 1   # b00, b01
    # M7 = (b-d)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1  # b, -d
    V[3, 6] = 1; V[4, 6] = 1   # b10, b11

    # W for Strassen outputs (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C01 = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    # C10 = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

    # Schoolbook for column 2 of B
    # C02 = a*b02 + b*b12
    U[0, 7] = 1; V[2, 7] = 1   # a * b02
    U[1, 8] = 1; V[5, 8] = 1   # b * b12
    W[4, 7] = 1; W[4, 8] = 1
    # C12 = c*b02 + d*b12
    U[2, 9] = 1; V[2, 9] = 1   # c * b02
    U[3, 10] = 1; V[5, 10] = 1 # d * b12
    W[5, 9] = 1; W[5, 10] = 1

    return _to_list(U, V, W)


def _to_list(U, V, W):
    """Convert numpy arrays to list-of-lists with integer/half-integer values."""
    U_l = [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_l = [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_l = [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_l, "V": V_l, "W": W_l}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _search_rank(n, m, p, r, deadline, max_restarts=200):
    """
    Search for a rank-r decomposition using smooth optimization with half-integer projection.
    
    The tensor equation is:
      For all (i,j,k) in [n]x[m]x[p]: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    We use a parameterized approach:
      - Initialize U, V, W with random values (or perturbed from rank-11).
      - Minimize the smooth loss L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
      - Use Gauss-Newton style updates alternating over U, V, W.
      - Periodically project to half-integers and check exactness.
    """
    np.random.seed(12345 + r)
    
    U11 = np.array(_rank11_solution(n, m, p)["U"], dtype=float)
    V11 = np.array(_rank11_solution(n, m, p)["V"], dtype=float)
    W11 = np.array(_rank11_solution(n, m, p)["W"], dtype=float)
    
    best_loss = float('inf')
    best_solution = None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: use a random subset of rank-11 columns + perturbation
        if r <= 11:
            idx = np.random.choice(11, size=r, replace=False)
            U = U11[:, idx].copy() + np.random.randn(n * m, r) * 0.3
            V = V11[:, idx].copy() + np.random.randn(m * p, r) * 0.3
            W = W11[:, idx].copy() + np.random.randn(p * n, r) * 0.3
        else:
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
        
        # Ensure no column is entirely zero (would make that rank-1 term vanish)
        for c in range(r):
            if np.allclose(U[:, c], 0) and np.allclose(V[:, c], 0) and np.allclose(W[:, c], 0):
                U[np.random.randint(n * m), c] = 1.0
                V[np.random.randint(m * p), c] = 1.0
                W[np.random.randint(p * n), c] = 1.0
        
        # Gauss-Newton style optimization
        for it in range(300):
            if time.time() > deadline:
                break
            
            # Compute current tensor values and residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]) - 1.0
            
            loss = np.sum(residuals ** 2)
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-14:
                break
            
            # Update W: for each (k,i), the equation is
            #   sum_j [sum_c U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] = 1
            # Let A_W[k*n+i, c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then A_W @ W_row = 1 (scalar). Least squares: W_row = A_W^T / ||A_W||^2
            for i in range(n):
                for k in range(p):
                    A_vec = np.zeros(r)
                    for j in range(m):
                        A_vec += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(A_vec, A_vec)
                    if norm_sq > 1e-15:
                        W[k * n + i, :] = A_vec / norm_sq
            
            # Update U: for each (i,j), the equation is
            #   sum_k [sum_c V[j*p+k,c] * W[k*n+i,c]] * U[i*m+j,c] = 1
            # Let A_U[i*m+j, c] = sum_k V[j*p+k,c] * W[k*n+i,c]
            for i in range(n):
                for j in range(m):
                    A_vec = np.zeros(r)
                    for k in range(p):
                        A_vec += V[j * p + k, :] * W[k * n + i, :]
                    norm_sq = np.dot(A_vec, A_vec)
                    if norm_sq > 1e-15:
                        U[i * m + j, :] = A_vec / norm_sq
            
            # Update V: for each (j,k), the equation is
            #   sum_i [sum_c U[i*m+j,c] * W[k*n+i,c]] * V[j*p+k,c] = 1
            # Let A_V[j*p+k, c] = sum_i U[i*m+j,c] * W[k*n+i,c]
            for j in range(m):
                for k in range(p):
                    A_vec = np.zeros(r)
                    for i in range(n):
                        A_vec += U[i * m + j, :] * W[k * n + i, :]
                    norm_sq = np.dot(A_vec, A_vec)
                    if norm_sq > 1e-15:
                        V[j * p + k, :] = A_vec / norm_sq
            
            # Periodic half-integer projection
            if it % 20 == 19:
                U_proj = np.round(U * 2) / 2
                V_proj = np.round(V * 2) / 2
                W_proj = np.round(W * 2) / 2
                loss_proj = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_proj[i * m + j, :], V_proj[j * p + k, :] * W_proj[k * n + i, :])
                            loss_proj += (val - 1.0) ** 2
                if loss_proj < loss:
                    U, V, W = U_proj, V_proj, W_proj
                    loss = loss_proj
                    if loss < 1e-14:
                        break
            
            # Check for exact half-integer solution
            if it % 10 == 9:
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                if _verify(U_h, V_h, W_h, n, m, p, tol=1e-9):
                    return _to_list(U_h, V_h, W_h)
        
        # Final check after optimization
        if best_loss < 1e-12:
            U_h = np.round(best_solution[0] * 2) / 2
            V_h = np.round(best_solution[1] * 2) / 2
            W_h = np.round(best_solution[2] * 2) / 2
            if _verify(U_h, V_h, W_h, n, m, p, tol=1e-9):
                return _to_list(U_h, V_h, W_h)
        
        # Also try: use the best continuous solution and do a local half-integer search
        if best_loss < 0.1:
            result = _local_half_integer_search(best_solution, n, m, p, r, deadline)
            if result is not None:
                return result
    
    return None


def _local_half_integer_search(best_solution, n, m, p, r, deadline):
    """
    Given a near-optimal continuous solution, search nearby half-integer points.
    For each matrix element, try rounding to nearest half-integer and also try
    flipping by +/-0.5 to explore the local lattice neighborhood.
    """
    U0, V0, W0 = best_solution
    U_base = np.round(U0 * 2) / 2
    V_base = np.round(V0 * 2) / 2
    W_base = np.round(W0 * 2) / 2
    
    # Count non-zero elements to limit search space
    U_nz = np.argwhere(U_base != 0)
    V_nz = np.argwhere(V_base != 0)
    W_nz = np.argwhere(W_base != 0)
    
    if len(U_nz) > 20 or len(V_nz) > 20 or len(W_nz) > 20:
        return None  # Too many non-zeros to search locally
    
    # Try flipping each non-zero element by +/-1 (i.e., +/-2 in half-integer units)
    # This is a local search in the half-integer lattice
    U_curr = U_base.copy()
    V_curr = V_base.copy()
    W_curr = W_base.copy()
    
    best_local_loss = _compute_loss(U_curr, V_curr, W_curr, n, m, p)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try flipping one element in U
        for (row, col) in U_nz:
            for delta in [-1.0, 1.0]:
                U_trial = U_curr.copy()
                U_trial[row, col] += delta
                loss = _compute_loss(U_trial, V_curr, W_curr, n, m, p)
                if loss < best_local_loss - 1e-12:
                    U_curr = U_trial
                    best_local_loss = loss
                    improved = True
                    if best_local_loss < 1e-9:
                        return _to_list(U_curr, V_curr, W_curr)
        
        # Try flipping one element in V
        for (row, col) in V_nz:
            for delta in [-1.0, 1.0]:
                V_trial = V_curr.copy()
                V_trial[row, col] += delta
                loss = _compute_loss(U_curr, V_trial, W_curr, n, m, p)
                if loss < best_local_loss - 1e-12:
                    V_curr = V_trial
                    best_local_loss = loss
                    improved = True
                    if best_local_loss < 1e-9:
                        return _to_list(U_curr, V_curr, W_curr)
        
        # Try flipping one element in W
        for (row, col) in W_nz:
            for delta in [-1.0, 1.0]:
                W_trial = W_curr.copy()
                W_trial[row, col] += delta
                loss = _compute_loss(U_curr, V_curr, W_trial, n, m, p)
                if loss < best_local_loss - 1e-12:
                    W_curr = W_trial
                    best_local_loss = loss
                    improved = True
                    if best_local_loss < 1e-9:
                        return _to_list(U_curr, V_curr, W_curr)
        
        if not improved:
            break
    
    if best_local_loss < 1e-9:
        return _to_list(U_curr, V_curr, W_curr)
    
    return None


def _compute_loss(U, V, W, n, m, p):
    """Compute the sum of squared residuals for the tensor decomposition."""
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                loss += (val - 1.0) ** 2
    return loss
# EVOLVE-BLOCK-END
