import numpy as np
from scipy.linalg import expm, eigh
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    Uses structural inference to reconstruct A with minimal function calls.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 300s limit

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0 or dt == 0:
        return y0

    if N != 100:
        return _general_solve(f, t0, t1, y0, deadline)

    # 1. Probe corner (0,0) -> index 0
    e0 = np.zeros(N)
    e0[0] = 1.0
    try:
        col0 = np.asarray(f(t0, e0), dtype=float).flatten()
    except Exception:
        return _fallback_rk4(f, t0, t1, y0)

    if time.time() > deadline:
        return _fallback_rk4(f, t0, t1, y0)

    # 2. Probe interior (5,5) -> index 55
    e55 = np.zeros(N)
    e55[55] = 1.0
    try:
        col55 = np.asarray(f(t0, e55), dtype=float).flatten()
    except Exception:
        return _fallback_rk4(f, t0, t1, y0)

    if time.time() > deadline:
        return _fallback_rk4(f, t0, t1, y0)

    # 3. Infer weights
    # For a standard 2D Laplacian on a grid:
    # A[i, j] corresponds to node (i, j).
    # col0 is A @ e0, which is the first column of A.
    # A[0, 0] is the diagonal for corner (0,0).
    # A[1, 0] is the coupling to (0,1) [right].
    # A[10, 0] is the coupling to (1,0) [down].
    
    w_diag_corner = col0[0]
    w_right = col0[1]
    w_down = col0[10]
    
    # Check if weights are consistent
    if np.isclose(w_right, w_down, rtol=1e-9, atol=1e-12):
        w_off = w_right
    else:
        # Anisotropic or non-standard? Try to use average or fallback
        w_off = (w_right + w_down) / 2.0

    # For interior node 55 (5,5):
    # Diagonal should be w_diag_int
    # Neighbors: 54 (up), 56 (down), 55-1 (left), 55+1 (right)
    # In a standard Laplacian, w_diag_int = w_diag_corner + 2*w_off? 
    # No, for Dirichlet:
    # Corner: -2 (1D) -> -4 (2D)? 
    # Let's check the values.
    # If Dirichlet:
    # 1D Laplacian L: diag -2, off 1.
    # 2D A = Lx I + I Ly.
    # Corner (0,0): A[0,0] = L[0,0] + L[0,0] = -2 + -2 = -4.
    # Interior (5,5): A[55,55] = L[5,5] + L[5,5] = -2 + -2 = -4.
    # So diagonal is constant -4 for Dirichlet.
    # Off-diagonals are 1.
    
    # If Periodic:
    # 1D L: diag -2, off 1 (cyclic).
    # 2D A: diag -4. Off-diagonals 1.
    
    # If Neumann (standard 2nd order):
    # 1D L: diag -2 (interior), -1 (boundary) if using ghost u[-1]=u[1]? 
    # Actually, common Neumann implementation: 
    # d^2u/dx^2 at boundary x=0: (u[1] - u[0]) / h^2 (1st order) or (2u[0] - 2u[1]) / h^2?
    # Let's just construct candidates and check residual.

    candidates = []
    
    # Candidate 1: Dirichlet
    A_dir = _build_laplacian_dirichlet(10, w_diag_corner, w_off)
    candidates.append(A_dir)
    
    # Candidate 2: Periodic
    A_per = _build_laplacian_periodic(10, w_diag_corner, w_off)
    candidates.append(A_per)
    
    # Candidate 3: Neumann (One-sided / Ghost node u[-1]=u[1] implies off-diag 2 at boundary? 
    # Or simpler: just assume diagonal is same, but boundary neighbors missing?
    # Let's try a standard Neumann where boundary diagonal is adjusted.
    # For 1D Neumann with ghost u[-1] = u[1]:
    # (u[1] - 2u[0] + u[-1]) / h^2 = (2u[1] - 2u[0]) / h^2.
    # So L[0,0] = -2, L[0,1] = 2.
    # Let's build this.
    A_neu = _build_laplacian_neumann(10, w_diag_corner, w_off)
    candidates.append(A_neu)

    # Evaluate residuals for all candidates
    best_A = None
    best_res = float('inf')
    
    for A_cand in candidates:
        # Check col0
        res0 = np.linalg.norm(A_cand @ e0 - col0)
        # Check col55
        res55 = np.linalg.norm(A_cand @ e55 - col55)
        total_res = res0 + res55
        
        if total_res < best_res:
            best_res = total_res
            best_A = A_cand

    # Check if best candidate is good enough
    scale = np.linalg.norm(col0) + np.linalg.norm(col55)
    if scale > 0 and best_res / scale < 1e-8:
        A = best_A
    else:
        # If structure inference failed, fallback to full reconstruction or RK4
        # Given the strict call budget, if structure fails, we might be in trouble.
        # But for a standard Heat equation, it should work.
        # Let's try one more probe: an edge node (0, 5) -> index 5
        e5 = np.zeros(N)
        e5[5] = 1.0
        try:
            col5 = np.asarray(f(t0, e5), dtype=float).flatten()
        except Exception:
            return _fallback_rk4(f, t0, t1, y0)
            
        # Re-evaluate with 3 probes
        best_A = None
        best_res = float('inf')
        for A_cand in candidates:
            res0 = np.linalg.norm(A_cand @ e0 - col0)
            res55 = np.linalg.norm(A_cand @ e55 - col55)
            res5 = np.linalg.norm(A_cand @ e5 - col5)
            total_res = res0 + res55 + res5
            if total_res < best_res:
                best_res = total_res
                best_A = A_cand
        
        scale = np.linalg.norm(col0) + np.linalg.norm(col55) + np.linalg.norm(col5)
        if scale > 0 and best_res / scale < 1e-8:
            A = best_A
        else:
            # Last resort: Full reconstruction is too expensive (100 calls).
            # Use RK4 with high steps? Or just return best guess?
            # Let's use RK4 with 1000 steps.
            return _fallback_rk4(f, t0, t1, y0)

    # Compute exact solution
    try:
        # Check symmetry
        if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
            eigvals, eigvecs = eigh(A)
            exp_eigvals = np.exp(dt * eigvals)
            y_proj = eigvecs.T @ y0
            y_end = eigvecs @ (exp_eigvals * y_proj)
        else:
            expA = expm(dt * A)
            y_end = expA @ y0
    except Exception:
        y_end = _fallback_rk4(f, t0, t1, y0)

    return y_end

def _build_laplacian_dirichlet(n, w_diag, w_off):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            if j > 0:
                A[idx, i * n + (j - 1)] = w_off
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_off
            if i > 0:
                A[idx, (i - 1) * n + j] = w_off
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_off
    return A

def _build_laplacian_periodic(n, w_diag, w_off):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            A[idx, i * n + ((j - 1) % n)] = w_off
            A[idx, i * n + ((j + 1) % n)] = w_off
            A[idx, ((i - 1) % n) * n + j] = w_off
            A[idx, ((i + 1) % n) * n + j] = w_off
    return A

def _build_laplacian_neumann(n, w_diag, w_off):
    # Assuming 1D Neumann with ghost u[-1]=u[1] leads to L[0,0]=-2, L[0,1]=2.
    # So in 2D, corner diagonal = -4, edge off-diag = 2?
    # Let's construct based on 1D logic.
    # L_1D:
    # L[0,0] = w_diag_1d
    # L[0,1] = w_off_1d * 2 (if ghost)
    # L[i,i] = w_diag_1d for i>0
    # L[i, i+1] = w_off_1d
    # But we only have w_diag_corner (2D) and w_off (2D).
    # If Dirichlet: w_diag_corner = 2 * w_diag_1d, w_off = w_off_1d.
    # If Neumann (ghost): w_diag_corner = 2 * w_diag_1d, w_off_edge = 2 * w_off_1d.
    # So w_off_edge should be 2 * w_off_interior.
    # We inferred w_off from corner neighbors. In Neumann, corner neighbors are edges.
    # So w_off_inferred is likely 2 * w_off_1d.
    # Let's build with this assumption.
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] = w_off
            else:
                A[idx, i * n + 1] = w_off * 2.0 # Ghost node effect
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_off
            else:
                A[idx, i * n + (n - 2)] = w_off * 2.0 # Ghost node effect
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] = w_off
            else:
                A[idx, 1 * n + j] = w_off * 2.0
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_off
            else:
                A[idx, (n - 2) * n + j] = w_off * 2.0
    return A

def _fallback_rk4(f, t0, t1, y0, steps=1000):
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y

def _general_solve(f, t0, t1, y0, deadline):
    # Fallback for non-100 size
    N = len(y0)
    # Try to reconstruct A with N calls
    A = np.zeros((N, N))
    for i in range(N):
        if time.time() > deadline - 10:
            break
        e_i = np.zeros(N)
        e_i[i] = 1.0
        try:
            A[:, i] = np.asarray(f(t0, e_i), dtype=float).flatten()
        except Exception:
            return _fallback_rk4(f, t0, t1, y0)
    
    try:
        if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
            eigvals, eigvecs = eigh(A)
            exp_eigvals = np.exp((t1 - t0) * eigvals)
            y_proj = eigvecs.T @ y0
            return eigvecs @ (exp_eigvals * y_proj)
        else:
            expA = expm((t1 - t0) * A)
            return expA @ y0
    except Exception:
        return _fallback_rk4(f, t0, t1, y0)
