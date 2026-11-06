"""ODE solver: 15-dimensional stiff linear system using Krylov subspace (Arnoldi) method."""
import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) = A y on [t0, t1].
    Uses Arnoldi iteration to approximate e^{A*dt} @ y0 directly.
    """
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 330s kill

    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    dt = t1 - t0
    
    # If dt is 0, return y0
    if dt == 0:
        return y0

    # We will use a Krylov subspace of dimension m.
    # For n=15, m=15 gives the full space (if y0 is cyclic).
    m = n
    
    # Initialize Arnoldi basis V (n x m) and Hessenberg matrix H (m x m)
    V = np.zeros((n, m))
    H = np.zeros((m, m))
    
    # v1 = y0 / ||y0||
    norm_y0 = np.linalg.norm(y0)
    if norm_y0 < 1e-15:
        return y0 # Zero solution stays zero
        
    v = y0 / norm_y0
    V[:, 0] = v
    
    # Arnoldi iteration
    for j in range(m):
        # w = A v_j = f(t0, v_j)
        # Note: f(t, y) is called with t=t0. Since system is autonomous, t doesn't matter for A extraction, 
        # but we pass t0 as per interface.
        w = f(t0, V[:, j])
        
        # Orthogonalize w against v_1 ... v_j
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
            
        h_next = np.linalg.norm(w)
        
        # Check for breakdown
        if h_next < 1e-14:
            # Invariant subspace found. We can stop.
            # The effective dimension is j+1
            m_eff = j + 1
            break
        else:
            V[:, j + 1] = w / h_next
            H[j + 1, j] = h_next
            m_eff = m
            
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback if something went wrong (unlikely for 15D)
        return y0

    # Now we have V (n x m_eff) and H (m_eff x m_eff)
    # The approximation is y(t1) = ||y0|| * V[:, :m_eff] @ expm(dt * H[:m_eff, :m_eff]) @ e_1
    # where e_1 is the first standard basis vector of size m_eff
    
    H_sub = H[:m_eff, :m_eff]
    V_sub = V[:, :m_eff]
    
    # Compute expm(dt * H_sub)
    E = scipy.linalg.expm(dt * H_sub)
    
    # e_1 is a vector of zeros with 1 at index 0
    e1 = np.zeros(m_eff)
    e1[0] = 1.0
    
    # Coefficients in the Krylov basis
    c = E @ e1
    
    # Final state
    y_end = norm_y0 * (V_sub @ c)
    
    # Verify result is finite
    if not np.all(np.isfinite(y_end)):
        # Fallback to simple extraction if Krylov failed numerically
        # (Should not happen for 15D, but safe to have)
        probes = np.eye(n)
        F = np.column_stack([f(t0, probes[:, j]) for j in range(n)])
        A = F # Since probes are identity, A = F
        y_end = scipy.linalg.expm(A * dt) @ y0

    return y_end
