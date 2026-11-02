"""Soft-min optimization for Erdos minimum overlap using FFT-based gradients."""
import numpy as np
import time

N = 100
DEADLINE = 135  # seconds, safety margin before 150s kill


def _compute_overlaps(h):
    """
    Compute all overlap values C_k for k=0..N-1.
    C_k = sum_{i=0}^{N-k-1} h_i * (1 - h_{i+k})
    """
    h = np.asarray(h, dtype=np.float64)
    overlaps = np.zeros(N)
    for k in range(N):
        if k == 0:
            overlaps[k] = np.sum(h * (1.0 - h))
        else:
            overlaps[k] = np.sum(h[:N-k] * (1.0 - h[k:]))
    return overlaps


def _soft_min_obj_and_grad(h, tau=10.0):
    """
    Compute soft-min objective and gradient using log-sum-exp trick.
    Objective: tau * log(sum(exp(C_k / tau))) approximates max(C_k)
    Gradient: sum_k (exp(C_k/tau) / sum(exp(C_j/tau))) * grad(C_k)
    """
    h = np.asarray(h, dtype=np.float64)
    C = _compute_overlaps(h)
    
    # Soft min: tau * log(sum(exp(C_k / tau)))
    # For numerical stability, subtract max
    C_shifted = C - np.max(C)
    weights = np.exp(C_shifted / tau)
    Z = np.sum(weights)
    obj = np.max(C) + tau * np.log(Z)
    
    # Gradient: sum_k (w_k / Z) * grad(C_k)
    probs = weights / Z
    
    grad = np.zeros(N)
    for k in range(N):
        if probs[k] < 1e-12:
            continue
        pk = probs[k]
        if k == 0:
            # grad C_0: d/dh_i [h_i - h_i^2] = 1 - 2*h_i
            grad += pk * (1.0 - 2.0 * h)
        else:
            # grad C_k: d/dh_j of sum_{i=0}^{N-k-1} h_i(1-h_{i+k})
            # h_j appears in term i=j as h_j, and in term i=j-k as h_{i+k}
            # Contribution from h_j as h_i: (1 - h_{j+k}) for j in [0, N-k-1]
            # Contribution from h_j as h_{i+k}: -h_{j-k} for j in [k, N-1]
            g_k = np.zeros(N)
            if N - k > 0:
                g_k[:N-k] = 1.0 - h[k:]
            if k < N:
                g_k[k:] -= h[:N-k]
            grad += pk * g_k
    
    return obj, grad


def _project_to_simplex(h):
    """
    Project h onto the set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses iterative water-filling.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    for _ in range(200):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            # Need to add mass; add to elements with room
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            # Need to remove mass; remove from elements above 0
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
    
    h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    
    return h


def _lbfgs_step(h, grad, history):
    """
    Simple L-BFGS step computation.
    history: list of (s, y) pairs where s = h_new - h_old, y = grad_new - grad_old
    """
    m = min(20, len(history))
    if m == 0:
        return -grad
    
    s_list = [history[i][0] for i in range(len(history)-m, len(history))]
    y_list = [history[i][1] for i in range(len(history)-m, len(history))]
    
    # L-BFGS two-loop recursion
    q = grad.copy()
    rho = [1.0 / np.dot(s_list[i], y_list[i]) if np.dot(s_list[i], y_list[i]) != 0 else 0.0 
           for i in range(m)]
    alpha = np.zeros(m)
    
    # First loop (backward)
    for i in range(m-1, -1, -1):
        alpha[i] = rho[i] * np.dot(s_list[i], q)
        q -= alpha[i] * y_list[i]
    
    # Initial Hessian approximation
    if m > 0:
        gamma = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-12)
    else:
        gamma = 1.0
    
    r = gamma * q
    
    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * np.dot(y_list[i], r)
        r += (alpha[i] - beta) * s_list[i]
    
    return -r


def _line_search(h, grad, direction, obj_fn, max_iter=50):
    """
    Backtracking line search.
    """
    alpha = 1.0
    c1 = 1e-4
    f0 = obj_fn(h)
    g0 = np.dot(grad, direction)
    
    for _ in range(max_iter):
        h_new = _project_to_simplex(h + alpha * direction)
        f_new = obj_fn(h_new)
        
        if f_new <= f0 + c1 * alpha * g0:
            return h_new, alpha
        
        alpha *= 0.5
    
    return h, alpha


def construct():
    t0 = time.time()
    
    # Multiple initializations
    best_h = None
    best_true_val = float('inf')
    
    def true_max_overlap(h):
        C = _compute_overlaps(h)
        return np.max(C)
    
    inits = [
        np.full(N, 0.5),
        np.sin(np.arange(N) * 2 * np.pi / 10) * 0.4 + 0.5,
        np.sin(np.arange(N) * 2 * np.pi / 7) * 0.4 + 0.5,
        np.random.uniform(0, 1, N),
        np.concatenate([np.zeros(50), np.ones(50)]),
        np.concatenate([np.ones(50), np.zeros(50)]),
        np.tile([0.0, 1.0], N//2),
        np.tile([1.0, 0.0], N//2),
    ]
    
    for h0 in inits:
        h0 = _project_to_simplex(h0)
        val = true_max_overlap(h0)
        if val < best_true_val:
            best_true_val = val
            best_h = h0.copy()
        
        if time.time() - t0 > DEADLINE - 30:
            break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_true_val = true_max_overlap(best_h)
    
    # Optimize from best initialization
    h = best_h.copy()
    history = []
    
    # Anneal tau from large to small
    tau_values = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    
    for tau in tau_values:
        if time.time() - t0 > DEADLINE - 5:
            break
        
        iters = 0
        max_iters = 200
        stall_count = 0
        last_obj = float('inf')
        
        while iters < max_iters and time.time() - t0 < DEADLINE - 5:
            obj, grad = _soft_min_obj_and_grad(h, tau)
            
            # Check for convergence
            if abs(last_obj - obj) < 1e-8:
                stall_count += 1
            else:
                stall_count = 0
            last_obj = obj
            
            if stall_count > 20:
                break
            
            # L-BFGS direction
            direction = _lbfgs_step(h, grad, history)
            
            # Ensure we're moving in a descent direction
            if np.dot(grad, direction) > 0:
                direction = -grad
            
            # Line search
            h_new, alpha = _line_search(h, grad, direction, 
                                        lambda x: _soft_min_obj_and_grad(x, tau)[0])
            
            # Update history
            s = h_new - h
            if np.linalg.norm(s) > 1e-10:
                obj_new, grad_new = _soft_min_obj_and_grad(h_new, tau)
                y = grad_new - grad
                if np.dot(s, y) > 1e-10:
                    history.append((s, y))
                    if len(history) > 20:
                        history.pop(0)
            
            h = h_new
            iters += 1
        
        # Update best based on true objective
        true_val = true_max_overlap(h)
        if true_val < best_true_val:
            best_true_val = true_val
            best_h = h.copy()
    
    # Final projection
    final_h = _project_to_simplex(best_h)
    
    return final_h
