"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def construct():
    t_start = time.time()
    DEADLINE = 145.0  # Safety margin

    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0

    def subgradient_step(h, eta):
        s = np.sum(h)
        if s == 0:
            return h, np.inf
        
        conv = np.convolve(h, h)
        k_star = np.argmax(conv)
        M = conv[k_star]
        
        # Gradient of M = conv[k_star] w.r.t h
        # grad_M[j] = 2 * h[k_star - j] if 0 <= k_star - j < N else 0
        grad_M = np.zeros(N)
        
        # Valid j range: max(0, k_star - (N - 1)) to min(N - 1, k_star)
        j_min = max(0, k_star - (N - 1))
        j_max = min(N - 1, k_star)
        
        if j_min <= j_max:
            # h indices: k_star - j for j in [j_min, j_max]
            # This is h[k_star - j_max] to h[k_star - j_min]
            # We need to assign to grad_M[j_min:j_max+1]
            # grad_M[j_min] = 2 h[k_star - j_min]
            # grad_M[j_max] = 2 h[k_star - j_max]
            idx_h = np.arange(k_star - j_max, k_star - j_min + 1)
            # h[idx_h] is h[k_star - j_max] ... h[k_star - j_min]
            # We need h[k_star - j_min] ... h[k_star - j_max]
            # So we reverse h[idx_h]
            grad_M[j_min:j_max+1] = 2.0 * h[idx_h][::-1]
            
        # Gradient of V = A * M / S^2
        # dV = A * (dM * S^-2 + M * d(S^-2))
        # d(S^-2) = -2 S^-3 * dS = -2 S^-3 * 1
        A = 2.0 * N
        grad_V = A * (grad_M / (s ** 2) - 2.0 * M / (s ** 3) * np.ones(N))
        
        # Step
        h_new = h - eta * grad_V
        
        # Normalize to sum N to keep scale fixed and avoid drift
        sum_new = np.sum(h_new)
        if sum_new != 0:
            h_new = h_new * (N / sum_new)
            
        val = compute_value(h_new)
        return h_new, val

    # Try multiple initializations
    initializers = [
        np.ones(N),
        np.random.choice([-1, 1], N),
        np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)),
        np.random.normal(0, 1, N)
    ]
    
    for init in initializers:
        if time.time() > DEADLINE - 10.0:
            break
            
        h = init.copy()
        # Normalize initial
        s = np.sum(h)
        if s == 0:
            h = np.ones(N)
            s = N
        else:
            h = h * (N / s)
            
        val = compute_value(h)
        if val < best_val:
            best_h = h.copy()
            best_val = val
            
        # Adaptive learning rate
        eta = 1.0
        no_improve_count = 0
        max_no_improve = 50
        
        for step in range(500):
            if time.time() > DEADLINE - 5.0:
                break
                
            h_new, val_new = subgradient_step(h, eta)
            
            if val_new < val:
                h = h_new
                val = val_new
                no_improve_count = 0
                eta = eta * 0.99
            else:
                eta = eta * 0.7
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break
                
        # Add some random perturbations and try again
        for _ in range(5):
            if time.time() > DEADLINE - 5.0:
                break
            h_pert = h + np.random.normal(0, 0.1, N)
            s_pert = np.sum(h_pert)
            if s_pert == 0:
                continue
            h_pert = h_pert * (N / s_pert)
            val_pert = compute_value(h_pert)
            if val_pert < best_val:
                best_h = h_pert.copy()
                best_val = val_pert

    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
