import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _ac_fft(x):
    """Full linear autoconvolution via FFT (real)."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    ac = np.fft.irfft(X * X, fsize)
    return ac[:size]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    conv = np.fft.irfft(X * X, fsize)
    conv = conv[:size]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _peak_subgradient(h, k):
    """
    Compute the subgradient of |ac_k| with respect to h.
    ac_k = sum_{i=0}^{N-1-k} h_i h_{i+k}
    d(ac_k)/dh_i = h_{i+k} for i < N-k
    d(ac_k)/dh_i = h_{i-k} for i >= k (from the symmetric term in full conv)
    
    For linear autoconvolution ac = h * h:
    ac_k = sum_{i} h_i h_{i+k}
    If we consider the full vector of autoconvolution coefficients, the gradient of ac_k w.r.t h is a vector g where:
    g_i = h_{i+k} if i+k < N
    g_i = h_{i-k} if i-k >= 0
    Note: For k=0, g_i = 2*h_i.
    
    Since we want to minimize |ac_k|, the subgradient is sign(ac_k) * g.
    """
    grad = np.zeros(N)
    sign_k = np.sign(_ac_fft(h)[k]) # We pass h, but we need ac_k. 
    # Actually, computing ac_k every time is expensive if done inside a loop. 
    # Better to pass ac_k or compute it once.
    # Let's refactor: _peak_suppression_step(h, ac, k)
    pass

def _waterfill_v2(h_init, deadline):
    """
    Structured Water-Filling / Peak Suppression.
    1. Start with binary h.
    2. Maintain a continuous 'soft' variable u.
    3. Iteratively:
       a. Compute autoconvolution ac of current binary h (or u?).
          Using u allows smoother gradients. But the objective is defined on binary h?
          The task says "sequence of signed step heights". Usually implies discrete.
          However, "signed equal-width step functions" might allow continuous heights?
          "minimizing the maximum absolute autoconvolution divided by squared integral".
          If heights are continuous, we can scale them. But the ratio is scale invariant.
          So we can assume h in [-1, 1] or similar.
          The current program forces binary {-1, 1}.
          Let's stick to binary {-1, 1} for the final answer, but use a soft variable for optimization.
    """
    h = np.sign(h_init).astype(float)
    h[h == 0] = 1.0
    u = h.copy() # Soft variable
    best_h = h.copy()
    best_val = compute_raw_value(h)
    
    # Precompute FFT size
    fsize = 1
    while fsize < 2 * N - 1:
        fsize *= 2
        
    step_size = 0.5
    
    for it in range(500):
        if time.time() > deadline:
            break
            
        # Evaluate current binary solution
        h_bin = np.sign(u)
        h_bin[h_bin == 0] = 1.0
        val = compute_raw_value(h_bin)
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        # Compute autoconvolution of the SOFT variable u to get smooth gradients
        # This is a surrogate: we minimize max |ac(u)|, then project to binary.
        # Or minimize max |ac(h_bin)|? The latter is non-differentiable.
        # Let's use the soft autoconvolution.
        U = np.fft.rfft(u, fsize)
        ac_soft = np.fft.irfft(U * U, fsize)
        ac_soft = ac_soft[:2*N-1]
        
        abs_ac = np.abs(ac_soft)
        k_max = np.argmax(abs_ac)
        
        if abs_ac[k_max] < 1e-9:
            break
            
        # Compute subgradient of |ac_{k_max}| w.r.t u
        # ac_k = sum_{i=0}^{N-1-k} u_i u_{i+k}
        # grad[i] = sign(ac_k) * (u_{i+k} if i < N-k else 0) + sign(ac_k) * (u_{i-k} if i >= k else 0)
        # Note: For k=0, both terms are u_i, so grad[i] = 2 * sign(ac_0) * u_i
        
        sign_k = np.sign(ac_soft[k_max])
        grad = np.zeros(N)
        
        if k_max == 0:
            grad[:] = 2.0 * sign_k * u
        else:
            # Positive lag contribution: ac_k involves u_i * u_{i+k} for i=0..N-1-k
            # d/d u_i = u_{i+k}
            grad[:N-k_max] += sign_k * u[k_max:]
            # Negative lag contribution (symmetry of autoconv): ac_{-k} = ac_k
            # d/d u_i = u_{i-k} for i=k..N-1
            grad[k_max:] += sign_k * u[:N-k_max]
            
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        # Update soft variable
        # We want to reduce the peak, so we move in the negative gradient direction
        # u_new = u - step_size * grad
        # But we also want to keep u close to binary. Add a regularization?
        # Or just project after update.
        u = u - step_size * grad
        
        # Adaptive step size: if val increases, reduce step
        # (Simplified: just use fixed step for now, or check improvement)
        
        # Optional: If we are stuck, increase step size slightly?
        # For now, keep it simple.
        
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    # Chirp sequences
    t = np.arange(N)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(15):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Structured Water-Filling / Peak Suppression
    # This is the main structural change.
    # We run water-filling from the best candidates.
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Run from top 5 diverse starts
    top_starts = scored_candidates[:5]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 10:
            break
        
        h_wf, val_wf = _waterfill_v2(h_init, deadline - 5)
        if val_wf < best_val:
            best_val = val_wf
            best_x = h_wf.copy()
            
    # 5. Final Polish: Simple local search on the best result
    if time.time() < deadline - 5:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        best_val = compute_raw_value(h_work)
        
        # Single-bit flip local search (fast, few passes)
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                h_work[i] = -h_work[i]
                new_val = compute_raw_value(h_work)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h_work[i] = -h_work[i]  # revert
            if not improved:
                break
        
        best_x = h_work.copy()
        
    return best_x

if __name__ == "__main__":
    # Quick self-test
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
