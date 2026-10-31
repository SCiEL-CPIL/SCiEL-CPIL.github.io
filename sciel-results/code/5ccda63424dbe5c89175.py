"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 115  # seconds, safety margin before 150s kill


def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts.
    overlap_k = sum_i h_i (1 - h_{i+k}) = sum(h) - sum_i h_i h_{i+k} = N/2 - circ_corr(h,h)[k]
    So minimizing max overlap = maximizing min circ_corr.
    """
    n = len(h)
    # circular autocorrelation via FFT
    F = np.fft.rfft(h, n=2 * n)
    corr = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    # corr[k] = sum_i h_i h_{i+k mod n}
    max_overlap = n / 2.0 - np.min(corr)
    return max_overlap


def _all_overlaps(h):
    n = len(h)
    F = np.fft.rfft(h, n=2 * n)
    corr = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    return n / 2.0 - corr


def _project_to_constraint(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2 exactly."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-14:
            if diff2 > 0:
                idx = np.argmax(1.0 - h)
                add = min(diff2, 1.0 - h[idx])
                h[idx] += add
            else:
                idx = np.argmin(h)
                sub = min(-diff2, h[idx])
                h[idx] -= sub
    return h


def _coordinate_descent(h, max_iter=500, tol=1e-10):
    """Local refinement via coordinate descent with projection."""
    h = h.copy()
    for _ in range(max_iter):
        if time.time() - _start_time > _DEADLINE:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try moving h[i] to optimize the max overlap
            # The overlap at shift k is N/2 - corr[k], where corr[k] includes h[i]*h[(i+k)%N]
            # We want to minimize max overlap = maximize min corr
            # For coordinate i, changing h[i] affects corr[k] for all k
            # Simple approach: try a few candidate values for h[i]
            
            # Compute current max overlap
            ov = _all_overlaps(h)
            max_ov = np.max(ov)
            
            # Try perturbing h[i]
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                new_val = old_val + delta
                if 0.0 <= new_val <= 1.0:
                    h[i] = new_val
                    # Adjust others to maintain sum
                    diff = N / 2.0 - np.sum(h)
                    if abs(diff) > 1e-14:
                        # Distribute correction to other elements
                        others = np.arange(N)
                        others = others[others != i]
                        # Add correction to element with most room
                        if diff > 0:
                            room = 1.0 - h[others]
                            idx_add = others[np.argmax(room)]
                            h[idx_add] += min(diff, room.max())
                        else:
                            room = h[others]
                            idx_sub = others[np.argmin(room)]
                            h[idx_sub] += max(diff, -room.min())
                        h = np.clip(h, 0.0, 1.0)
                    
                    new_ov = _all_overlaps(h)
                    new_max_ov = np.max(new_ov)
                    
                    if new_max_ov < max_ov - 1e-12:
                        max_ov = new_max_ov
                        improved = True
                    else:
                        h[i] = old_val
                        break
        
        if not improved:
            break
    
    return h


def _simulated_annealing(h_init, n_iter=2000, T0=0.5, T_end=0.01):
    """Simulated annealing to explore the solution space."""
    h = h_init.copy()
    current_score = _compute_max_overlap(h)
    best_h = h.copy()
    best_score = current_score
    
    # Temperature schedule
    cooling = (T_end / T0) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        if time.time() - _start_time > _DEADLINE:
            break
        
        T = T0 * (cooling ** it)
        
        # Propose a move: perturb a random subset of coordinates
        n_perturb = np.random.randint(1, max(2, N // 10))
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        h_new = h.copy()
        for idx in indices:
            h_new[idx] += np.random.normal(0, 0.1)
        
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _project_to_constraint(h_new)
        
        new_score = _compute_max_overlap(h_new)
        
        # Accept if better, or with probability exp(-(new_score - current_score)/T)
        if new_score < current_score:
            h = h_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                h = h_new
                current_score = new_score
    
    return best_h, best_score


def _spectral_init():
    """Initialize using the eigenvector of the circulant matrix."""
    # The overlap minimization is related to maximizing min eigenvalue of circulant
    # Use frequency domain: h with flat power spectrum minimizes autocorrelation variance
    # Start with a pattern that has low autocorrelation
    
    # Try a few structured initializations
    inits = []
    
    # Uniform
    inits.append(np.full(N, 0.5))
    
    # Alternating (maximizes distance)
    alt = np.where(np.arange(N) % 2 == 0, 1.0, 0.0)
    inits.append(alt)
    
    # Pseudo-random with low autocorrelation (m-sequence like)
    np.random.seed(42)
    for _ in range(5):
        r = np.random.uniform(0, 1, N)
        inits.append(r)
    
    # Sine waves at various frequencies
    for freq in [1, 2, 3, 5, 7, 11]:
        s = 0.5 + 0.5 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        inits.append(s)
    
    # Block patterns
    for block_size in [N // 4, N // 3, N // 2]:
        blk = np.zeros(N)
        blk[:block_size] = 1.0
        inits.append(blk)
    
    # Project all and score
    best_init = None
    best_init_score = np.inf
    for h0 in inits:
        h0 = _project_to_constraint(h0)
        s = _compute_max_overlap(h0)
        if s < best_init_score:
            best_init_score = s
            best_init = h0.copy()
    
    return best_init


def construct():
    global _start_time
    
    # Get a good initial solution
    h_init = _spectral_init()
    best_h = h_init.copy()
    best_score = _compute_max_overlap(best_h)
    
    # First pass: coordinate descent from initial
    h_cd = _coordinate_descent(h_init, max_iter=200)
    score_cd = _compute_max_overlap(h_cd)
    if score_cd < best_score:
        best_score = score_cd
        best_h = h_cd.copy()
    
    # Simulated annealing from multiple starts
    np.random.seed(42)
    
    n_sa_runs = 10
    for run in range(n_sa_runs):
        if time.time() - _start_time > _DEADLINE:
            break
        
        # Create diverse starting points
        if run == 0:
            h_start = best_h.copy()
        elif run == 1:
            h_start = _spectral_init().copy()
        else:
            base = best_h if np.random.random() < 0.5 else _spectral_init()
            h_start = base + np.random.normal(0, 0.15, N)
            h_start = np.clip(h_start, 0, 1)
            h_start = _project_to_constraint(h_start)
        
        # Run simulated annealing
        h_sa, score_sa = _simulated_annealing(h_start, n_iter=1500, T0=0.3, T_end=0.005)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
        
        # Local refinement via coordinate descent
        h_ref = _coordinate_descent(h_sa, max_iter=300)
        score_ref = _compute_max_overlap(h_ref)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
        
        # Additional refinement: try small perturbations of best
        if time.time() - _start_time < _DEADLINE:
            for _ in range(5):
                h_pert = best_h + np.random.normal(0, 0.03, N)
                h_pert = _project_to_constraint(h_pert)
                h_pert = _coordinate_descent(h_pert, max_iter=100)
                s_pert = _compute_max_overlap(h_pert)
                if s_pert < best_score:
                    best_score = s_pert
                    best_h = h_pert.copy()
    
    # Final coordinate descent polish
    if time.time() - _start_time < _DEADLINE:
        h_final = _coordinate_descent(best_h, max_iter=500)
        score_final = _compute_max_overlap(h_final)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            idx = np.argmin(best_h) if diff2 > 0 else np.argmax(best_h)
            best_h[idx] += diff2
    
    return best_h
# EVOLVE-BLOCK-END
