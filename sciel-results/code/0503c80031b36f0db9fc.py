# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        # P[X1+X2+X3 < 2*X4] where all are s[0]
        # s+s+s < 2*s => 3s < 2s => s < 0. Since s>=0, this is 0 unless s<0
        # With strict inequality and s>=0: 3s < 2s means s < 0, so P=0
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)


def _compute_P_with_new_atom(s, w, new_s, new_w):
    """Compute P when adding a new atom at position new_s with weight new_w,
    renormalizing all weights."""
    s_new = np.concatenate([s, [new_s]])
    w_new = np.concatenate([w, [new_w]])
    total = np.sum(w_new)
    if total <= 1e-9:
        return 0.0
    w_norm = w_new / total
    return _compute_P_fast(s_new, w_norm)


def _coordinate_ascent(s, w, max_iters=200, tol=1e-10, time_limit=30.0, t_start=None):
    """
    Coordinate ascent on support positions and weights.
    For each atom, try moving it to candidate positions and redistributing weight.
    """
    if t_start is None:
        t_start = time.time()
    
    n = len(s)
    if n < 2:
        return s, w, _compute_P_fast(s, w)
    
    current_P = _compute_P_fast(s, w)
    w_sum = np.sum(w)
    w = w / w_sum
    
    for iteration in range(max_iters):
        if time.time() > t_start + time_limit:
            break
        
        improved = False
        
        # --- Update support positions ---
        for i in range(n):
            # Try moving atom i to several candidate positions
            # Candidates: current position +/- small steps, and positions relative to other atoms
            candidates = [s[i]]
            # Relative positions to other atoms
            for j in range(n):
                if j != i:
                    candidates.append(s[j] * 0.5)
                    candidates.append(s[j] * 0.25)
                    candidates.append(s[j] * 2.0)
                    candidates.append(s[j] + 0.1)
                    candidates.append(max(0.0, s[j] - 0.1))
            # Small perturbations
            step = max(0.01, (s.max() - s.min()) * 0.05 if n > 1 else 0.01)
            for frac in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0]:
                candidates.append(s[i] * frac if s[i] > 0 else step * frac)
                candidates.append(max(0.0, s[i] - step * frac))
                candidates.append(s[i] + step * frac)
            
            best_pos = s[i]
            best_P_i = current_P
            
            for c in candidates:
                c = max(0.0, c)
                if abs(c - s[i]) < 1e-12:
                    continue
                s_trial = s.copy()
                s_trial[i] = c
                P_trial = _compute_P_fast(s_trial, w)
                if P_trial > best_P_i + 1e-12:
                    best_P_i = P_trial
                    best_pos = c
            
            if best_pos != s[i]:
                s[i] = best_pos
                current_P = best_P_i
                improved = True
        
        # --- Update weights ---
        # For each weight, try redistributing: move some mass from one atom to another
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Try moving delta mass from i to j
                for delta_frac in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                    delta = w[i] * delta_frac
                    if delta >= w[i] * 0.9:
                        continue
                    w_trial = w.copy()
                    w_trial[i] -= delta
                    w_trial[j] += delta
                    if w_trial[i] < 0 or w_trial[j] < 0:
                        continue
                    P_trial = _compute_P_fast(s, w_trial)
                    if P_trial > current_P + 1e-12:
                        current_P = P_trial
                        w = w_trial
                        improved = True
        
        if not improved:
            break
    
    return s, w, current_P


def _try_add_atom(s, w, t_start, time_limit):
    """Try adding a new atom at various candidate positions. Return best if improved."""
    n = len(s)
    current_P = _compute_P_fast(s, w)
    
    # Candidate positions for new atom
    candidates = []
    if n == 0:
        candidates = [0.0, 1.0]
    else:
        s_min = s.min()
        s_max = s.max()
        span = max(1e-6, s_max - s_min)
        # Midpoints between existing atoms
        for i in range(n - 1):
            candidates.append((s[i] + s[i+1]) / 2.0)
        # Extensions
        candidates.append(max(0.0, s_min - span * 0.5))
        candidates.append(s_max + span * 0.5)
        candidates.append(s_max + span)
        # Geometric positions
        if s_max > 0:
            candidates.append(s_max * 0.1)
            candidates.append(s_max * 0.01)
        # Uniform grid points
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            candidates.append(s_min + frac * span)
    
    best_P = current_P
    best_s = s.copy()
    best_w = w.copy()
    
    for c in candidates:
        c = max(0.0, c)
        # Check if too close to existing atom
        if np.any(np.abs(s - c) < 1e-6):
            continue
        
        # Try with small initial weight
        for init_w_frac in [0.05, 0.1, 0.2]:
            if time.time() > t_start + time_limit:
                break
            new_w = w.copy()
            new_s = s.copy()
            # Give the new atom some weight, take from existing
            total_w = np.sum(w)
            new_w_val = total_w * init_w_frac
            # Distribute the reduction proportionally
            w_trial = w * (1.0 - init_w_frac)
            w_trial = np.concatenate([w_trial, [new_w_val]])
            s_trial = np.concatenate([new_s, [c]])
            
            P_trial = _compute_P_fast(s_trial, w_trial)
            if P_trial > best_P + 1e-10:
                best_P = P_trial
                best_s = s_trial.copy()
                best_w = w_trial.copy()
    
    return best_s, best_w, best_P


def _try_remove_atom(s, w, t_start, time_limit):
    """Try removing each atom and see if P improves or stays similar (simpler model)."""
    n = len(s)
    if n <= 2:
        return s, w, _compute_P_fast(s, w)
    
    current_P = _compute_P_fast(s, w)
    best_P = current_P
    best_s = s.copy()
    best_w = w.copy()
    
    for i in range(n):
        if time.time() > t_start + time_limit:
            break
        # Remove atom i, renormalize
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        s_trial = s[mask]
        w_trial = w[mask]
        P_trial = _compute_P_fast(s_trial, w_trial)
        # Only remove if it doesn't hurt much (within 1% of current)
        if P_trial > best_P - 0.001 * current_P and P_trial > best_P:
            best_P = P_trial
            best_s = s_trial.copy()
            best_w = w_trial.copy()
    
    return best_s, best_w, best_P


def _simulated_annealing_search(t_start, total_time_budget):
    """
    Main search: start with a small measure, iteratively add/remove/move atoms
    with simulated annealing acceptance.
    """
    deadline = t_start + total_time_budget
    
    # Initialize with a good 2-point solution
    s = np.array([0.0, 1.0])
    w = np.array([0.618, 0.382])
    current_P = _compute_P_fast(s, w)
    
    best_s = s.copy()
    best_w = w.copy()
    best_P = current_P
    
    # Temperature for simulated annealing
    T = 0.01
    T_min = 1e-8
    cooling_rate = 0.95
    
    np.random.seed(123)
    
    iteration = 0
    while time.time() < deadline - 30:
        iteration += 1
        
        # Decide operation: add, remove, or move
        n = len(s)
        if n < 5 and np.random.random() < 0.3:
            # Try adding an atom
            s_new, w_new, P_new = _try_add_atom(s, w, time.time(), 5.0)
        elif n > 2 and np.random.random() < 0.2:
            # Try removing an atom
            s_new, w_new, P_new = _try_remove_atom(s, w, time.time(), 3.0)
        else:
            # Coordinate ascent to refine current
            s_new, w_new, P_new = _coordinate_ascent(s, w, max_iters=50, time_limit=10.0, t_start=time.time())
        
        if P_new > current_P:
            # Accept
            s = s_new
            w = w_new
            current_P = P_new
            T *= cooling_rate
            if current_P > best_P:
                best_P = current_P
                best_s = s.copy()
                best_w = w.copy()
        elif np.random.random() < np.exp((P_new - current_P) / max(T, 1e-12)):
            # Accept with probability (simulated annealing)
            s = s_new
            w = w_new
            current_P = P_new
            T *= cooling_rate
        else:
            T *= cooling_rate
        
        # Occasionally try a random perturbation of the whole solution
        if iteration % 10 == 0 and time.time() < deadline - 60:
            # Random restart with slight perturbation of best
            s_trial = best_s + np.random.normal(0, 0.05, len(best_s))
            s_trial = np.maximum(s_trial, 0.0)
            w_trial = best_w + np.random.normal(0, 0.02, len(best_w))
            w_trial = np.maximum(w_trial, 1e-4)
            w_trial = w_trial / np.sum(w_trial)
            
            s_trial, w_trial, P_trial = _coordinate_ascent(s_trial, w_trial, max_iters=30, time_limit=8.0, t_start=time.time())
            if P_trial > best_P:
                best_P = P_trial
                best_s = s_trial.copy()
                best_w = w_trial.copy()
                s = s_trial
                w = w_trial
                current_P = P_trial
    
    # Final polish on best
    if time.time() < deadline - 15:
        best_s, best_w, best_P = _coordinate_ascent(best_s, best_w, max_iters=100, time_limit=15.0, t_start=time.time())
    
    return best_s, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Adaptive atom count with coordinate ascent + simulated annealing.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    
    # Fallback baseline
    s_base = np.array([0.0, 1.0])
    w_base = np.array([0.618, 0.382])
    P_base = _compute_P_fast(s_base, w_base)
    
    # Run the main adaptive search
    best_s, best_w, best_P = _simulated_annealing_search(t_start, 540.0)
    
    # If the search didn't improve over baseline, use baseline
    if best_P < P_base:
        best_s = s_base
        best_w = w_base
        best_P = P_base
    
    # Final validation
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return s_base, w_base
    
    if len(best_s) != len(best_w):
        return s_base, w_base
    
    # Ensure non-negative
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return s_base, w_base
    
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return s_base, w_base
    
    # Sort by support for consistency
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
