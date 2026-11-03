import numpy as np
import time

N = 100

def _linear_overlap_values(h):
    """Compute overlap values for all shifts s=1..N-1."""
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        overlaps[s - 1] = np.sum(h[:N - s] * (1.0 - h[s:]))
    return overlaps

def _score(h):
    """Return the maximum overlap (lower is better)."""
    return np.max(_linear_overlap_values(h))

def _project_sum_box(h, target=None):
    """Project h onto {0<=h_i<=1, sum(h)=target} via water-filling."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, 1.0)
    current_sum = h.sum()
    diff = target - current_sum
    if abs(diff) < 1e-14:
        return h
    # Water-filling: adjust values to meet the sum constraint
    # If diff > 0, we need to add mass; add to smallest values first (those with most room)
    # If diff < 0, we need to remove mass; remove from largest values first
    # Simple iterative approach: distribute evenly, clip, repeat
    for _ in range(50):
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        new_diff = target - h.sum()
        if abs(new_diff) < 1e-14:
            break
        # If we can't distribute evenly due to clipping, do targeted adjustment
        if new_diff > 0:
            idx = np.argsort(h)
            remaining = new_diff
            for i in idx:
                if remaining <= 0:
                    break
                room = 1.0 - h[i]
                add = min(room, remaining)
                h[i] += add
                remaining -= add
        else:
            idx = np.argsort(h)[::-1]
            remaining = -new_diff
            for i in idx:
                if remaining <= 0:
                    break
                room = h[i]
                sub = min(room, remaining)
                h[i] -= sub
                remaining -= sub
        diff = target - h.sum()
    return h

def _smooth_max_grad(values, temperature=0.1):
    """Gradient of smooth max (softmax weights)."""
    max_val = np.max(values)
    exp_vals = np.exp((values - max_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def _pgd_optimize(h, temperature=0.1, lr=0.01, n_steps=5000):
    """Projected Gradient Descent with smooth max objective."""
    h = np.array(h, dtype=float)
    best_h = h.copy()
    best_score = _score(h)
    
    for step in range(n_steps):
        overlaps = _linear_overlap_values(h)
        weights = _smooth_max_grad(overlaps, temperature)
        
        # Gradient of smooth max w.r.t. h
        grad_h = np.zeros(N)
        for s in range(1, N):
            idx = s - 1
            w = weights[idx]
            # overlap_s = sum_{i=0}^{N-s-1} h[i]*(1-h[i+s])
            # d/dh[i] for i < N-s: (1-h[i+s])
            # d/dh[i] for i >= s: -h[i-s]
            grad_h[:N - s] += w * (1.0 - h[s:])
            grad_h[s:] += w * (-h[:N - s])
        
        # Gradient descent step
        h = h - lr * grad_h
        h = _project_sum_box(h)
        
        # Check score
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Adaptive learning rate: decrease if not improving
        if step % 500 == 499 and step > 0:
            lr *= 0.7
    
    return best_h, best_score

def construct():
    deadline = time.time() + 140.0
    np.random.seed(42)
    
    # Strategy: Model h as convex combination of two basis patterns
    # Basis 1: Thue-Morse-like alternating pattern (balanced)
    # Basis 2: High-contrast pattern with different period
    
    best_h = None
    best_score = float('inf')
    
    # Generate diverse basis patterns
    bases = []
    
    # Thue-Morse
    h_tm = np.zeros(N)
    for i in range(N):
        cnt = bin(i).count('1')
        h_tm[i] = 1.0 if cnt % 2 == 1 else 0.0
    bases.append(h_tm)
    
    # Periodic patterns with different periods
    for period in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        h_pat = np.zeros(N)
        for i in range(N):
            # Alternating pattern with this period
            h_pat[i] = 1.0 if (i % (2 * period)) < period else 0.0
        bases.append(h_pat)
    
    # Random bases
    for _ in range(20):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand)
        bases.append(h_rand)
    
    # For each pair of bases, optimize mixing weight and global scaling
    for i in range(len(bases)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(bases)):
            if time.time() > deadline:
                break
            
            b1 = bases[i]
            b2 = bases[j]
            
            # Optimize over mixing weight alpha in [0, 1]
            # h = alpha * b1 + (1 - alpha) * b2, then project to sum = N/2
            best_alpha_score = float('inf')
            best_alpha_h = None
            
            # Coarse search over alpha
            for alpha in np.linspace(0, 1, 21):
                h_mix = alpha * b1 + (1 - alpha) * b2
                h_mix = _project_sum_box(h_mix)
                score = _score(h_mix)
                if score < best_alpha_score:
                    best_alpha_score = score
                    best_alpha_h = h_mix.copy()
            
            if best_alpha_h is None:
                continue
            
            # Fine search around best alpha
            if best_alpha_score < best_score:
                # Refine with PGD
                h_refined, score_refined = _pgd_optimize(
                    best_alpha_h, 
                    temperature=0.05, 
                    lr=0.05, 
                    n_steps=2000
                )
                if score_refined < best_score:
                    best_score = score_refined
                    best_h = h_refined.copy()
    
    # If we found something good, do additional polishing
    if best_h is not None and time.time() < deadline - 20:
        # Multi-temperature polishing
        for temp in [0.02, 0.05, 0.1]:
            if time.time() > deadline:
                break
            h_polished, score_polished = _pgd_optimize(
                best_h,
                temperature=temp,
                lr=0.02,
                n_steps=3000
            )
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
    
    # Final random restarts if time permits
    if best_h is None or time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline:
                break
            h_init = np.random.uniform(0, 1, N)
            h_init = _project_sum_box(h_init)
            h_trial, score_trial = _pgd_optimize(
                h_init,
                temperature=0.1,
                lr=0.1,
                n_steps=2000
            )
            if score_trial < best_score:
                best_score = score_trial
                best_h = h_trial.copy()
    
    if best_h is None:
        # Fallback: Thue-Morse
        best_h = np.zeros(N)
        for i in range(N):
            cnt = bin(i).count('1')
            best_h[i] = 1.0 if cnt % 2 == 1 else 0.0
        best_h = _project_sum_box(best_h)
    
    return best_h
