"""
Erdos minimum overlap: minimize max correlation of h with 1-h.

Approach: Subgradient optimization of the max-correlation objective
with structured initializations and local pair-swap refinement.
"""

import numpy as np
import time

N = 100


def _value(h):
    """Compute the evaluator's value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _project(h):
    """Project h to feasible set: 0<=h<=1, sum(h)=N/2 within tolerance."""
    h = np.clip(h.astype(float), 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = np.sum(h) - target
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            # Need to reduce sum; reduce elements that are > 0
            mask = h > 1e-14
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] -= diff / cnt
        else:
            # Need to increase sum; increase elements that are < 1
            mask = h < 1.0 - 1e-14
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] -= diff / cnt  # diff negative, so adds
        h = np.clip(h, 0.0, 1.0)
    # Final tiny correction
    diff = np.sum(h) - target
    if abs(diff) > 1e-12:
        idx = np.argmax(h) if diff > 0 else np.argmin(h)
        h[idx] -= diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _compute_subgradient(h):
    """
    Compute a subgradient of f(h) = max_lag corr_lag(h) / N * 2.
    
    For a given lag L (which can be negative), the correlation is:
    corr_L = sum_i h[i] * (1 - h[i+L]) for valid i
    
    This is bilinear in h. The gradient w.r.t. h[i] is:
    - For each valid pair (i, i+L): d/dh[i] = (1-h[i+L]), d/dh[i+L] = -h[i]
    
    We return the gradient of the active (max) lag, or a weighted combination
    of all lags near the max for stability.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    max_val = np.max(corr)
    
    # Find all lags that are within epsilon of max (for stability)
    eps = 1e-6 * max(1.0, abs(max_val))
    active_mask = corr >= max_val - eps
    active_lags = np.where(active_mask)[0]
    
    if len(active_lags) == 0:
        active_lags = [np.argmax(corr)]
    
    # Weight active lags by how close they are to max
    weights = np.exp(-(max_val - corr[active_lags]) / (0.1 * max(1.0, abs(max_val))))
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    
    for w, lag_idx in zip(weights, active_lags):
        lag = lag_idx - (N - 1)
        
        if lag >= 0:
            # corr = sum_{i=0}^{N-1-lag} h[i] * (1 - h[i+lag])
            for i in range(N - lag):
                grad[i] += w * (1 - h[i + lag])
                grad[i + lag] += w * (-h[i])
        else:
            L = -lag
            # corr = sum_{i=L}^{N-1} h[i] * (1 - h[i-L])
            for i in range(L, N):
                grad[i] += w * (1 - h[i - L])
                grad[i - L] += w * (-h[i])
    
    return grad


def _subgradient_descent(h, max_iter=500, deadline=None):
    """
    Subgradient descent on the max-correlation objective.
    """
    best_h = h.copy()
    best_val = _value(best_h)
    step_size = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _compute_subgradient(best_h)
        
        # Project gradient to maintain sum constraint
        grad_centered = grad - np.mean(grad)
        
        # Try several step sizes
        improved = False
        for alpha in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.1]:
            new_h = best_h - alpha * grad_centered
            new_h = _project(new_h)
            v = _value(new_h)
            if v < best_val - 1e-14:
                best_h = new_h
                best_val = v
                improved = True
                break
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_h, best_val


def _pair_swap_search(h, max_rounds=2000, deadline=None):
    """
    Local search using pair swaps: pick two indices, transfer delta from one to another.
    """
    best_h = h.copy()
    best_val = _value(best_h)
    rng = np.random.default_rng(12345)
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try random pair swaps
        for _ in range(20):
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j:
                continue
            
            # Try both directions
            for sign in [1, -1]:
                for delta in [0.01, 0.02, 0.05, 0.1, 0.2]:
                    new_h = best_h.copy()
                    new_h[i] += sign * delta
                    new_h[j] -= sign * delta
                    
                    # Check bounds before projecting
                    if new_h[i] < -0.1 or new_h[i] > 1.1 or new_h[j] < -0.1 or new_h[j] > 1.1:
                        continue
                    
                    new_h = _project(new_h)
                    v = _value(new_h)
                    if v < best_val - 1e-14:
                        best_h = new_h
                        best_val = v
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try larger moves
            for _ in range(10):
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                
                delta = rng.uniform(0.1, 0.4) * (1 if rng.random() < 0.5 else -1)
                new_h = best_h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                new_h = _project(new_h)
                v = _value(new_h)
                if v < best_val - 1e-14:
                    best_h = new_h
                    best_val = v
                    improved = True
                    break
        
        if not improved and round_num > 100:
            # Try to reset step size occasionally
            pass
    
    return best_h, best_val


def _generate_patterns():
    """Generate diverse structured patterns for initialization."""
    patterns = []
    
    # 1. Alternating blocks of various sizes
    for block_size in range(2, 15):
        for offset in range(block_size):
            h = np.zeros(N)
            for i in range(N):
                block_idx = (i + offset) // block_size
                h[i] = 1.0 if block_idx % 2 == 0 else 0.0
            patterns.append(h)
    
    # 2. Sine waves
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for phase in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            h = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h = np.clip(h, 0, 1)
            patterns.append(h)
    
    # 3. Triangle waves
    for period in [3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 20]:
        t = np.arange(N) % period / period
        h = 2 * t if t < 0.5 else 2 * (1 - t)
        patterns.append(h)
    
    # 4. Thue-Morse sequence
    def thue_morse(n):
        return bin(n).count('1') % 2
    
    for shift in range(5):
        tm = np.array([thue_morse(i + shift) for i in range(N)])
        h = tm.astype(float)
        patterns.append(h)
    
    # 5. Van der Corput low-discrepancy
    def van_der_corput(n, base=2):
        res = 0.0
        f = 1.0 / base
        while n > 0:
            res += f * (n % base)
            n //= base
            f /= base
        return res
    
    for base in [2, 3, 5]:
        vd = np.array([van_der_corput(i, base) for i in range(N)])
        h = 0.5 + 0.5 * (2 * vd - 1)
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    # 6. Random smooth patterns
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Generate smooth random pattern
        freqs = rng.uniform(0.5, 5, 5)
        phases = rng.uniform(0, 2*np.pi, 5)
        amps = rng.uniform(0.2, 0.5, 5)
        h = 0.5
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h = np.clip(h, 0, 1)
        patterns.append(h)
    
    return patterns


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Start with seed
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Phase 1: Try structured patterns
    patterns = _generate_patterns()
    
    for pat in patterns:
        if time.time() > deadline - 80.0:
            break
        h = _project(pat)
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Phase 2: Subgradient descent on best
    if time.time() < deadline - 60.0:
        h_ref, v_ref = _subgradient_descent(best_h, max_iter=300, deadline=deadline - 50.0)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()
    
    # Phase 3: Pair swap refinement
    if time.time() < deadline - 30.0:
        h_ref, v_ref = _pair_swap_search(best_h, max_rounds=1000, deadline=deadline - 20.0)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()
    
    # Phase 4: Multi-start from diverse patterns
    if time.time() < deadline - 40.0:
        # Score patterns
        scored = []
        for pat in patterns[:60]:
            if time.time() > deadline - 35.0:
                break
            h = _project(pat)
            v = _value(h)
            scored.append((v, h))
        scored.sort(key=lambda x: x[0])
        
        # Refine top patterns
        for v, h in scored[:10]:
            if time.time() > deadline - 15.0:
                break
            
            # Quick subgradient
            if time.time() < deadline - 12.0:
                h_ref, v_ref = _subgradient_descent(h, max_iter=100, deadline=deadline - 10.0)
                if v_ref < best_val:
                    best_val = v_ref
                    best_h = h_ref.copy()
            
            # Quick pair swap
            if time.time() < deadline - 8.0:
                h_ref, v_ref = _pair_swap_search(best_h, max_rounds=300, deadline=deadline - 5.0)
                if v_ref < best_val:
                    best_val = v_ref
                    best_h = h_ref.copy()
    
    # Phase 5: Final polish
    if time.time() < deadline - 5.0:
        h_ref, v_ref = _pair_swap_search(best_h, max_rounds=500, deadline=deadline - 2.0)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()
    
    # Final projection
    best_h = _project(best_h)
    final_val = _value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure sum is exactly N/2
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-10:
        mask = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] -= diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
        else:
            idx = np.argmax(best_h) if diff > 0 else np.argmin(best_h)
            best_h[idx] -= diff
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
