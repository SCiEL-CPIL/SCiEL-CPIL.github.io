import time
import numpy as np

N = 50
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    l2sq = float(np.dot(h, h))
    # linear autoconvolution
    hh = np.convolve(h, h, mode='full')
    sup = float(np.max(np.abs(hh)))
    if sup < 1e-15:
        return 0.0
    return l2sq / (l1 * sup)

def _correlation_peaks(h):
    """Compute the autoconvolution peaks for all lags."""
    hh = np.convolve(h, h, mode='full')
    return hh

def _minmax_update(h, max_iter=2000, deadline=_DEADLINE, tol=1e-10):
    """
    Sinkhorn-Knopp style multiplicative updates for the min-max game.
    
    The defender (h) wants to minimize max_k (h*h)[k] given ||h||_1 = 1,
    while maximizing ||h||_2^2.
    
    We use a game-theoretic formulation:
    - Attacker selects lag k with probability p_k
    - Defender allocates mass h_i
    
    The value function is V = ||h||_2^2 / max_k (h*h)[k]
    
    Multiplicative update: h_i <- h_i * (2*h_i / max_k (h*h)[k])^eta
    where eta is a step size, normalized to keep ||h||_1 = 1.
    """
    h = np.asarray(h, dtype=float).copy()
    if np.sum(h) < 1e-15:
        h = np.ones(N)
    
    # Normalize to unit L1
    h = h / np.sum(h)
    
    current_score = _ratio(h)
    
    for iteration in range(max_iter):
        if _elapsed() > deadline - 5.0:
            break
        
        # Compute autoconvolution
        hh = np.convolve(h, h, mode='full')
        peak = np.max(hh)
        
        if peak < 1e-15:
            break
        
        # The "attack" vector: which lags are most dangerous
        # For each position i, its contribution to the peak correlation
        # is related to how much it contributes to the max lag
        
        # Compute gradient-like signal: for each i, how much does h_i
        # contribute to the peak?
        # (h*h)[k] = sum_i h_i * h_{i+k}
        # d/dh_i [(h*h)[k]] = h_{i+k} + h_{i-k} (for valid indices)
        
        # Find the peak lag
        peak_lag = np.argmax(hh)
        
        # Compute the "pressure" on each coordinate from the peak correlation
        # h_i contributes to (h*h)[peak_lag] as h_i * h_{i + peak_lag - (N-1)}
        # Actually, let's think differently.
        
        # Alternative: use a multiplicative update inspired by the ratio
        # We want to increase h_i where it helps L2^2 more than it hurts the peak
        # L2^2 = sum h_i^2, so increasing h_i by dh adds ~2*h_i*dh to L2^2
        # But it also increases the peak correlation
        
        # Use a simple multiplicative update:
        # h_i <- h_i * (h_i / peak)^eta, then normalize
        # This increases large h_i (good for L2) and decreases small ones
        
        eta = 0.5  # step size
        
        # Compute the update factor
        # We want to boost positions that are "safe" (don't contribute much to peak)
        # and boost the L2 norm
        
        # Method: h_i_new = h_i * (h_i / peak)^eta
        # This is like a power iteration that amplifies large values
        
        update_factor = (h / peak) ** eta
        h_new = h * update_factor
        h_new = np.maximum(h_new, 0)
        
        if np.sum(h_new) < 1e-15:
            break
        
        h_new = h_new / np.sum(h_new)
        
        new_score = _ratio(h_new)
        
        if new_score > current_score + tol:
            h = h_new
            current_score = new_score
        elif iteration > 100:
            # If we haven't improved in a while, try a different step size
            eta = eta * 0.8
            if eta < 0.01:
                break
    
    return h, current_score

def _minmax_game_update(h, max_iter=3000, deadline=_DEADLINE):
    """
    More sophisticated min-max game using explicit attacker/defender updates.
    
    Attacker: maintains a distribution p over lags k=0,...,2N-2
    Defender: maintains h with ||h||_1 = 1, h >= 0
    
    The payoff to the defender is ||h||_2^2 / (sum_k p_k * (h*h)[k])
    
    We use multiplicative weight updates for both players.
    """
    h = np.asarray(h, dtype=float).copy()
    if np.sum(h) < 1e-15:
        h = np.ones(N)
    h = h / np.sum(h)
    
    # Attacker's distribution over lags (2N-1 possible lags)
    num_lags = 2 * N - 1
    p = np.ones(num_lags) / num_lags
    
    current_score = _ratio(h)
    eta_def = 0.1  # defender step size
    eta_att = 0.1  # attacker step size
    
    for iteration in range(max_iter):
        if _elapsed() > deadline - 5.0:
            break
        
        # Compute autoconvolution
        hh = np.convolve(h, h, mode='full')
        expected_peak = float(np.dot(p, hh))
        
        if expected_peak < 1e-15:
            break
        
        l2sq = float(np.dot(h, h))
        payoff = l2sq / expected_peak
        
        if payoff > current_score + 1e-10:
            current_score = payoff
        
        # Defender update: maximize l2sq / expected_peak
        # Gradient w.r.t. h_i: 2*h_i/expected_peak - l2sq * d(expected_peak)/dh_i / expected_peak^2
        # d(expected_peak)/dh_i = sum_k p_k * (h_{i+k-(N-1)} + h_{i-k+(N-1)}) for valid indices
        
        # Simplified multiplicative update for defender:
        # Boost h_i proportional to h_i (to increase L2) but penalize based on correlation
        # h_i <- h_i * (h_i / expected_peak)^eta_def
        
        def_factor = (h / expected_peak) ** eta_def
        h_new = h * def_factor
        h_new = np.maximum(h_new, 0)
        if np.sum(h_new) < 1e-15:
            h_new = h
        else:
            h_new = h_new / np.sum(h_new)
        
        # Attacker update: concentrate on lags with high correlation
        # p_k <- p_k * (hh_k / expected_peak)^eta_att
        att_factor = (hh / expected_peak) ** eta_att
        p_new = p * att_factor
        p_new = np.maximum(p_new, 0)
        if np.sum(p_new) < 1e-15:
            p_new = p
        else:
            p_new = p_new / np.sum(p_new)
        
        # Accept if it improves
        h_candidate = h_new
        p_candidate = p_new
        
        hh_cand = np.convolve(h_candidate, h_candidate, mode='full')
        expected_peak_cand = float(np.dot(p_candidate, hh_cand))
        if expected_peak_cand > 1e-15:
            l2sq_cand = float(np.dot(h_candidate, h_candidate))
            payoff_cand = l2sq_cand / expected_peak_cand
            if payoff_cand > current_score - 1e-8:
                h = h_candidate
                p = p_candidate
        
        # Decay step sizes
        if iteration % 100 == 0 and iteration > 0:
            eta_def *= 0.95
            eta_att *= 0.95
            if eta_def < 0.001:
                break
    
    return h, current_score

def _coordinate_descent(h, max_iter=300, deadline=_DEADLINE):
    """
    Cyclic coordinate descent maximizing the ratio.
    For each coordinate, try multiplicative and additive perturbations.
    """
    h = h.copy()
    current_score = _ratio(h)
    
    # Multiplicative factors to try
    mult_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.7, 2.0, 3.0, 5.0, 10.0]
    
    for iteration in range(max_iter):
        if _elapsed() > deadline - 5.0:
            break
        improved = False
        for i in range(N):
            old = h[i]
            best_local = old
            best_local_score = current_score
            
            # Try multiplicative changes
            for f in mult_factors:
                candidate = old * f
                if candidate < 0:
                    candidate = 0.0
                h[i] = candidate
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = candidate
            
            # Try setting to zero
            if old > 0:
                h[i] = 0.0
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = 0.0
            
            # Try small additive perturbations (for sparse structures)
            for delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
                candidate = old + delta
                h[i] = candidate
                s = _ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = candidate
            
            h[i] = best_local
            if best_local != old:
                improved = True
        
        if not improved:
            break
    
    final_score = _ratio(h)
    return h, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Generate diverse structured initializations
    structures = []
    
    # 1. Single spike (known baseline)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    structures.append(h_spike)
    
    # 2. Two-element patterns
    for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h2 = np.zeros(N)
        h2[0] = a
        h2[1] = 1.0 - a
        structures.append(h2)
    
    # 3. Ramp functions
    for slope in [0.5, 1.0, 2.0, 3.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # 4. Symmetric bumps of various widths
    center = N // 2
    for width in [3, 5, 8, 10, 15, 20, 25]:
        h = np.zeros(N)
        for i in range(max(0, center - width), min(N, center + width + 1)):
            h[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))
        structures.append(h)
    
    # 5. Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        for phase in range(min(period, 5)):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                structures.append(h)
    
    # 6. Random sparse patterns
    rng = np.random.default_rng(42)
    for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for trial in range(5):
            h = np.zeros(N)
            n_nonzero = max(1, int(N * sparsity))
            idx = rng.choice(N, n_nonzero, replace=False)
            h[idx] = rng.uniform(0.1, 1.0, n_nonzero)
            structures.append(h)
    
    # 7. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i + 1) ** (-alpha)
        structures.append(h)
    
    # 8. Two-bump structures
    for pos1 in [5, 10, 15, 20]:
        for pos2 in [30, 35, 40, 45]:
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            structures.append(h)
    
    # 9. Cosine windows
    for freq in [1, 2, 3, 4, 5]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            structures.append(h)
    
    # 10. Random smooth patterns
    for trial in range(10):
        h = rng.random(N)
        # Smooth it
        kernel = np.ones(5) / 5.0
        h_smooth = np.convolve(h, kernel, mode='same')
        h_smooth = np.maximum(h_smooth, 0)
        if np.sum(h_smooth) > 0:
            structures.append(h_smooth)
    
    # 11. New: structured patterns designed to have low autoconvolution peaks
    # Alternating high-low patterns that spread out the correlation
    for base in [1.0, 2.0, 5.0]:
        h = np.zeros(N)
        for i in range(0, N, 2):
            h[i] = base
        for i in range(1, N, 2):
            h[i] = 1.0
        structures.append(h)
    
    # Geometric spacing: positions at 1, 2, 4, 8, 16, 32
    h_geo = np.zeros(N)
    pos = 0
    while pos < N:
        h_geo[pos] = 1.0
        pos = pos * 2 + 1 if pos > 0 else 1
    structures.append(h_geo)
    
    # Evaluate and refine each structure
    for h_init in structures:
        if _elapsed() > deadline - 20.0:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        
        # Quick evaluation
        s = try_candidate(h_init)
        
        # Try min-max game update
        if _elapsed() < deadline - 15.0:
            h_mm, s_mm = _minmax_game_update(h_init, max_iter=1000, deadline=deadline)
            if s_mm > best_score:
                best_score = s_mm
                best_h = h_mm.copy()
        
        # Refine with coordinate descent
        h_refined, s_refined = _coordinate_descent(h_init, max_iter=100, deadline=deadline)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Final intensive refinement on the best found
    if best_h is not None and _elapsed() < deadline - 10.0:
        # Try min-max on the best
        h_mm_final, s_mm_final = _minmax_game_update(best_h, max_iter=2000, deadline=deadline)
        if s_mm_final > best_score:
            best_score = s_mm_final
            best_h = h_mm_final.copy()
        
        # Then coordinate descent polish
        h_final, s_final = _coordinate_descent(best_h, max_iter=500, deadline=deadline)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Ensure we have a valid result
    if best_h is None or np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return [float(v) for v in best_h]
