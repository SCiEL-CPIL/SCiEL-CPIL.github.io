import numpy as np
import time

N = 100

def _score(h):
    """Compute the maximum overlap of h with its shifted 1-h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _fix_sum(h, target=None):
    """Project h to have exact sum of N/2 while staying in [0,1]."""
    if target is None:
        target = N / 2.0
    h = np.array(h, dtype=float)
    for _ in range(100):
        diff = target - np.sum(h)
        h += diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        if abs(target - np.sum(h)) < 1e-15:
            break
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 0:
        # Distribute the remaining difference
        h += diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If still not exact, adjust one element
        diff2 = target - np.sum(h)
        if abs(diff2) > 1e-15:
            # Find an element that can absorb the difference
            for i in range(len(h)):
                if diff2 > 0 and h[i] < 1.0 - abs(diff2):
                    h[i] += diff2
                    break
                elif diff2 < 0 and h[i] > abs(diff2):
                    h[i] += diff2
                    break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    x = np.arange(N)
    
    # Start with uniform
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # Strategy 1: Two-level profiles with smooth transition
    # h[i] = v1 + (v2-v1) * sigmoid(B*(i-C))
    # Sum constraint: v1*N + (v2-v1)*sum(sigmoid) = N/2
    
    def make_profile(A, B, C):
        """Create profile h[i] = 0.5 + A * tanh(B*(i-C))"""
        i = np.arange(N)
        h = 0.5 + A * np.tanh(B * (i - C))
        return h
    
    def optimize_profile(B):
        """For fixed B, optimize A and C to satisfy sum constraint and minimize score."""
        # The sum of h depends on A and C
        # We need sum(h) = N/2
        # For tanh profile centered at C, if C = N/2 - 0.5 (center of array), sum is approximately N/2
        # But we need exact. Let's use the fact that for symmetric profiles, C should be near center.
        
        # Actually, let's parameterize differently:
        # h[i] = v_low + (v_high - v_low) * (1 + tanh(B*(i - C))) / 2
        # This gives h in [v_low, v_high]
        # Sum = v_low*N + (v_high-v_low)*sum((1+tanh(B*(i-C)))/2)
        
        # Let's use a simpler approach: binary search on C for given A, B
        # to satisfy the sum constraint, then check score.
        
        best_score = float('inf')
        best_AC = None
        
        # Try a range of A values
        for A in np.linspace(0.05, 0.5, 30):
            # For this A and B, find C such that sum(h) = N/2
            # h[i] = 0.5 + A * tanh(B*(i-C))
            # sum(h) = N/2 + A * sum(tanh(B*(i-C)))
            # We need sum(tanh(B*(i-C))) = 0
            # This happens when C is at the center of symmetry of the tanh sum
            
            # The sum of tanh(B*(i-C)) over i=0..N-1 is an odd function of (C - N/2 + 0.5)
            # So C = (N-1)/2 = 49.5 should give sum = 0
            # Let's verify and adjust if needed
            
            C0 = (N - 1) / 2.0
            h = make_profile(A, B, C0)
            sum_h = np.sum(h)
            
            if abs(sum_h - N/2.0) < 1e-10:
                pass
            else:
                # Binary search for C
                C_lo = 0.0
                C_hi = N - 1.0
                for _ in range(60):
                    C_mid = (C_lo + C_hi) / 2.0
                    h_mid = make_profile(A, B, C_mid)
                    sum_mid = np.sum(h_mid)
                    if sum_mid > N/2.0:
                        C_hi = C_mid
                    else:
                        C_lo = C_mid
                C0 = (C_lo + C_hi) / 2.0
            
            h = make_profile(A, B, C0)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_AC = (A, C0)
        
        return best_AC
    
    # Try different steepness values B
    for B in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        if time.time() > deadline - 10:
            break
        
        best_params = optimize_profile(B)
        if best_params is not None:
            A, C = best_params
            h_candidate = make_profile(A, B, C)
            h_candidate = _fix_sum(h_candidate)
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
    
    # Strategy 2: Two-level step function
    # h[i] = v1 for i < k, v2 for i >= k
    # v1*k + v2*(N-k) = N/2
    
    for k in range(1, N):
        if time.time() > deadline - 5:
            break
        
        # v1*k + v2*(N-k) = N/2
        # v2 = (N/2 - v1*k) / (N-k)
        # Constraints: 0 <= v1 <= 1, 0 <= v2 <= 1
        
        # v2 >= 0 => v1 <= N/(2k)
        # v2 <= 1 => N/2 - v1*k <= N-k => v1*k >= N/2 - N + k => v1 >= (N/2 - N + k)/k = 1 - N/(2k)
        
        v1_min = max(0.0, 1.0 - N/(2.0*k))
        v1_max = min(1.0, N/(2.0*k))
        
        if v1_min >= v1_max:
            continue
        
        # Sample v1 values
        for v1 in np.linspace(v1_min, v1_max, 20):
            v2 = (N/2.0 - v1*k) / (N-k)
            if v2 < -1e-10 or v2 > 1.0 + 1e-10:
                continue
            v2 = np.clip(v2, 0.0, 1.0)
            
            h = np.zeros(N)
            h[:k] = v1
            h[k:] = v2
            h = _fix_sum(h)
            sc = _score(h)
            if sc < score_best:
                score_best = sc
                h_best = h.copy()
    
    # Strategy 3: Fourier-based refinement of the best found so far
    # Use the best h as a starting point and try small Fourier perturbations
    
    if time.time() < deadline - 15:
        from scipy.optimize import minimize
        
        K = 5
        n_params = 2 * K
        x = np.arange(N)
        
        # Initialize from best h
        # Project h_best onto Fourier space
        h0 = np.zeros(n_params)
        for k in range(1, K + 1):
            cos_k = np.cos(2 * np.pi * k * x / N)
            sin_k = np.sin(2 * np.pi * k * x / N)
            h0[2*(k-1)] = np.dot(h_best - 0.5, cos_k) / np.dot(cos_k, cos_k)
            h0[2*(k-1)+1] = np.dot(h_best - 0.5, sin_k) / np.dot(sin_k, sin_k)
        
        def get_h(params):
            h = np.full(N, 0.5)
            for k in range(1, K + 1):
                c_k = params[2*(k-1)]
                s_k = params[2*(k-1)+1]
                h += c_k * np.cos(2 * np.pi * k * x / N)
                h += s_k * np.sin(2 * np.pi * k * x / N)
            return h
        
        def objective(params):
            h = get_h(params)
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            penalty = 50.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            
            return _smooth_max(overlap, alpha=300.0) + penalty
        
        try:
            res = minimize(
                objective,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            
            h_candidate = get_h(res.x)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            h_candidate = _fix_sum(h_candidate)
            
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
                
        except Exception:
            pass
    
    return h_best
