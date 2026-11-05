import numpy as np
import time

N = 400
_DEADLINE = 145.0


def _evaluate(h):
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _construct_zadoff_chu(f, q, M):
    """
    Construct a Zadoff-Chu-like sequence of length M.
    h[n] = exp(2j * pi * f * n * (n + q) / M) for n = 0, ..., M-1
    Take real part for real-valued heights.
    """
    n = np.arange(M)
    phase = 2.0 * np.pi * f * n * (n + q) / M
    return np.real(np.exp(1j * phase))


def _construct_zc_pair(f1, q1, f2, q2, M, amp1=1.0, amp2=1.0):
    """
    Construct h as a sum of two Zadoff-Chu sequences with different parameters.
    """
    h1 = amp1 * _construct_zadoff_chu(f1, q1, M)
    h2 = amp2 * _construct_zadoff_chu(f2, q2, M)
    return h1 + h2


def _local_refine_admm(h_init, time_budget, rng):
    """
    ADMM-inspired local refinement.
    We minimize ||conv(h,h) - C*1||_1 subject to sum(h)=1.
    Using ADMM: split h = z, with constraint sum(z)=1.
    """
    h = h_init.copy()
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) < 1e-10:
        return h, _evaluate(h)
    h = h / s
    
    best_val = _evaluate(h)
    t_start = time.time()
    
    # ADMM parameters
    rho = 0.1  # penalty parameter
    C = np.mean(np.convolve(h, h))  # initial constant estimate
    
    # ADMM variables
    z = h.copy()  # z satisfies constraint
    u = np.zeros(N)  # dual variable
    
    while time.time() - t_start < time_budget:
        # h-update: minimize rho/2 * ||h - z + u||^2 + ||conv(h,h) - C*1||_1
        # This is hard to solve exactly, so use a gradient step on the smooth part
        # and proximal for the l1 part.
        # Approximate: gradient of rho/2 * ||h - z + u||^2 is rho*(h - z + u)
        # For the l1 part, we use a subgradient approximation.
        
        diff = h - z + u
        grad_smooth = rho * diff
        
        # Compute subgradient for l1 term (approximate)
        conv_h = np.convolve(h, h)
        residual = conv_h - C
        # Subgradient of ||residual||_1 with respect to h
        # This requires chain rule: d||conv(h,h)||_1/dh
        # For simplicity, use a random perturbation approach
        step_size = 0.01
        h_new = h - step_size * grad_smooth
        
        # Project to sum=1
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-10:
            h_new = h_new / s_new
        
        v_new = _evaluate(h_new)
        if v_new < best_val:
            best_val = v_new
            h = h_new
            # Update z to project onto constraint set (sum=1)
            z = h.copy()
            u += diff
        else:
            # Try a random perturbation
            idx = rng.integers(0, N)
            h_rand = h.copy()
            h_rand[idx] += rng.normal(0, 0.05)
            s_rand = np.sum(h_rand)
            if abs(s_rand) > 1e-10:
                h_rand = h_rand / s_rand
            v_rand = _evaluate(h_rand)
            if v_rand < best_val:
                best_val = v_rand
                h = h_rand
                z = h.copy()
                u += diff
    
    return h, best_val


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # 2.0
    
    # Phase 1: Search over Zadoff-Chu parameters
    # f and q determine the chirp rate and offset
    # For Zadoff-Chu, we need gcd(f, M) = 1 for good properties
    # M = 400 = 2^4 * 5^2, so f must be coprime with 400
    
    rng = np.random.default_rng(42)
    
    # Try various combinations of f1, q1, f2, q2
    # f must be coprime with N=400
    # Coprime with 400: not divisible by 2 or 5
    # So f in {1, 3, 7, 9, 11, 13, 17, 19, 21, 23, ...}
    
    candidate_f_values = []
    for f in range(1, 401):
        if np.gcd(f, 400) == 1:
            candidate_f_values.append(f)
    
    # Take a subset to limit computation
    # Use smaller f values first
    candidate_f_values = candidate_f_values[:50]
    
    q_values = list(range(0, 10))
    
    evaluated = 0
    for f1 in candidate_f_values[:20]:
        if time.time() > deadline - 30:
            break
        for q1 in q_values[:5]:
            if time.time() > deadline - 30:
                break
            for f2 in candidate_f_values[:10]:
                if time.time() > deadline - 30:
                    break
                if f2 == f1:
                    continue
                for q2 in q_values[:3]:
                    if time.time() > deadline - 30:
                        break
                    
                    h_candidate = _construct_zc_pair(f1, q1, f2, q2, N)
                    v_candidate = _evaluate(h_candidate)
                    evaluated += 1
                    
                    if v_candidate < best_val:
                        best_val = v_candidate
                        best_h = h_candidate.copy()
    
    # Phase 2: Local refinement on best candidate using ADMM-inspired method
    if time.time() < deadline - 10:
        h_refined, v_refined = _local_refine_admm(best_h, time_budget=min(20.0, deadline - time.time() - 10), rng=rng)
        if v_refined < best_val:
            best_val = v_refined
            best_h = h_refined.copy()
    
    # Phase 3: Try a few more random configurations and refine
    for i in range(10):
        if time.time() > deadline - 15:
            break
        
        f1 = rng.choice(candidate_f_values)
        q1 = rng.choice(q_values)
        f2 = rng.choice(candidate_f_values)
        q2 = rng.choice(q_values)
        if f2 == f1:
            f2 = (f2 + 1) % 400
            if np.gcd(f2, 400) != 1:
                f2 = (f2 + 2) % 400
        
        h_candidate = _construct_zc_pair(f1, q1, f2, q2, N)
        v_candidate = _evaluate(h_candidate)
        
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate.copy()
        
        if time.time() < deadline - 10:
            h_refined, v_refined = _local_refine_admm(h_candidate, time_budget=2.0, rng=rng)
            if v_refined < best_val:
                best_val = v_refined
                best_h = h_refined.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
