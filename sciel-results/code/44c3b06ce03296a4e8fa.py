import numpy as np
import time

def construct():
    deadline = time.time() + 110  # safety margin for overhead
    
    # Precompute z^k for k=0..23 on a dense grid
    N = 2048
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    # Z[k] = z^k for k=0..23, shape (24, N)
    Z = z[None, :] ** np.arange(24)[:, None]
    
    def score_fast(coeffs):
        """
        Fast evaluation using dense grid only.
        Returns max |P(z)| on the grid.
        """
        # coeffs: array of 24 values in {+1, -1}
        vals = coeffs @ Z  # shape (N,)
        return np.max(np.abs(vals))

    def score_precise(coeffs):
        """
        Precise evaluation with golden-section refinement of top peaks.
        """
        # coeffs: array of 24 values in {+1, -1}
        vals = coeffs @ Z  # shape (N,)
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        
        # Refine top peaks with golden section
        top_k = 10
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = m
        
        dtheta = 2 * np.pi / N
        gr = (np.sqrt(5) - 1) / 2
        
        for i in idx:
            lo = theta[i] - dtheta
            hi = theta[i] + dtheta
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate P at c and d
            zc = np.exp(1j * c)
            zd = np.exp(1j * d)
            # Use Horner's method or dot product? 
            # Dot product with precomputed powers is faster for small N
            # But we need powers of zc and zd, not just z^k on grid.
            # So we compute zc^k and zd^k.
            # Since 24 is small, a loop is fine, but vectorized is better.
            # Let's just use sum(coeffs[k] * zc**k)
            # Actually, we can precompute powers if we wanted, but zc changes.
            # 24 multiplications is trivial.
            
            for _ in range(40):
                pc = sum(coeffs[k] * zc**k for k in range(24))
                pd = sum(coeffs[k] * zd**k for k in range(24))
                ac = abs(pc)
                ad = abs(pd)
                
                if ac < ad:
                    b = d
                    d = c
                    zd = zc
                    ad = ac
                    c = b - gr * (b - a)
                    zc = np.exp(1j * c)
                else:
                    a = c
                    c = d
                    zc = zd
                    ac = ad
                    d = a + gr * (b - a)
                    zd = np.exp(1j * d)
                    
                if time.time() > deadline - 5:
                    break
            
            zm = np.exp(1j * (a + b) / 2)
            pm = sum(coeffs[k] * zm**k for k in range(24))
            am = abs(pm)
            if am > best:
                best = am
        
        return best

    best_coeffs = None
    best_score = float('inf')
    
    # Seed 1: Golay-Rudin-Shapiro truncation
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    gr_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Seed 2: Random
    rng = np.random.default_rng(42)
    
    seeds = [gr_coeffs]
    for _ in range(20):
        seeds.append(rng.choice([-1.0, 1.0], size=24))
    
    # Local search from each seed
    for s_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        coeffs = seed.copy()
        # Initial precise score to set baseline
        cur_score = score_precise(coeffs)
        cur_fast_score = score_fast(coeffs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = coeffs.copy()
            
        # Precompute current P(z) on grid for fast neighbor evaluation
        P_vals = coeffs @ Z
        
        improved = True
        # Use a loop that allows multiple flips
        while improved and time.time() < deadline - 15:
            improved = False
            best_flip = None
            best_flip_score = cur_score
            best_flip_fast = cur_fast_score
            
            # Try flipping each coefficient
            for k in range(24):
                # P_new = P - 2 * c_k * z^k
                P_new = P_vals - 2.0 * coeffs[k] * Z[k]
                fast_score = np.max(np.abs(P_new))
                
                # Only do precise evaluation if fast score is promising
                # We compare against current fast score to ensure improvement
                # Note: fast score is a lower bound on true score.
                # If fast_score < cur_fast_score, it's likely better.
                # If fast_score is close, we might still want to check.
                # Let's use a threshold: if fast_score < cur_fast_score * 0.99 or just < cur_fast_score
                if fast_score < best_flip_fast:
                    # Perform precise evaluation for this candidate
                    new_coeffs = coeffs.copy()
                    new_coeffs[k] *= -1
                    precise_score = score_precise(new_coeffs)
                    
                    if precise_score < best_flip_score:
                        best_flip_score = precise_score
                        best_flip_fast = fast_score
                        best_flip = k
            
            if best_flip is not None:
                coeffs[best_flip] *= -1
                cur_score = best_flip_score
                cur_fast_score = best_flip_fast
                # Update P_vals
                P_vals = coeffs @ Z
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_coeffs = coeffs.copy()
    
    # Final polish: try to improve best_coeffs with more precise search if time remains
    if best_coeffs is not None and time.time() < deadline - 5:
        # Try random perturbations of the best solution
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            cand = best_coeffs.copy()
            # Flip 1 or 2 random coefficients
            n_flips = rng.integers(1, 3)
            for _ in range(n_flips):
                k = rng.integers(0, 24)
                cand[k] *= -1
            s = score_precise(cand)
            if s < best_score:
                best_score = s
                best_coeffs = cand.copy()

    if best_coeffs is None:
        return [1] * 24
    return [int(c) for c in best_coeffs]
