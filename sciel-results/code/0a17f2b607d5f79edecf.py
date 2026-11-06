import numpy as np
import time

def construct():
    deadline = time.time() + 130  # safety margin
    
    # Precompute z^k for k=0..23 on a dense grid
    N = 4096
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    # Z[k] = z^k for k=0..23, shape (24, N)
    Z = z[None, :] ** np.arange(24)[:, None]
    
    def score(coeffs):
        # coeffs: array of 24 values in {+1, -1}
        # P(z) = sum c_k z^k = coeffs @ Z
        vals = coeffs @ Z  # shape (N,)
        m = np.max(np.abs(vals))
        # Refine top peaks with golden section
        # Find top few peaks
        abs_vals = np.abs(vals)
        # Get indices of top 10 peaks
        top_k = 10
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = m
        for i in idx:
            # Golden section in [theta[i-1], theta[i+1]] (circular)
            # theta spacing
            dtheta = 2 * np.pi / N
            lo = theta[i] - dtheta
            hi = theta[i] + dtheta
            gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            # Evaluate at c and d
            for _ in range(30):
                zc = np.exp(1j * c)
                zd = np.exp(1j * d)
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
            # Final evaluation
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
    for _ in range(50):
        seeds.append(rng.choice([-1.0, 1.0], size=24))
    
    # Local search from each seed
    for s_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        coeffs = seed.copy()
        cur_score = score(coeffs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = coeffs.copy()
        
        # Local search: flip one coefficient at a time
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            best_flip = None
            best_flip_score = cur_score
            for k in range(24):
                new_coeffs = coeffs.copy()
                new_coeffs[k] *= -1
                s = score(new_coeffs)
                if s < best_flip_score:
                    best_flip_score = s
                    best_flip = k
            if best_flip is not None:
                coeffs[best_flip] *= -1
                cur_score = best_flip_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_coeffs = coeffs.copy()
    
    # Return best
    if best_coeffs is None:
        return [1] * 24
    return [int(c) for c in best_coeffs]
