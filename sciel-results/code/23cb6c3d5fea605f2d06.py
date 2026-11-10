import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate RS sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    # p is length 32, take first 24
    rs_24 = p[:24]
    
    # 2. Define evaluation function
    M = 1 << 16 # 65536 points
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Precompute powers? No, compute on the fly in vectorized way inside eval
    
    def max_norm(c):
        # c is list of 24 ints
        # Compute P(theta)
        # k = 0..23
        k = np.arange(24)
        # Z[k, j] = exp(i * k * theta[j])
        # This creates a 24x65536 array.
        # To save memory/time, we can iterate k? 
        # Vectorized is faster.
        # exp(i * outer(k, theta))
        Z = np.exp(1j * np.outer(k, theta))
        P = np.sum(np.array(c)[:, None] * Z, axis=0)
        abs_P = np.abs(P)
        idx_max = np.argmax(abs_P)
        theta_peak = theta[idx_max]
        
        # Refine
        # Golden section search in [theta_peak - dtheta, theta_peak + dtheta]
        dtheta = 2*np.pi/M
        a = theta_peak - dtheta
        b = theta_peak + dtheta
        if a < 0: a += 2*np.pi
        if b > 2*np.pi: b -= 2*np.pi
        # Handle wrap-around if necessary, but for local search it's usually fine
        # Assuming no wrap for simplicity in refinement, or handle carefully.
        # Given M is large, wrap is unlikely to affect the local peak shape significantly unless peak is at 0/2pi.
        
        # Golden section constants
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(t):
            # Evaluate P at single t
            # P = sum c_k exp(i k t)
            # This is slow in python loop, but only ~30 iterations
            # Can use numpy scalar
            vals = np.exp(1j * k * t)
            return np.abs(np.sum(np.array(c) * vals))
            
        fc = f(c)
        fd = f(d)
        
        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        theta_opt = (a + b) / 2
        return f(theta_opt)

    def score(c):
        return max_norm(c) / np.sqrt(24)

    # 3. Search
    best_c = rs_24[:]
    best_score = score(best_c)
    
    # Local Search: Single Flips
    improved = True
    while improved:
        improved = False
        for i in range(24):
            if time.time() > deadline:
                return best_c
            
            new_c = best_c[:]
            new_c[i] = -new_c[i]
            s = score(new_c)
            if s < best_score:
                best_score = s
                best_c = new_c
                improved = True
        
        # Double Flips (optional, if time)
        if time.time() < deadline - 10:
            # Try some double flips
            # Random pairs? Or all pairs? 24*23/2 = 276 pairs.
            # 276 evaluations. Each eval is ~0.1s?
            # M=65536. 24*65536 ops. ~1.5M ops. NumPy is fast.
            # Let's say 10ms per eval. 276 * 10ms = 2.7s.
            # This is feasible.
            for i in range(24):
                if time.time() > deadline:
                    return best_c
                for j in range(i+1, 24):
                    new_c = best_c[:]
                    new_c[i] = -new_c[i]
                    new_c[j] = -new_c[j]
                    s = score(new_c)
                    if s < best_score:
                        best_score = s
                        best_c = new_c
                        improved = True

    return best_c
