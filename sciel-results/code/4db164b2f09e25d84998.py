"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Start with a sequence that has good autocorrelation properties
    # Use a chirp-like or Barker-inspired initial pattern
    # Try: x[i] = 1 for all i, then optimize signs
    x = np.ones(N)
    
    def compute_score(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        # Full autoconvolution
        c = np.correlate(x, x, mode='full')
        # c has length 2N-1, centered at N-1
        # DC term is c[N-1] = sum(x^2)
        # We want max |c[k]| / s^2
        # Include all lags
        max_abs_c = np.max(np.abs(c))
        return max_abs_c / (s * s)
    
    best_x = x.copy()
    best_score = compute_score(x)
    
    # Local search: flip signs to reduce max|c|/s^2
    # Use a smooth surrogate: soft-max of |c| values
    def soft_max_abs_c(x, beta=50.0):
        c = np.correlate(x, x, mode='full')
        ac = np.abs(c)
        # log-sum-exp for soft max
        m = np.max(ac)
        if m < 1e-10:
            return 0.0
        return m + np.log(np.sum(np.exp(beta * (ac - m)))) / beta
    
    def soft_score(x, beta=50.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        return soft_max_abs_c(x, beta) / (s * s)
    
    # Local sign-flip search
    # Precompute for efficiency
    # c[k] = sum_i x[i]*x[i+k], flipping x[j] changes c[k] for k where i or i+k = j
    
    # Use a simpler approach: random sign flips with acceptance
    np.random.seed(42)
    x = np.ones(N).astype(np.float64)
    # Start with a pattern: first half +1, second half -1? No, that gives sum=0.
    # Start with all +1, then try flips.
    
    current_score = compute_score(x)
    best_score = current_score
    best_x = x.copy()
    
    # Try flipping individual elements
    # For efficiency, compute correlation once and update incrementally
    # c_full = np.correlate(x, x, 'full')
    # Flipping x[j]: new c[k] = old c[k] - 2*x[j]*(sum over i where i+k=j of x[i]) + ... 
    # This is complex. Let's use a simpler random restart approach.
    
    # Strategy: try several structured initial patterns, then local search
    patterns = []
    
    # Pattern 1: all ones
    patterns.append(np.ones(N))
    
    # Pattern 2: +1 with a few -1s to reduce sidelobes
    # Pattern 3: based on Legendre symbol
    def legendre_symbol(a, p):
        ls = pow(a, (p-1)//2, p)
        if ls == p-1:
            return -1
        return ls
    
    # Use a prime close to N
    # Find prime near N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
    
    p = N
    while not is_prime(p):
        p += 1
    
    # Legendre sequence of length p, padded to N
    leg = np.array([legendre_symbol(i, p) for i in range(1, p+1)], dtype=np.float64)
    if len(leg) < N:
        leg = np.pad(leg, (0, N - len(leg)), mode='constant', constant_values=1.0)
    else:
        leg = leg[:N]
    patterns.append(leg)
    
    # Pattern 4: alternating with period
    patterns.append(np.array([1.0 if (i % 4) < 2 else -1.0 for i in range(N)]))
    
    # Pattern 5: chirp
    t = np.arange(N)
    patterns.append(np.sign(np.cos(2*np.pi*t*t/(2*N))))
    
    # For each pattern, do local search
    for x0 in patterns:
        x = x0.copy()
        curr = compute_score(x)
        if curr < best_score:
            best_score = curr
            best_x = x.copy()
        
        # Local search: try flipping each position
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for j in range(N):
                if time.time() > deadline:
                    break
                x[j] = -x[j]
                new_s = np.sum(x)
                if abs(new_s) < 1e-10:
                    x[j] = -x[j]
                    continue
                c = np.correlate(x, x, mode='full')
                new_score = np.max(np.abs(c)) / (new_s * new_s)
                if new_score < curr - 1e-12:
                    curr = new_score
                    improved = True
                    if curr < best_score:
                        best_score = curr
                        best_x = x.copy()
                else:
                    x[j] = -x[j]
    
    # Final polish with smaller steps if time permits
    if time.time() < deadline - 5:
        x = best_x.copy()
        curr = compute_score(x)
        for _ in range(10):  # a few more passes
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for j in range(N):
                    if time.time() > deadline:
                        break
                    x[j] = -x[j]
                    new_s = np.sum(x)
                    if abs(new_s) < 1e-10:
                        x[j] = -x[j]
                        continue
                    c = np.correlate(x, x, mode='full')
                    new_score = np.max(np.abs(c)) / (new_s * new_s)
                    if new_score < curr - 1e-14:
                        curr = new_score
                        improved = True
                        if curr < best_score:
                            best_score = curr
                            best_x = x.copy()
                    else:
                        x[j] = -x[j]
    
    return best_x
# EVOLVE-BLOCK-END
