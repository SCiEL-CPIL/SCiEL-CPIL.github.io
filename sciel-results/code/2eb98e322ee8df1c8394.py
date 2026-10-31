import numpy as np
import time

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(h):
    """Compute full autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    # Use FFT for efficient convolution
    f = np.fft.rfft(h, size=2 * size)
    ac = np.fft.irfft(f * f, n=2 * size)[:size]
    return ac


def _score(h):
    """Compute max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)


def _walsh_hadamard_sequence(n):
    """Generate a Walsh-Hadamard based signed sequence."""
    # Use the first row of a Walsh-Hadamard matrix (all ones) 
    # or a more interesting pattern
    # Actually, let's use a Rademacher function based approach
    # Or use the properties of Hadamard matrices
    
    # Generate using the Hadamard transform of a sparse vector
    # For simplicity, use a deterministic pattern based on bit-reversal
    seq = np.ones(n)
    for i in range(n):
        # Count trailing zeros in binary representation
        x = i
        tz = 0
        while x % 2 == 0 and x > 0:
            tz += 1
            x //= 2
        seq[i] = 1.0 if tz % 2 == 0 else -1.0
    return seq


def _m_sequence(n):
    """Generate an m-sequence-like pattern using LFSR."""
    # Use a simple LFSR with a primitive polynomial
    # For length close to N, we'll use a maximal-length sequence
    # Primitive polynomial for degree 8: x^8 + x^4 + x^3 + x^2 + 1
    # But we need length ~400, so let's use degree 9: x^9 + x^5 + 1
    
    # Actually, let's use a simpler approach: generate using a known primitive polynomial
    # For our purposes, we'll use a pseudo-random signed sequence with good autocorrelation properties
    
    # Use the Gold code approach: combine two m-sequences
    # Simpler: use a deterministic pattern based on quadratic residues
    
    seq = np.zeros(n)
    for i in range(n):
        # Use the Legendre symbol mod a prime
        # Find a prime >= n
        p = n
        while True:
            if p > 1:
                is_prime = True
                for d in range(2, int(np.sqrt(p)) + 1):
                    if p % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    break
            p += 1
        
        if i % p == 0:
            seq[i] = 1.0
        else:
            # Legendre symbol (i|p)
            ls = pow(i % p, (p - 1) // 2, p)
            if ls == p - 1:
                seq[i] = -1.0
            else:
                seq[i] = 1.0
    return seq


def _local_search(h, max_time=60.0):
    """Perform local search by flipping signs to reduce max|autoconv|."""
    h = h.copy()
    best_score = _score(h)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        ac = _autoconv(h)
        max_abs = np.max(np.abs(ac))
        
        # Try flipping each sign and see if it reduces the objective
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            
            # Flip sign at position i
            h[i] *= -1
            new_score = _score(h)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                h[i] *= -1
        
        if not improved:
            # Try a random restart of a few positions
            for _ in range(5):
                idx = np.random.randint(0, N, size=3)
                for i in idx:
                    h[i] *= -1
                new_score = _score(h)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    for i in idx:
                        h[i] *= -1
            
            if not improved:
                break
    
    return h, best_score


def _gradient_descent(h, max_time=30.0):
    """Perform gradient descent on a smooth proxy objective."""
    h = h.copy().astype(float)
    start_time = time.time()
    lr = 0.1
    best_h = h.copy()
    best_score = _score(h)
    
    while time.time() - start_time < max_time:
        # Compute gradient of max|autoconv| w.r.t. h
        # Use subgradient: the index where |autoconv| is maximized
        ac = _autoconv(h)
        idx = np.argmax(np.abs(ac))
        sign = np.sign(ac[idx])
        
        # Gradient of ac[idx] w.r.t. h is h shifted
        # ac[idx] = sum_i h[i] * h[i+idx]
        # d(ac[idx])/dh[j] = h[j+idx] + h[j-idx] (with boundary handling)
        grad = np.zeros_like(h)
        for j in range(N):
            if j + idx < N:
                grad[j] += h[j + idx]
            if j - idx >= 0:
                grad[j] += h[j - idx]
        
        # Subgradient of |ac[idx]|
        grad *= sign
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad /= gnorm
        
        # Update
        h -= lr * grad
        
        # Project to signed values (soft threshold)
        h = np.sign(h) * np.maximum(0, np.abs(h) - 0.1)
        
        new_score = _score(h)
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
        
        # Decay learning rate
        lr *= 0.99
    
    return best_h, best_score


def construct():
    np.random.seed(42)
    
    # Generate multiple candidate initializations
    candidates = []
    
    # 1. Walsh-Hadamard based
    candidates.append(_walsh_hadamard_sequence(N))
    
    # 2. Quadratic residue based
    candidates.append(_m_sequence(N))
    
    # 3. Random signed
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 4. Another deterministic pattern: alternating blocks
    h4 = np.ones(N)
    for i in range(0, N, 4):
        h4[i:i+2] = -1.0
    candidates.append(h4)
    
    # 5. Sine-based pattern
    x = np.arange(N)
    candidates.append(np.sign(np.sin(2 * np.pi * x / N * 3)))
    
    # Evaluate initial scores
    best_candidate = None
    best_initial_score = float('inf')
    for c in candidates:
        s = _score(c)
        if s < best_initial_score:
            best_initial_score = s
            best_candidate = c.copy()
    
    # Apply local search to the best candidate
    h_local, score_local = _local_search(best_candidate, max_time=50.0)
    
    # Apply gradient descent to the best candidate
    if time.time() < DEADLINE:
        h_grad, score_grad = _gradient_descent(best_candidate, max_time=30.0)
        if score_grad < score_local:
            h_local = h_grad
            score_local = score_grad
    
    # Try local search from other candidates too
    for c in candidates:
        if time.time() > DEADLINE:
            break
        if np.allclose(c, best_candidate):
            continue
        h_try, score_try = _local_search(c, max_time=10.0)
        if score_try < score_local:
            h_local = h_try
            score_local = score_try
    
    # Final check
    if time.time() > DEADLINE - 2.0:
        return h_local
    
    # One more round of local search
    h_final, score_final = _local_search(h_local, max_time=15.0)
    
    return h_final


# Ensure we don't exceed time limit
if __name__ == "__main__":
    result = construct()
    print(f"Score: {_score(result)}")
    print(f"Length: {len(result)}")
