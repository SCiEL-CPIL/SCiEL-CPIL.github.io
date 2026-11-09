import numpy as np
import time

def _eval_peak(c):
    M = 8192
    V = np.fft.fft(c, M)
    k_max = np.argmax(np.abs(V))
    
    center = 2 * np.pi * k_max / M
    delta = np.pi / M
    
    def f(theta):
        js = np.arange(24)
        vals = np.sum(c * np.exp(1j * theta * js))
        return np.abs(vals)
        
    gr = (np.sqrt(5) - 1) / 2
    a = center - delta
    b = center + delta
    
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1 = f(c1)
    f2 = f(c2)
    
    for _ in range(50):
        if f1 < f2:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
        else:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
            
    theta_opt = (a + b) / 2
    return f(theta_opt)

def construct():
    deadline = time.time() + 140
    best_poly = None
    best_score = float('inf')
    
    seeds = []
    seeds.append([1] * 24)
    
    # Generate Rudin-Shapiro sequence of length 32
    # a[0] = 1
    # a[2n] = a[n]
    # a[2n+1] = -a[n+1] is NOT the standard definition.
    # Standard:
    # a_0 = 1
    # a_{2n} = a_n
    # a_{2n+1} = -a_{n+1} ??
    # Let's check the first few terms of RS: 1, 1, -1, 1, -1, -1, 1, -1...
    # Using recursion:
    # P_0 = 1
    # P_{k+1}(z) = P_k(z) (1 + z^{2^k})
    # Coeffs of P_{k+1} are coeffs of P_k followed by coeffs of P_k shifted by 2^k.
    # So if P_k has coeffs p, P_{k+1} has coeffs p + [0]*2^k + p.
    # This creates a sequence of length 2^{k+1}.
    # For k=0, P_0=[1]. P_1 = [1, 0, 1]. Length 3.
    # This is not length 2.
    # The RS polynomial of degree $2^k-1$ has length $2^k$.
    # The recursion $R_k(z) = R_{k-1}(z) (1 + z^{2^{k-1}})$ works if $R_{k-1}$ has length $2^{k-1}$.
    # $R_0(z) = 1$ (length 1).
    # $R_1(z) = 1(1+z) = 1+z$ (length 2). Coeffs [1, 1].
    # $R_2(z) = (1+z)(1+z^2) = 1+z+z^2+z^3$ (length 4). Coeffs [1, 1, 1, 1].
    # $R_3(z) = (1+z+z^2+z^3)(1+z^4) = 1+z+z^2+z^3+z^4+z^5+z^6+z^7$ (length 8).
    # This generates all 1s. This is NOT the Rudin-Shapiro sequence.
    # The RS sequence has $\pm 1$ coefficients.
    # The correct recursion for the *sequence* is:
    # $a_{2n} = a_n$
    # $a_{2n+1} = -a_{n+1}$?
    # Let's check:
    # n=0: a0=1.
    # n=1: a2=a1? No.
    # Let's use the known sequence: 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1...
    # I will just use random seeds. The search is fast enough.
    
    rng = np.random.default_rng(42)
    for _ in range(50):
        seeds.append(rng.choice([-1, 1], size=24).tolist())
        
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current = seed.copy()
        current_score = _eval_peak(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_score = current_score
            best_flip_idx = -1
            
            for j in range(24):
                flip = current.copy()
                flip[j] *= -1
                flip_score = _eval_peak(flip)
                
                if flip_score < best_flip_score:
                    best_flip_score = flip_score
                    best_flip_idx = j
                    
            if best_flip_idx != -1:
                current[best_flip_idx] *= -1
                current_score = best_flip_score
                improved = True
                
        if current_score < best_score:
            best_score = current_score
            best_poly = current

    if best_poly is None:
        return [1] * 24
        
    return best_poly
