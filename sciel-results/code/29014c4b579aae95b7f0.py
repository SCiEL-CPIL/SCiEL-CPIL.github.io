"""Rewritten program using cellular-automaton / statistical-mechanics ground-state search."""
import numpy as np
import time

N = 400


def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)


def _init_binary(n, seed):
    """Generate a random binary sign sequence."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)


def _update_ac(h, ac, i, delta_h):
    """
    Update autoconvolution incrementally when h[i] changes by delta_h.
    ac_new = ac_old + 2 * delta_h * (h shifted by i) + delta_h^2 * (impulse at 2i)
    Note: h is assumed to be the OLD h before update for the cross term calculation.
    """
    n = len(h)
    size = 2 * n - 1
    # Cross term: 2 * delta_h * convolve(h, e_i)
    # convolve(h, e_i) is just h shifted by i.
    # In linear convolution index k, the term is h[k-i].
    # Valid range for k-i in [0, n-1] => i <= k <= i + n - 1.
    
    # We need to add 2 * delta_h * h[k-i] to ac[k] for valid k.
    # And delta_h^2 to ac[2*i] (since e_i * e_i is impulse at 2i).
    
    # Let's do it via array slicing for speed
    start = i
    end = i + n
    # The slice of h that contributes is h[0:n]
    # ac[start:end] += 2 * delta_h * h
    
    # Wait, the convolution is sum_j h[j] h[k-j].
    # If we change h[i] to h[i] + delta, the new term is:
    # old: h[i]h[i] + sum_{j!=i} h[j]h[k-j]
    # new: (h[i]+delta)h[i] + h[i](h[i]+delta) + ... 
    # Actually simpler:
    # ac_new[k] = sum_j (h[j] + delta*delta_{j,i}) (h[k-j] + delta*delta_{k-j,i})
    # = ac_old[k] + delta * sum_j h[k-j] delta_{j,i} + delta * sum_j h[j] delta_{k-j,i} + delta^2 delta_{j,i}delta_{k-j,i}
    # = ac_old[k] + delta * h[k-i] + delta * h[i] (if k-i is valid? No, sum_j h[j] delta_{k-j,i} is h[k-i] if k-i in [0,n-1])
    # Wait, sum_j h[j] delta_{k-j,i} = h[k-i] if k-i is a valid index for h.
    # So cross terms are: delta * h[k-i] (from first part) + delta * h[k-i] (from second part)?
    # Let's check indices carefully.
    # Term 1: sum_j h[j] * delta_{j,i} * h[k-j] = h[i] * h[k-i] (if k-i valid)
    # Term 2: sum_j h[j] * delta_{k-j,i} * h[k-j] ... no.
    # Expansion: (h + d e_i) * (h + d e_i) = h*h + d e_i*h + d h*e_i + d^2 e_i*e_i
    # (e_i * h)[k] = h[k-i]
    # (h * e_i)[k] = h[k-i]
    # So cross term is 2 * d * h[k-i].
    # Valid for k-i in [0, n-1] => i <= k <= i+n-1.
    
    if start < size and end > 0:
        # Clip to valid ac range
        s_lo = max(0, start)
        s_hi = min(size, end)
        # Corresponding indices in h: k - i
        h_lo = s_lo - i
        h_hi = s_hi - i
        ac[s_lo:s_hi] += 2.0 * delta_h * h[h_lo:h_hi]
        
    # Impulse term
    imp_idx = 2 * i
    if imp_idx < size:
        ac[imp_idx] += delta_h * delta_h
        
    return ac


def _construct_sa():
    """
    Simulated Annealing on binary signs.
    """
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = None
    
    # Try a few different seeds
    for seed in [42, 123, 999, 7]:
        if time.time() > deadline - 5:
            break
            
        h = _init_binary(N, seed)
        s = np.sum(h)
        if abs(s) < 1e-9:
            # Flip one to avoid zero sum, though unlikely for N=400
            h[0] = -h[0]
            s = np.sum(h)
            
        ac = _autoconv(h)
        current_score = np.max(np.abs(ac)) / (s * s)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Annealing parameters
        # We want to minimize max|ac| / s^2. Since s is fixed (for binary, s changes by -2 or +2 on flip), 
        # we can just minimize max|ac| primarily, but s^2 varies.
        # Actually, for binary sequences, s is integer. 
        # Let's just use the full score as energy.
        
        T = 1.0
        T_min = 1e-4
        alpha = 0.995
        max_iter = 20000
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Pick random index
            i = np.random.randint(N)
            
            # Flip sign
            delta_h = -2.0 * h[i]
            h[i] += delta_h
            
            # Update AC
            ac = _update_ac(h, ac, i, delta_h)
            
            new_s = s + delta_h
            if abs(new_s) < 1e-9:
                # Revert
                h[i] -= delta_h
                ac = _update_ac(h, ac, i, -delta_h)
                s -= delta_h
                continue
                
            new_score = np.max(np.abs(ac)) / (new_s * new_s)
            delta_E = new_score - current_score
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                # Accept
                s = new_s
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Revert
                h[i] -= delta_h
                ac = _update_ac(h, ac, i, -delta_h)
                
            T *= alpha
            
        # Local refinement (greedy)
        if time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    delta_h = -2.0 * h[i]
                    h[i] += delta_h
                    ac = _update_ac(h, ac, i, delta_h)
                    new_s = s + delta_h
                    if abs(new_s) < 1e-9:
                        h[i] -= delta_h
                        ac = _update_ac(h, ac, i, -delta_h)
                        continue
                    new_score = np.max(np.abs(ac)) / (new_s * new_s)
                    if new_score < current_score:
                        s = new_s
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_h = h.copy()
                    else:
                        h[i] -= delta_h
                        ac = _update_ac(h, ac, i, -delta_h)

    return best_h


def construct():
    t_start = time.time()
    
    # Use the SA approach
    h = _construct_sa()
    
    # Fallback if something went wrong
    if h is None:
        h = np.ones(N)
        
    return h
