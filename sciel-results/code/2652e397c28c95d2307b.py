import numpy as np
import time
import random

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    C = np.fft.irfft(F * F, n=fft_size)
    return C[:size]


def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)


def _make_sparse_sequence(support, signs):
    """Create a sparse sequence with given support positions and signs."""
    f = np.zeros(N)
    for pos, sign in zip(support, signs):
        f[pos] = sign
    return f


def _simulated_annealing_sparse(rng, initial_support, initial_signs, time_budget=30.0):
    """
    Simulated annealing on sparse signed sequences.
    State: (support_set, signs)
    Moves: add/remove/flip sign of an element
    """
    support = list(initial_support)
    signs = list(initial_signs)
    f = _make_sparse_sequence(support, signs)
    current_score = _score(f)
    
    best_support = support.copy()
    best_signs = signs.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1.0  # Temperature
    cooling = 0.999
    
    while time.time() - start_time < time_budget and T > 0.01:
        if time.time() > DEADLINE:
            break
        
        move_type = rng.random()
        
        if move_type < 0.3 and len(support) < 30:
            # Add a new element
            pos = rng.integers(0, N)
            if pos not in support:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                support.append(pos)
                signs.append(sign)
        elif move_type < 0.6 and len(support) > 5:
            # Remove an element
            idx = rng.integers(0, len(support))
            support.pop(idx)
            signs.pop(idx)
        elif move_type < 0.9 and len(support) > 0:
            # Flip sign of an element
            idx = rng.integers(0, len(support))
            signs[idx] = -signs[idx]
        else:
            # Move an element to a new position
            if len(support) == 0:
                continue
            idx = rng.integers(0, len(support))
            old_pos = support[idx]
            new_pos = rng.integers(0, N)
            if new_pos not in support:
                support[idx] = new_pos
            else:
                # Revert
                support[idx] = old_pos
                continue
        
        new_f = _make_sparse_sequence(support, signs)
        new_score = _score(new_f)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_support = support.copy()
                best_signs = signs.copy()
        else:
            # Revert
            if move_type < 0.3 and len(support) > len(initial_support):
                support.pop()
                signs.pop()
            elif move_type < 0.6:
                # Can't easily revert removal without storing state
                pass
            elif move_type < 0.9:
                signs[idx] = -signs[idx]
            else:
                support[idx] = old_pos
        
        T *= cooling
    
    best_f = _make_sparse_sequence(best_support, best_signs)
    return best_f, best_score


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Baseline: all ones
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Strategy 1: Sparse sequences with simulated annealing
    for restart in range(10):
        if time.time() > DEADLINE:
            break
        
        # Start with a random sparse support
        K = rng.integers(8, 25)
        support = list(rng.choice(N, size=K, replace=False))
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(K)]
        
        # Ensure non-zero sum
        S = sum(signs)
        if abs(S) < 3:
            # Adjust signs to increase |S|
            for i in range(K):
                if signs[i] * S < 0:
                    signs[i] = -signs[i]
                    S = sum(signs)
            if abs(S) < 3:
                # Make all same sign
                signs = [1.0] * K
                S = K
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(20.0, remaining_time * 0.8 / (10 - restart))
        
        if time_budget < 1.0:
            break
        
        f_sa, score_sa = _simulated_annealing_sparse(rng, support, signs, time_budget=time_budget)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Strategy 2: Structured sparse patterns (arithmetic progressions, etc.)
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        
        # Try arithmetic progression support
        K = rng.integers(10, 20)
        start = rng.integers(0, N // 2)
        step = rng.integers(2, N // K + 1)
        support = [(start + i * step) % N for i in range(K)]
        # Remove duplicates
        support = list(dict.fromkeys(support))
        K = len(support)
        
        if K < 5:
            continue
        
        # Random signs with bias toward positive sum
        signs = []
        for i in range(K):
            if i < K // 2:
                signs.append(1.0)
            else:
                signs.append(1.0 if rng.random() < 0.7 else -1.0)
        
        S = sum(signs)
        if abs(S) < 3:
            signs = [1.0] * K
        
        f_struct = _make_sparse_sequence(support, signs)
        score_struct = _score(f_struct)
        
        if score_struct < best_score:
            best_score = score_struct
            best_f = f_struct.copy()
    
    # Strategy 3: Local improvement on best sparse solution
    if best_score < 0.05:  # Only if we have a good sparse solution
        f_local = best_f.copy()
        support = list(np.nonzero(f_local)[0])
        signs = [f_local[i] for i in support]
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(15.0, remaining_time * 0.5)
        
        if time_budget > 2.0:
            f_improved, score_improved = _simulated_annealing_sparse(
                rng, support, signs, time_budget=time_budget
            )
            if score_improved < best_score:
                best_score = score_improved
                best_f = f_improved.copy()
    
    return best_f
