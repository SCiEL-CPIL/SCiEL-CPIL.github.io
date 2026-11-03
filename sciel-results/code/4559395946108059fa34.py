"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

# EVOLVE-BLOCK-START

def _score_set_fft(U):
    """
    Compute the score for a finite set U using FFT for speed.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    if 0 not in U:
        return -1.0
    max_u = max(U)
    if max_u == 0:
        return -1.0
    
    # Compute |U+U| using FFT
    sum_len = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute |U-U| using FFT (autocorrelation)
    # The difference set U-U = {a-b : a,b in U}
    # We compute the autocorrelation: for each shift d, count pairs (a,b) with a-b=d
    # autocorr[d] = sum_i indicator[i] * indicator[i+d] for d >= 0
    # But we need |{a-b : a,b in U}| which is the number of distinct values.
    # Use the fact that the indicator of U-U can be computed via:
    # Create a vector where position (max_u + d) gets 1 if d is in U-U
    # This is the cross-correlation of indicator with itself, shifted.
    
    # Alternative: compute directly for moderate sizes, FFT for large
    if n <= 200:
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        diffset_size = len(diffset)
    else:
        # For large sets, use FFT-based computation of difference set size
        # The number of distinct differences equals the number of nonzero entries
        # in the autocorrelation function (shifted appropriately)
        # autocorr[k] = sum_i f[i] * f[i+k] where f is indicator
        # A difference d exists iff autocorr[d] > 0 (for d>=0) or autocorr[-d] > 0
        # Actually, let's compute it properly:
        # Create a vector g where g[i] = 1 if i in U
        # Then the cross-correlation h[d] = sum_i g[i] * g[i-d] tells us
        # if difference d exists (h[d] > 0)
        # We can compute this via FFT: h = ifft(fft(g) * conj(fft(g)))
        # but we need to be careful about the shift.
        
        # Simpler approach: use the fact that |U-U| can be computed by
        # counting nonzero entries in the circular autocorrelation
        freq_conj = np.conjugate(freq)
        autocorr = ifft(freq * freq_conj).real
        # autocorr[k] = sum_i g[i] * g[(i+k) mod fft_len]
        # For our purposes, the differences that exist are those where
        # autocorr[max_u + d] > 0 for d in range(-max_u, max_u+1)
        # Actually, let me think again...
        # The standard autocorrelation gives us for each lag d, the number of
        # pairs (i, i+d) both in U. So difference d exists iff autocorr[d] > 0.
        # But autocorr is computed on a circular array of length fft_len.
        # For d in [-max_u, max_u], autocorr[(d + fft_len) % fft_len] > 0 means d exists.
        
        diffset_size = 0
        for d in range(-max_u, max_u + 1):
            idx = (d + fft_len) % fft_len
            if autocorr[idx] > 1e-9:
                diffset_size += 1
    
    if sumset_size == 0:
        return -1.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom

def _score_set_direct(U):
    """Compute the score for a finite set U using direct computation (for small sets)."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    maxU = max(U)
    if maxU == 0:
        return -1.0
    
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ss = len(sumset)
    ds = len(diffset)
    if ss == 0:
        return -1.0
    ratio = ds / ss
    if ratio <= 0:
        return -1.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom

def _simulated_annealing(U, deadline, max_iter=5000):
    """
    Simulated annealing to optimize the set.
    Moves: add, remove, or replace an element.
    """
    start = time.time()
    U = sorted(set(U))
    current_score = _score_set_fft(U)
    best_U = list(U)
    best_score = current_score
    
    T = 0.5  # Initial temperature
    cooling = 0.9995
    
    max_val = max(U) if U else 20
    search_range = max(max_val * 3, 100)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if time.time() - start > 60:
            break
        
        # Generate a random move
        move_type = random.random()
        new_U = list(U)
        
        if move_type < 0.3 and len(U) < 300:
            # Add a random element
            new_val = random.randint(1, search_range)
            if new_val not in U:
                new_U.append(new_val)
            else:
                continue
        elif move_type < 0.5 and len(U) > 2:
            # Remove a random non-zero element
            candidates = [x for x in U if x != 0]
            if candidates:
                rem = random.choice(candidates)
                new_U = [x for x in new_U if x != rem]
            else:
                continue
        elif move_type < 0.8:
            # Replace a random non-zero element
            candidates = [x for x in U if x != 0]
            if candidates:
                old = random.choice(candidates)
                new = random.randint(1, search_range)
                if new not in U or new == old:
                    new_U = [new if x == old else x for x in new_U]
                else:
                    continue
            else:
                continue
        else:
            # Shift a random non-zero element
            candidates = [x for x in U if x != 0]
            if candidates:
                old = random.choice(candidates)
                delta = random.choice([-10, -5, -2, -1, 1, 2, 5, 10, 20])
                new = old + delta
                if new >= 0 and new != old and new not in U:
                    new_U = [new if x == old else x for x in new_U]
                else:
                    continue
            else:
                continue
        
        new_U = sorted(set(new_U))
        if 0 not in new_U or len(new_U) < 2:
            continue
        
        new_score = _score_set_fft(new_U)
        
        # Accept or reject
        if new_score >= current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_U = list(U)
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta_score = new_score - current_score
            if delta_score > -T * 10:  # Avoid underflow
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    U = new_U
                    current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def _greedy_construct(max_size=50, deadline=None):
    """
    Greedy construction: start with {0,1} and add elements that maximize the score.
    """
    U = [0, 1]
    best_score = _score_set_fft(U)
    
    max_val = 10
    for step in range(max_size - 2):
        if deadline is not None and time.time() > deadline:
            break
        
        best_candidate = None
        best_candidate_score = best_score
        
        # Try adding each candidate value
        for candidate in range(2, max_val + 1):
            if candidate in U:
                continue
            new_U = U + [candidate]
            new_U = sorted(set(new_U))
            s = _score_set_fft(new_U)
            if s > best_candidate_score + 1e-12:
                best_candidate_score = s
                best_candidate = candidate
        
        if best_candidate is not None:
            U.append(best_candidate)
            U = sorted(set(U))
            best_score = best_candidate_score
            max_val = max(max_val, U[-1] + 5)
        else:
            # No improvement, expand search range
            max_val += 10
    
    return U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_set_fft(best_U)
    
    # Phase 1: Greedy construction
    if time.time() < deadline - 30:
        greedy_U, greedy_score = _greedy_construct(max_size=30, deadline=deadline - 30)
        if greedy_score > best_score:
            best_score = greedy_score
            best_U = greedy_U
    
    # Phase 2: Simulated annealing from best
    if time.time() < deadline - 20:
        sa_U, sa_score = _simulated_annealing(best_U, deadline=deadline - 20, max_iter=3000)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
    
    # Phase 3: Multiple random restarts with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        num_restarts = min(5, int(remaining_time / 8))
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate a random starting set
            size = random.randint(5, 20)
            max_val = random.randint(30, 150)
            U = {0}
            while len(U) < size:
                U.add(random.randint(1, max_val))
            U = sorted(U)
            
            s = _score_set_fft(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
            
            # Simulated annealing from this start
            if time.time() < deadline - 5:
                sa_U, sa_score = _simulated_annealing(U, deadline=deadline - 5, max_iter=1500)
                if sa_score > best_score:
                    best_score = sa_score
                    best_U = sa_U
    
    # Phase 4: Final local polish using direct scoring for the best set
    if time.time() < deadline - 2:
        # Try small perturbations on the best set
        max_val = max(best_U) if best_U else 20
        search_range = max(max_val * 2, 50)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            new_U = list(best_U)
            move_type = random.random()
            if move_type < 0.4 and len(new_U) < 200:
                new_val = random.randint(1, search_range)
                if new_val not in new_U:
                    new_U.append(new_val)
            elif move_type < 0.7 and len(new_U) > 2:
                candidates = [x for x in new_U if x != 0]
                if candidates:
                    rem = random.choice(candidates)
                    new_U = [x for x in new_U if x != rem]
            else:
                candidates = [x for x in new_U if x != 0]
                if candidates:
                    old = random.choice(candidates)
                    new = random.randint(1, search_range)
                    if new not in new_U:
                        new_U = [new if x == old else x for x in new_U]
            
            new_U = sorted(set(new_U))
            if 0 not in new_U or len(new_U) < 2:
                continue
            
            s = _score_set_fft(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
    
    return best_U

# EVOLVE-BLOCK-END
