"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_fft(U):
    """
    Compute the score using FFT for efficiency.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    if not U or len(U) < 2:
        return 0.0
    
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    n = len(U)
    
    # Compute |U+U|
    # Sumset size is the number of non-zero coefficients in convolution of indicator vector with itself.
    # Let f be indicator of U. (f*f)[k] = sum_{i+j=k} f[i]f[j].
    # We need count of k where (f*f)[k] > 0.
    
    # Compute |U-U|
    # Difference set size is number of non-zero coefficients in convolution of f with reversed f.
    # Let g be indicator of -U (shifted to non-negative).
    # Or simpler: (f * f_reversed)[k] counts pairs (i,j) such that i - j = k - offset.
    
    # FFT size needs to be at least 2*max_u + 1 for sums, and 2*max_u + 1 for diffs (range -max_u to max_u).
    # Actually, sum range is [0, 2*max_u]. Diff range is [-max_u, max_u].
    # Let's use size N = next power of 2 >= 2*max_u + 1.
    
    N = 1
    while N < 2 * max_u + 1:
        N *= 2
    
    # Create indicator vectors
    f = np.zeros(N)
    for u in U:
        f[u] = 1.0
        
    # Convolution for Sumset
    F = np.fft.fft(f)
    sum_fft = F * F
    sum_conv = np.fft.ifft(sum_fft).real
    # Count non-zeros in sum_conv[0 : 2*max_u + 1]
    sum_size = np.count_nonzero(sum_conv[:2*max_u + 1] > 1e-9)
    
    # Convolution for Difference Set
    # We want |{u1 - u2 : u1, u2 in U}|
    # This is equivalent to the support size of f * reverse(f) shifted appropriately.
    # Let f_rev be f reversed and shifted.
    # Actually, correlation is easier: corr[k] = sum f[i] f[i+k].
    # Or use FFT: ifft(fft(f) * conj(fft(f))) gives autocorrelation.
    # Autocorrelation c[k] = sum_i f[i] f[i+k] (circular).
    # For linear autocorrelation, we need zero padding.
    # c[k] for k in [-max_u, max_u].
    # ifft(fft(f) * conj(fft(f))) gives c[k] where index 0 is k=0, index 1 is k=1, ..., index N-1 is k=-1.
    
    diff_fft = F * np.conj(F)
    diff_conv = np.fft.ifft(diff_fft).real
    
    # The linear autocorrelation values are:
    # c[k] for k >= 0 is at index k.
    # c[k] for k < 0 is at index N + k.
    # We need to count non-zeros in the range [-max_u, max_u].
    
    # Indices:
    # k = -max_u ... -1 -> indices N - max_u ... N - 1
    # k = 0 -> index 0
    # k = 1 ... max_u -> indices 1 ... max_u
    
    # Construct the relevant slice
    # Note: diff_conv might have small negative values due to floating point errors.
    # We threshold at 1e-9.
    
    # Slice 1: k < 0
    part_neg = diff_conv[N - max_u : N]
    # Slice 2: k >= 0
    part_pos = diff_conv[0 : max_u + 1]
    
    # Combine and count
    diff_vals = np.concatenate([part_neg, part_pos])
    diff_size = np.count_nonzero(diff_vals > 1e-9)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / denom

def _generate_primes(limit):
    """Generate primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _golomb_ruler(n):
    """
    Generate a Golomb ruler of length n (n marks).
    Simple greedy construction.
    """
    if n <= 1:
        return [0]
    
    marks = [0]
    diffs = set()
    
    for _ in range(n - 1):
        best_mark = None
        best_diffs = None
        best_score = -1
        
        # Try adding a new mark at position p
        # p must be greater than the last mark
        # We want to minimize the number of duplicate differences, or just find a valid one.
        # Greedy: find the smallest p > marks[-1] such that all new differences are unique.
        
        p = marks[-1] + 1
        while True:
            new_diffs = set()
            valid = True
            for m in marks:
                d = p - m
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                best_mark = p
                break
            p += 1
            
            # Safety break if it takes too long
            if p > 10000:
                break
                
        if best_mark is None:
            # Fallback: just append a large number
            best_mark = marks[-1] + 100
            
        marks.append(best_mark)
        for m in marks[:-1]:
            diffs.add(best_mark - m)
            
    return marks

def _local_search_fft(U, deadline, max_iter=50000):
    """
    Local search using FFT scoring.
    Moves: modify element, add element, remove element.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score_fft(best_U)
    
    rng = random.Random(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Randomly choose a move type
        move_type = rng.randint(0, 2)
        improved = False
        
        if move_type == 0: # Modify
            if len(best_U) > 2: # Keep at least 2 elements
                idx = rng.randint(1, len(best_U) - 1)
                old_val = best_U[idx]
                
                # Perturbation based on current scale
                scale = max(1, best_U[-1] // 10)
                delta = rng.randint(-scale, scale)
                if delta == 0:
                    continue
                    
                new_val = old_val + delta
                if new_val < 0 or new_val in best_U:
                    continue
                
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(new_U)
                
                s = _compute_score_fft(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    
        elif move_type == 1: # Add
            if len(best_U) < 100: # Limit size to keep FFT fast
                max_u = best_U[-1]
                # Try adding near existing elements or at the end
                if rng.random() < 0.5:
                    base = rng.choice(best_U)
                    new_val = base + rng.randint(1, max(1, max_u // 5))
                else:
                    new_val = max_u + rng.randint(1, max(1, max_u // 5))
                    
                if new_val in best_U:
                    continue
                    
                new_U = sorted(best_U + [new_val])
                s = _compute_score_fft(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    
        else: # Remove
            if len(best_U) > 2:
                idx = rng.randint(1, len(best_U) - 1)
                new_U = best_U[:idx] + best_U[idx+1:]
                s = _compute_score_fft(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
    
    return best_U

def construct():
    deadline = time.time() + 135  # Safety margin from 150s
    
    start_variants = []
    
    # 1. Golomb Rulers
    # Golomb rulers maximize |U-U|, which is good for the numerator.
    # We try various sizes.
    for n in [5, 10, 15, 20, 30, 40]:
        if time.time() > deadline - 100:
            break
        gr = _golomb_ruler(n)
        start_variants.append(gr)
        
    # 2. Prime step sequences
    primes = _generate_primes(200)
    current = [0]
    for p in primes[:20]:
        current.append(current[-1] + p)
        start_variants.append(current[:])
        
    # 3. Randomized sets
    for _ in range(5):
        if time.time() > deadline - 90:
            break
        n = random.randint(5, 30)
        max_val = random.randint(100, 500)
        U = sorted(random.sample(range(1, max_val), n-1))
        U = [0] + U
        start_variants.append(U)
        
    # Pick the best starting variant using FFT score
    best_start = None
    best_start_score = -1
    for sv in start_variants:
        s = _compute_score_fft(sv)
        if s > best_start_score:
            best_start_score = s
            best_start = sv
            
    U = best_start if best_start else [0, 1, 3]
    
    # Run local search
    # We can run multiple rounds or just one long one.
    # Given FFT speed, one long one is probably better.
    U = _local_search_fft(U, deadline, max_iter=100000)
    
    # Final polish: try scaling
    # Scaling doesn't change |U-U|/|U+U| ratio?
    # If U' = kU, then U'+U' = k(U+U), so |U'+U'| = |U+U|.
    # U'-U' = k(U-U), so |U'-U'| = |U-U|.
    # Ratio is invariant.
    # Denominator is log(2*max(kU)+1) = log(2k*max(U)+1).
    # So larger k decreases the denominator, increasing the score?
    # Wait. Score = 1 + log(ratio) / log(2*max(U)+1).
    # If ratio > 1, log(ratio) > 0. Increasing denominator decreases score.
    # If ratio < 1, log(ratio) < 0. Increasing denominator makes the term less negative (closer to 0), increasing score.
    # Usually |U-U| is larger than |U+U|?
    # For Golomb ruler, |U-U| = n(n-1)+1. |U+U| is roughly n^2/2? No, sums are distinct too for Golomb?
    # Actually, for Golomb rulers, differences are distinct. Sums can collide.
    # So ratio is likely > 1.
    # If ratio > 1, we want SMALL denominator -> SMALL max(U).
    # So scaling up is bad. Scaling down (if integer) might be good?
    # But we can't scale down easily while keeping integers and 0.
    
    # So we should focus on minimizing max(U) while maintaining high ratio.
    # Local search with "modify" moves can effectively shrink the set.
    
    # Ensure 0 is in the set and all are nonnegative
    U = sorted(set([x for x in U if x >= 0]))
    if 0 not in U:
        U = [0] + U
        
    return U

# EVOLVE-BLOCK-END
