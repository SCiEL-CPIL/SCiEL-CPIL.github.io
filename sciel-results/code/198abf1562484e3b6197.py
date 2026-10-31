import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin vs 150 s kill

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    # We need to represent the indicator function of U.
    # Let f(x) = 1 if x in U, 0 otherwise.
    # Sum set S = {a+b | a,b in U}. The number of representations is the convolution f * f.
    # |S| is the number of indices k where (f*f)[k] > 0.
    
    # Difference set D = {a-b | a,b in U}.
    # Note that a-b can be negative.
    # Let g(x) = f(x).
    # The autocorrelation h(k) = sum_x f(x) f(x+k) counts pairs (x, x+k) such that both are in U.
    # This corresponds to differences k.
    # Specifically, h(k) > 0 iff there exists a, b in U such that a - b = k.
    # So |D| is the number of k in [-max_u, max_u] such that h(k) > 0.
    
    # FFT size needs to be at least 2*max_val + 1 for convolution to avoid wrap-around.
    # Let N = next power of 2 >= 2*max_val + 1.
    
    M = max_val
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    # Create indicator array
    # f is length N, with f[x] = 1 if x in U
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sum set: convolution f * f
    # Inverse FFT of F^2
    conv_sum = np.fft.ifft(F * F).real
    # Due to floating point errors, values might be slightly negative or non-integer.
    # We just need > 0.
    # The valid range for sums is [0, 2*M].
    # conv_sum[k] represents the count of pairs (a,b) with a+b=k.
    # We count k in [0, 2*M] where conv_sum[k] > 0.5 (threshold to handle noise).
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set: autocorrelation
    # Autocorrelation is ifft(F * conj(F))
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # auto_diff[k] for k in [0, N-1] represents sum_x f(x) f(x-k) ??
    # Let's check the indexing.
    # Standard circular autocorrelation r[k] = sum_x f[x] f[(x+k) mod N].
    # For linear autocorrelation (no wrap-around), we need N large enough.
    # r[k] for k >= 0 corresponds to sum_x f[x] f[x+k].
    # If f[x] and f[x+k] are both 1, then x in U and x+k in U => (x+k) - x = k.
    # So r[k] > 0 implies k is in U-U.
    # What about negative differences?
    # r[k] = sum_x f[x] f[x+k].
    # Let a = x+k, b = x. Then a-b = k.
    # If we want -k, we look at r[N-k]?
    # Actually, the set of differences is symmetric: if d in U-U, then -d in U-U.
    # So |U-U| = 2 * (number of positive differences) + (1 if 0 in U-U).
    # 0 is always in U-U since 0 in U (0-0=0).
    # We need to count distinct non-zero differences.
    # The autocorrelation r[k] for k in [1, M] gives the number of pairs with difference k.
    # If r[k] > 0, then k is a difference.
    # Since differences are symmetric, the total size is 1 + 2 * count(k in [1, M] such that r[k] > 0).
    
    # Let's verify the index mapping for numpy ifft of F*conj(F).
    # numpy.fft.ifft returns array y where y[k] = sum_j x[j] exp(2 pi i j k / N).
    # Here x = F * conj(F).
    # This computes the circular autocorrelation.
    # y[k] = sum_j F[j] conj(F[j]) exp(2 pi i j k / N) ?? No.
    # Let's stick to the property: ifft(fft(f) * conj(fft(f))) is the circular autocorrelation.
    # r[k] = sum_{x=0}^{N-1} f[x] f[(x+k) mod N].
    # For k in [0, M], since f is zero outside [0, M], the wrap-around doesn't affect the sum for the "positive" lag interpretation if we are careful.
    # Actually, standard linear autocorrelation c[k] = sum_x f[x] f[x+k] is obtained from the real part of ifft(fft(f) * conj(fft(f))) at index k.
    # And c[-k] = c[k].
    # So we just need to count k in [1, M] where c[k] > 0.
    
    # The output of ifft is centered? No, numpy ifft output index k corresponds to lag k.
    # c[0] = sum f[x]^2 = |U|.
    # c[k] for k>0 is sum_x f[x] f[x+k].
    
    # We need to be careful with the range.
    # The array auto_diff has length N.
    # Indices 0 to M contain the non-negative lags.
    # Indices N-M to N-1 contain the negative lags (symmetric).
    
    # Count positive differences
    # Threshold 0.5 to avoid noise
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0 # If max is 0, log(1)=0, score=1. But U must contain 0. If U={0}, |U-U|=1, |U+U|=1, ratio=1, log=0. Score=1.
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _random_initial_set(n, max_val):
    """
    Generate a random set of size n containing 0, within [0, max_val].
    """
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    
    # Ensure 0 is included
    others = random.sample(range(1, max_val + 1), n - 1)
    u = [0] + others
    return sorted(u)

def _local_search(u, max_val, iterations, temperature_start=1.0, cooling=0.999):
    """
    Simulated Annealing local search.
    Moves: 
    1. Replace a random element with a new random value in [0, max_val].
    2. Add a new random element (if we allow variable size? No, fix size for stability? 
       The problem asks for a finite set. Size is not fixed. 
       However, larger sets generally have larger |U-U| and |U+U|.
       Let's try to optimize for fixed size first, then maybe vary size.
       Actually, the score depends on the ratio and the max value.
       Let's fix the size n and max_val M during a search phase, and vary them between phases.
    """
    current_u = list(u)
    current_score = _score(current_u)
    best_u = list(current_u)
    best_score = current_score
    
    temperature = temperature_start
    
    for i in range(iterations):
        if time.time() >= _DEADLINE:
            break
            
        # Generate neighbor
        neighbor = list(current_u)
        move_type = random.random()
        
        if move_type < 0.7:
            # Replace a random element (not 0, to keep 0)
            if len(neighbor) > 1:
                # Pick index > 0
                idx = random.randint(1, len(neighbor) - 1)
                # New value
                new_val = random.randint(0, max_val)
                # Ensure uniqueness? The set representation handles it, but we want to maintain size n.
                # If new_val is already in neighbor, we might reduce size.
                # Better: pick a value not in neighbor? Or just replace and deduplicate.
                # If we deduplicate, size might change.
                # Let's enforce fixed size by ensuring we pick a value not in the set, 
                # or if we replace with existing, we don't change the set effectively.
                # To be efficient, let's just pick a random value.
                neighbor[idx] = new_val
                # Deduplicate and sort
                neighbor = sorted(set(neighbor))
                # If size changed, we might want to restore it?
                # If we replace x with y where y is in set, we lose x. Size decreases.
                # If we want fixed size, we should only pick y not in set.
                # For now, let's allow size to fluctuate slightly, or force fixed size.
                # Fixed size is easier to reason about for the ratio.
                # Let's try to maintain size n.
                if len(neighbor) != len(current_u):
                    # Try to add a random element to restore size
                    while len(neighbor) < len(current_u):
                        candidate = random.randint(0, max_val)
                        if candidate not in neighbor:
                            neighbor.append(candidate)
                            break
                    # If still too small, revert?
                    if len(neighbor) != len(current_u):
                        continue
                    # If too big (shouldn't happen with replace), remove random non-zero
                    if len(neighbor) > len(current_u):
                        # Remove a random non-zero element
                        non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                        if non_zero_indices:
                            rm_idx = random.choice(non_zero_indices)
                            neighbor.pop(rm_idx)
                            neighbor = sorted(neighbor)
            else:
                continue
        else:
            # Perturb: add or remove an element
            if move_type < 0.85:
                # Add
                candidate = random.randint(0, max_val)
                if candidate not in neighbor:
                    neighbor.append(candidate)
                    neighbor = sorted(neighbor)
            else:
                # Remove
                if len(neighbor) > 1:
                    non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                    if non_zero_indices:
                        rm_idx = random.choice(non_zero_indices)
                        neighbor.pop(rm_idx)
                        neighbor = sorted(neighbor)
                else:
                    continue
        
        if not neighbor:
            continue
            
        neighbor_score = _score(neighbor)
        
        # Acceptance probability
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-9)):
            current_u = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_u = list(current_u)
                best_score = current_score
        
        temperature *= cooling
        
    return best_u, best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    best_u = [0, 1, 2]
    best_s = _score(best_u)
    
    # We will explore different sizes n and max values M.
    # The score is 1 + log(ratio) / log(2M+1).
    # To maximize, we want high ratio and small M? Or high ratio even if M is large?
    # log(2M+1) grows slowly.
    # If ratio is close to 1, score is close to 1.
    # If ratio > 1, score > 1.
    # Max possible ratio for a set of size n is n^2 (if all differences distinct) / 1 (if all sums distinct? No).
    # |U-U| <= n^2. |U+U| >= n (if 0 in U).
    # Actually, for a Sidon set, |U+U| ~ n^2. For a Golomb ruler, |U-U| ~ n^2.
    # If U is a geometric progression, sums are distinct, differences are distinct.
    # Example: U = {0, 1, 2, 4, ...}.
    # Sums: all distinct? 0+0=0, 0+1=1, 0+2=2, 1+1=2 (collision!).
    # So sums are not always distinct.
    # Differences: 1-0=1, 2-1=1 (collision!).
    
    # Let's try to find sets where |U-U| is large relative to |U+U|.
    # This happens if U is "sum-free" like? No, sum-free means no a+b=c.
    # We want many differences and few sums.
    # Or many differences and comparable sums.
    
    # Strategy:
    # 1. Start with small sets and grow them.
    # 2. Use SA to optimize specific configurations.
    
    # Phase 1: Explore small to medium sizes
    # n from 3 to 50
    # M from n to 1000
    
    # We need to be careful with time. FFT is fast, but SA iterations must be limited.
    
    # Let's define a grid of (n, M) to explore.
    # For each (n, M), run a short SA.
    
    # But M affects the score denominator.
    # Small M -> small denominator -> larger score for same ratio.
    # But small M restricts the set.
    
    # Let's try a few specific structural candidates first.
    # 1. Geometric progression: 0, 1, B, B^2, ...
    # 2. Random sets with small M.
    
    # We'll iterate through a few (n, M) pairs.
    
    # To save time, we'll run SA for a fixed number of iterations per pair.
    # And we'll update the global best.
    
    # List of (n, M) to try
    # We want to cover a range of M.
    # If M is very large, log(2M+1) is large, so the log(ratio) term is diluted.
    # Unless ratio is huge.
    # Max ratio is roughly n^2 / (n^2/2) = 2? No.
    # |U-U| max n^2. |U+U| min n. Ratio max n.
    # log(n) / log(2M+1).
    # If M is small, say M=n, ratio ~ n, log(n)/log(2n) ~ 0.5. Score ~ 1.5.
    # If M is large, say M=n^2, ratio ~ n, log(n)/log(n^2) ~ 0.5. Score ~ 1.5.
    # So M doesn't matter as much as the ratio?
    # Wait, if M is small, we can't have large n.
    # If M=10, max n=11.
    # If M=1000, max n=1001.
    
    # Let's try to maximize the ratio.
    # Sets with many differences and few sums.
    # "Sum-free" sets have no a+b=c, but that doesn't minimize |U+U| directly.
    # Minimizing |U+U| means many collisions in sums.
    # This happens if U is an arithmetic progression?
    # AP: {0, 1, ..., n-1}.
    # Sums: {0, ..., 2n-2}. Size 2n-1.
    # Diffs: {-(n-1), ..., n-1}. Size 2n-1.
    # Ratio 1. Score 1.
    
    # We want ratio > 1.
    # So |U-U| > |U+U|.
    # This implies more collisions in sums than in differences?
    # No, |U-U| is the size of the set.
    # If |U-U| > |U+U|, then the difference set is "larger" (less dense) than the sum set?
    # Or just larger cardinality.
    
    # Let's try random sets and see.
    
    # We will run a loop of SA phases.
    # Each phase picks a random n and M.
    
    num_phases = 100 # Max number of SA runs
    phase_iter = 500 # Iterations per SA
    
    # To ensure we finish in time, we check time in the loop.
    
    for phase in range(num_phases):
        if time.time() >= _DEADLINE - 10.0:
            break
            
        # Choose n and M
        # Bias towards small M to keep denominator small?
        # Or bias towards n that allows good structure.
        
        n = random.randint(3, 60)
        # M should be at least n-1.
        # Let's pick M in [n, 10 * n]
        M = random.randint(n, 10 * n)
        
        # Initial candidate
        u_init = _random_initial_set(n, M)
        
        # Run SA
        u_best, s_best = _local_search(u_init, M, phase_iter, temperature_start=1.0, cooling=0.99)
        
        if s_best > best_s:
            best_s = s_best
            best_u = u_best
            
            # If we found a good one, maybe try to polish it with more iterations
            # But we have a time limit.
            
    # Final polish on the best found
    if time.time() < _DEADLINE - 5.0:
        m_best = max(best_u)
        # Increase M slightly to allow growth? Or keep M?
        # Keeping M fixed is safer.
        u_polished, s_polished = _local_search(best_u, m_best, 1000, temperature_start=0.5, cooling=0.995)
        if s_polished > best_s:
            best_s = s_polished
            best_u = u_polished
            
    # Ensure 0 is in best_u and all are non-negative integers
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
