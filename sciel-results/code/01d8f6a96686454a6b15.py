import time
import math
import random
import numpy as np

_START = time.time()
# 120s limit for evaluation, but evaluator kills at 150s. 
# We aim to finish well before 120s to be safe.
_DEADLINE = _START + 110.0

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses FFT for efficient computation of sumset and difference set sizes.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # FFT size must be large enough to hold the max sum (2*m) and avoid circular convolution artifacts.
    # Max sum is 2*m. We need size > 2*m. Next power of 2 is efficient.
    # Note: For differences, we shift by m to make them non-negative.
    # U+U range: [0, 2*m]
    # U-U range: [-m, m] -> shifted to [0, 2*m]
    
    # Optimization: If n is small, direct calculation is faster due to overhead.
    # But for n > 10, FFT is generally better or comparable.
    # Let's use FFT for n > 5.
    
    if n <= 5:
        sums = u_arr[:, None] + u_arr[None, :]
        diffs = u_arr[:, None] - u_arr[None, :]
        unique_sums = np.unique(sums)
        unique_diffs = np.unique(diffs)
        if len(unique_sums) == 0:
            return 0.0
        ratio = len(unique_diffs) / float(len(unique_sums))
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * m + 1)

    # FFT approach
    # We need a vector v of size L such that v[x] = 1 if x in U, else 0.
    # Convolution v * v gives counts of sums.
    # For differences, we can use v_reversed * v or similar.
    # Actually, |U-U| is the support of the difference set.
    # Let's construct a vector for U.
    
    L = 1
    while L <= 2 * m:
        L *= 2
        
    v = np.zeros(L, dtype=np.float64)
    v[u_arr] = 1.0
    
    # Sumset: conv(v, v)
    # Difference set: conv(v, v_flipped) where v_flipped[i] = v[L-1-i] ?
    # Or simpler: The difference set U-U is the support of the autocorrelation?
    # No, autocorrelation gives counts of pairs with specific difference.
    # Let's just compute the sets directly if L is small, or use FFT for counts.
    # Wait, np.unique on the result of FFT is tricky because of floating point errors.
    # However, we only need the *count* of non-zero elements.
    
    # Let's stick to a hybrid:
    # If L is not too big (e.g., < 10000), we can just use the direct method but with numpy unique which is C-optimized.
    # The bottleneck in the seed was likely the python loops in local search, not the scoring itself for small n.
    # But the lesson says FFT enables larger sets.
    
    # Let's try FFT for counting non-zeros.
    f = np.fft.fft(v)
    f2 = f * f
    c = np.fft.ifft(f2).real
    
    # The sumset support is where c > 0.5 (since counts are integers)
    # But we must be careful with floating point.
    # Also, we need the difference set.
    # Difference set U-U is the support of the set {a-b | a,b in U}.
    # This is equivalent to the support of the convolution of v and v_flipped?
    # Let v_flip be v reversed.
    # The cross-correlation of v and v gives the difference counts.
    
    # Actually, for small n (which we expect in SA to keep max(U) manageable for high score), 
    # direct calculation is robust and fast enough if we vectorize it.
    # Let's revert to vectorized direct calculation but optimized.
    
    # The previous seed used:
    # sums = u_arr[:, None] + u_arr[None, :]
    # This creates an n x n matrix. For n=50, that's 2500 elements. Very fast.
    # The issue is if n gets large (100+), n^2 becomes 10k+. Still fast in numpy.
    # The real gain is exploring LARGER n.
    
    # Let's use the direct method but ensure it's as fast as possible.
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    # np.unique is O(N log N) where N is n^2.
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _initial_random_set(n, max_val):
    """Generate a random sparse set."""
    rng = random.Random(int(time.time() * 1000) % 10000)
    if n <= 1:
        return [0]
    
    # Ensure 0 is included
    vals = {0}
    current = 0
    # Use geometric distribution for gaps to keep numbers sparse but cover a range
    # This helps in maximizing |U-U| without making max(U) too large
    for _ in range(n - 1):
        # Gap size roughly proportional to current value or fixed
        gap = rng.randint(1, max(1, int(max_val ** 0.5)))
        current += gap
        if current > max_val:
            current = max_val
        vals.add(current)
    
    return sorted(list(vals))

def construct():
    global _DEADLINE
    
    rng = random.Random(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = _score_from_set(best_u)
    
    # Simulated Annealing
    # State: list of integers
    # We will vary the size n as well.
    
    # Start with a few random initializations
    initial_configs = []
    for n in [10, 20, 30, 50]:
        for max_val in [100, 1000, 10000, 100000]:
            u = _initial_random_set(n, max_val)
            s = _score_from_set(u)
            initial_configs.append((s, u))
            if s > best_s:
                best_s = s
                best_u = u[:]
                
    # Sort initial configs by score
    initial_configs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top few to start SA
    start_u = initial_configs[0][1]
    current_u = start_u[:]
    current_s = initial_configs[0][0]
    
    if current_s > best_s:
        best_s = current_s
        best_u = current_u[:]
        
    # SA Parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    steps = 0
    max_steps = 20000
    
    while time.time() < _DEADLINE - 5.0 and steps < max_steps:
        steps += 1
        
        # Generate a neighbor
        # Moves:
        # 1. Add a random element
        # 2. Remove a random element (keep 0)
        # 3. Replace a random element with a nearby value
        # 4. Scale the whole set (multiply by small factor) - risky, might break 0? No, 0 stays 0.
        
        move_type = rng.random()
        new_u = current_u[:]
        
        if move_type < 0.3:
            # Add element
            if len(new_u) < 100: # Cap size to keep n^2 scoring fast
                max_val = max(new_u)
                # Try to add something in the range [1, max_val * 2] or sparse
                new_val = rng.randint(1, max(1, max_val * 2))
                if new_val not in new_u:
                    new_u.append(new_val)
                    new_u = sorted(set(new_u))
                else:
                    continue
        elif move_type < 0.6:
            # Remove element
            if len(new_u) > 2:
                # Don't remove 0
                idx = rng.randint(1, len(new_u) - 1)
                new_u.pop(idx)
                new_u = sorted(set(new_u))
            else:
                continue
        elif move_type < 0.9:
            # Replace element
            if len(new_u) > 1:
                idx = rng.randint(1, len(new_u) - 1)
                old_val = new_u[idx]
                # Perturb
                delta = rng.randint(-10, 10)
                new_val = old_val + delta
                if new_val < 0:
                    new_val = old_val + rng.randint(1, 5)
                if new_val != old_val:
                    if new_val in new_u:
                        continue
                    new_u[idx] = new_val
                    new_u = sorted(set(new_u))
                else:
                    continue
        else:
            # Scale
            # Multiply all non-zero elements by a small factor
            # This changes max(U) significantly
            factor = rng.choice([0.5, 2.0, 1.1, 0.9])
            new_u_scaled = [0]
            for x in new_u:
                if x != 0:
                    new_u_scaled.append(int(x * factor))
            # Ensure uniqueness and non-negative
            new_u = sorted(set([x for x in new_u_scaled if x >= 0]))
            if len(new_u) < 2:
                continue

        new_s = _score_from_set(new_u)
        
        # Acceptance criterion
        delta_s = new_s - current_s
        if delta_s > 0:
            # Always accept improvement
            current_u = new_u
            current_s = new_s
            if current_s > best_s:
                best_s = current_s
                best_u = current_u[:]
        else:
            # Accept worse with probability exp(-delta_s / T)
            # Note: delta_s is negative, so -delta_s is positive
            prob = math.exp(-delta_s / T)
            if rng.random() < prob:
                current_u = new_u
                current_s = new_s
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            # Restart with a new random config if we get stuck
            # Pick a random initial config
            init_s, init_u = initial_configs[rng.randint(0, len(initial_configs)-1)]
            current_u = init_u[:]
            current_s = init_s
            T = 1.0
            
    # Final check
    if best_s == 0.0:
        return [0, 1]
        
    return best_u

if __name__ == "__main__":
    # Quick test
    u = construct()
    s = _score_from_set(u)
    print(f"Result: {u[:10]}... Size: {len(u)}, Score: {s}")
