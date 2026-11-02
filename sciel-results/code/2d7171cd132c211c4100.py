"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _next_pow2(n):
    """Return the next power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def _score_set_fft(U):
    """Compute score using FFT for larger sets."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    
    # For small sets, use direct computation
    if n <= 20:
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        num_sums = len(np.unique(sums))
        num_diffs = len(np.unique(diffs))
    else:
        # Use FFT-based convolution for sums
        # U+U: convolve indicator of U with itself
        size = 2 * max_u + 1
        fft_size = _next_pow2(2 * size)
        f = np.zeros(fft_size, dtype=np.complex128)
        for u in U:
            f[u] = 1.0
        F = np.fft.fft(f)
        # Sumset: autocorrelation-like
        sum_conv = np.fft.ifft(F * F).real
        # Difference set: convolve U with reversed U
        # U-U = {a-b : a,b in U}
        # This is the support of f * f_reversed where f_reversed[i] = f[-i] mod something
        # Actually, U-U has range [-max_u, max_u], so size 2*max_u+1
        # Let's compute: for each d in [-max_u, max_u], count pairs (a,b) with a-b=d
        # This is cross-correlation
        # Use: ifft(fft(f) * conj(fft(f_reversed)))... 
        # Simpler: direct for moderate sizes, FFT for large
        if n <= 100:
            arr = np.array(U, dtype=np.int64)
            diffs = arr[:, None] - arr[None, :]
            num_diffs = len(np.unique(diffs))
            # For sums, use FFT
            num_sums = int(np.sum(sum_conv > 0.5))
        else:
            arr = np.array(U, dtype=np.int64)
            diffs = arr[:, None] - arr[None, :]
            num_diffs = len(np.unique(diffs))
            num_sums = int(np.sum(sum_conv > 0.5))
    
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    num_sums = len(np.unique(sums))
    num_diffs = len(np.unique(diffs))
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _golomb_seed(n, max_val):
    """Greedy Golomb ruler: pick n elements with distinct pairwise differences."""
    U = [0]
    diffs = set()
    for _ in range(1, n):
        best_v = None
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_v = v
                break
        if best_v is None:
            break
        U.append(best_v)
        for a in U:
            diffs.add(abs(best_v - a))
    return U

def _prime_step_construction(max_primes=40, max_val=50000):
    """Greedy construction using prime-step gaps to exploit additive independence."""
    # Generate primes up to max_val
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(max_val)
    if len(primes) < 2:
        return [0]
    
    U = [0]
    current_sum = 0
    best_score = _score_set(U)
    
    # Use larger prime steps to create more spread
    # Try different strategies: consecutive primes, every other prime, etc.
    for strategy in [0, 1, 2]:  # 0=consecutive, 1=every other, 2=every third
        if _elapsed() > _DEADLINE - 20:
            break
        U_temp = [0]
        current_sum_temp = 0
        score_temp = best_score
        primes_to_use = primes[1::(strategy+1)]
        
        for p in primes_to_use:
            if _elapsed() > _DEADLINE - 20:
                break
            new_val = current_sum_temp + p
            if new_val > max_val:
                continue
            candidate = U_temp + [new_val]
            s = _score_set(candidate)
            if s > score_temp + 1e-10:
                U_temp = candidate
                current_sum_temp = new_val
                score_temp = s
        
        if score_temp > best_score:
            best_score = score_temp
            U = U_temp
    
    return U

def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        # Try adding a new element - focus on values that create new difference patterns
        candidates = []
        
        # Values that are far apart to maximize difference set
        for multiplier in [2, 3, 5, 7, 11]:
            for delta in range(1, 20):
                candidates.append(max_val * multiplier + delta)
        
        # Prime-like steps from existing elements
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        for a in current_U:
            for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                if is_prime(p):
                    candidates.append(a + p)
                    candidates.append(a + p * 2)
        
        # Some interior values (sparse sampling)
        if max_val > 10:
            step_size = max(1, max_val // 50)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        
        # Random exploration
        for _ in range(30):
            candidates.append(random.randint(1, max_val * 5))
        
        # Remove duplicates and invalid values
        candidates = list(set([v for v in candidates if v > 0 and v not in cur_set]))
        random.shuffle(candidates)

        for new_val in candidates[:30]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break

        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:15]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break

        if not improved:
            stall_count += 1
            if stall_count > 40:
                # Perturb: replace a random element with a new value
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 5)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_set(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 150:
                break

    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Try geometric progressions
    for base in range(2, 30):
        if _elapsed() > _DEADLINE - 60:
            break
        for n in range(2, 12):
            if _elapsed() > _DEADLINE - 60:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= base
                U.append(val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # Exhaustive search for small sets (size 4-6) with bounded max element
    # Based on cell findings: exhaustive search up to size 6, max 40 improves score
    import itertools
    
    # Size 4: {0, a, b, c} with 0 < a < b < c <= 40
    for a in range(1, 40):
        if _elapsed() > _DEADLINE - 35:
            break
        for b in range(a + 1, 40):
            if _elapsed() > _DEADLINE - 35:
                break
            for c in range(b + 1, 41):
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Size 5: {0, a, b, c, d} with 0 < a < b < c < d <= 40
    for a in range(1, 30):
        if _elapsed() > _DEADLINE - 30:
            break
        for b in range(a + 1, 35):
            if _elapsed() > _DEADLINE - 30:
                break
            for c in range(b + 1, 40):
                if _elapsed() > _DEADLINE - 30:
                    break
                for d in range(c + 1, 41):
                    U = [0, a, b, c, d]
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
    
    # Size 6: {0, a, b, c, d, e} with 0 < a < b < c < d < e <= 30
    for a in range(1, 20):
        if _elapsed() > _DEADLINE - 25:
            break
        for b in range(a + 1, 25):
            if _elapsed() > _DEADLINE - 25:
                break
            for c in range(b + 1, 30):
                if _elapsed() > _DEADLINE - 25:
                    break
                for d in range(c + 1, 30):
                    if _elapsed() > _DEADLINE - 25:
                        break
                    for e in range(d + 1, 31):
                        U = [0, a, b, c, d, e]
                        s = _score_set(U)
                        if s > best_score:
                            best_score = s
                            best_U = U[:]

    # Try Golomb rulers as seeds
    for n in range(4, 10):
        if _elapsed() > _DEADLINE - 20:
            break
        for max_val in [n * n, n * n + n, n * n * 2]:
            if _elapsed() > _DEADLINE - 20:
                break
            U = _golomb_seed(n, max_val)
            if len(U) >= n - 1:
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # Try prime-step construction
    if _elapsed() < _DEADLINE - 40:
        for max_primes in [20, 30, 50, 80]:
            if _elapsed() > _DEADLINE - 40:
                break
            for max_val in [10000, 30000, 80000, 150000]:
                if _elapsed() > _DEADLINE - 40:
                    break
                U = _prime_step_construction(max_primes, max_val)
                if len(U) >= 3:
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
                    # Local search from prime construction
                    if _elapsed() < _DEADLINE - 10:
                        U2, score2 = _local_search(U, max_steps=800)
                        if score2 > best_score:
                            best_score = score2
                            best_U = U2

    # Local search from best
    if _elapsed() < _DEADLINE - 5:
        U, score = _local_search(best_U, max_steps=3000)
        if score > best_score:
            best_score = score
            best_U = U

    # Local search from Golomb seeds
    for n in [5, 6, 7, 8]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = _golomb_seed(n, n * n + n)
        if len(U) >= 4:
            U2, score2 = _local_search(U, max_steps=1000)
            if score2 > best_score:
                best_score = score2
                best_U = U2

    # Local search from geometric seeds
    for base in [2, 3, 4, 5, 7]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = [0]
        val = 1
        for i in range(1, 6):
            val *= base
            U.append(val)
        U2, score2 = _local_search(U, max_steps=1000)
        if score2 > best_score:
            best_score = score2
            best_U = U2

    # Final polish: one more local search pass on the best
    if _elapsed() < _DEADLINE - 2:
        U, score = _local_search(best_U, max_steps=1500)
        if score > best_score:
            best_score = score
            best_U = U

    return sorted(best_U)
