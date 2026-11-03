"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    
    # Use FFT for efficient sum/diff counting when max_u is not too large
    if max_u <= 200000:
        # Sum set: convolution of indicator with itself
        ind = np.zeros(max_u + 1, dtype=np.float64)
        ind[arr] = 1.0
        # Sum set size = number of positions where conv(ind, ind) > 0
        conv_sum = np.convolve(ind, ind, mode='full')
        num_sums = np.count_nonzero(conv_sum)
        
        # Difference set: |U-U| = number of distinct (a-b) for a,b in U
        # This equals the number of nonzero autocorrelation lags plus center
        # Use FFT-based autocorrelation
        n_fft = 1
        while n_fft < 2 * max_u + 1:
            n_fft *= 2
        fft_ind = np.fft.rfft(ind, n=n_fft)
        autocorr = np.fft.irfft(fft_ind * np.conj(fft_ind), n=n_fft)
        # autocorr[k] for k=0..max_u gives count of pairs with diff = k
        # Also negative diffs: autocorr is symmetric, so |U-U| = 1 + 2*sum(autocorr[1:max_u+1] > 0)
        num_diffs = 1
        for k in range(1, max_u + 1):
            if autocorr[k] > 0.5:  # count_nonzero with tolerance
                num_diffs += 2
        # More efficient: count nonzero in autocorr[1:max_u+1]
        num_nonzero_pos = np.count_nonzero(autocorr[1:max_u+1] > 0.5)
        num_diffs = 1 + 2 * num_nonzero_pos
    else:
        # Fallback to direct computation for very large max_u
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

    # Precompute primes up to a reasonable bound
    def _sieve(n):
        if n < 2:
            return []
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i in range(2, n+1) if is_p[i]]
    
    small_primes = _sieve(200)

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        # Try adding a new element - focus on values that create new difference patterns
        candidates = []
        
        # Values that are far apart to maximize difference set
        for multiplier in [2, 3, 5, 7, 11, 13]:
            for delta in range(1, 15):
                candidates.append(max_val * multiplier + delta)
        
        # Prime-like steps from existing elements
        for a in current_U:
            for p in small_primes[:20]:
                candidates.append(a + p)
                if a + 2*p <= max_val * 3:
                    candidates.append(a + 2 * p)
        
        # Values near existing elements (to create dense local structure)
        for a in current_U:
            for d in [1, 2, 3, 5, 8, 13]:
                if a + d > 0 and a + d not in cur_set:
                    candidates.append(a + d)
                if a - d > 0 and a - d not in cur_set:
                    candidates.append(a - d)
        
        # Some interior values (sparse sampling)
        if max_val > 10:
            step_size = max(1, max_val // 80)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        
        # Random exploration with wider range
        for _ in range(40):
            candidates.append(random.randint(1, max_val * 6))
        
        # Remove duplicates and invalid values
        candidates = list(set([v for v in candidates if v > 0 and v not in cur_set]))
        random.shuffle(candidates)

        for new_val in candidates[:40]:
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
                for idx in remove_candidates[:20]:
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
            if stall_count > 30:
                # Perturb: replace a random element with a new value
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 6)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_set(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 200:
                break

    return best_U, best_score

def _smooth_polish(U, max_time=30):
    """Smooth surrogate polish: parameterize set as continuous variables,
    use Nelder-Mead on a smoothed score, then round."""
    from scipy.optimize import minimize
    
    U = sorted(U)
    n = len(U)
    if n < 2:
        return U, _score_set(U)
    
    start = time.time()
    best_U = list(U)
    best_score = _score_set(best_U)
    
    # Parameterize: x[0] = 0 (fixed), x[1..n-1] are the other elements
    # Use log-scale for better conditioning
    x0 = np.zeros(n)
    for i in range(1, n):
        x0[i] = math.log(U[i] + 1)
    
    def smooth_score(params):
        """Evaluate score with a smoothed indicator (sigmoid-based)."""
        if time.time() - start > max_time:
            return 0.0
        # Convert log-params to values
        vals = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            vals[i] = math.exp(params[i]) - 1.0
        # Round to integers
        int_vals = np.round(vals).astype(np.int64)
        # Ensure non-negative and distinct
        int_vals = np.sort(int_vals)
        if int_vals[0] != 0:
            int_vals[0] = 0
        # Check distinctness
        if len(np.unique(int_vals)) < n:
            return 0.0
        U_candidate = int_vals.tolist()
        if 0 not in U_candidate:
            return 0.0
        return _score_set(U_candidate)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            lambda x: -smooth_score(x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.001}
        )
        if time.time() - start < max_time:
            best_params = result.x
            vals = np.zeros(n, dtype=np.float64)
            for i in range(1, n):
                vals[i] = math.exp(best_params[i]) - 1.0
            int_vals = np.sort(np.round(vals).astype(np.int64))
            if int_vals[0] != 0:
                int_vals[0] = 0
            if len(np.unique(int_vals)) == n:
                U_candidate = int_vals.tolist()
                s = _score_set(U_candidate)
                if s > best_score:
                    best_score = s
                    best_U = U_candidate
    except Exception:
        pass
    
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

    # Try small sets {0,1,a,b}, {0,1,a,b,c}
    for a in range(2, 25):
        if _elapsed() > _DEADLINE - 40:
            break
        for b in range(a + 1, 150):
            if _elapsed() > _DEADLINE - 40:
                break
            U = [0, 1, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    for a in range(2, 15):
        if _elapsed() > _DEADLINE - 25:
            break
        for b in range(a + 1, 80):
            if _elapsed() > _DEADLINE - 25:
                break
            for c in range(b + 1, 300):
                if _elapsed() > _DEADLINE - 25:
                    break
                U = [0, 1, a, b, c]
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

    # Local search from best (more steps thanks to faster scoring)
    if _elapsed() < _DEADLINE - 5:
        U, score = _local_search(best_U, max_steps=5000)
        if score > best_score:
            best_score = score
            best_U = U

    # Local search from Golomb seeds
    for n in [5, 6, 7, 8]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = _golomb_seed(n, n * n + n)
        if len(U) >= 4:
            U2, score2 = _local_search(U, max_steps=1500)
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
        U2, score2 = _local_search(U, max_steps=1500)
        if score2 > best_score:
            best_score = score2
            best_U = U2

    # Final polish: one more local search pass on the best
    if _elapsed() < _DEADLINE - 2:
        U, score = _local_search(best_U, max_steps=2000)
        if score > best_score:
            best_score = score
            best_U = U

    # Smooth optimization polish
    if _elapsed() < _DEADLINE - 10:
        U_smooth, score_smooth = _smooth_polish(best_U, max_time=25)
        if score_smooth > best_score:
            best_score = score_smooth
            best_U = U_smooth
        # Local search from smooth result
        if _elapsed() < _DEADLINE - 3:
            U2, score2 = _local_search(best_U, max_steps=1000)
            if score2 > best_score:
                best_score = score2
                best_U = U2

    # Final prime-step refinement if time permits
    if _elapsed() < _DEADLINE - 3:
        U = _prime_step_construction(80, 100000)
        if len(U) >= 3:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            if _elapsed() < _DEADLINE - 1:
                U2, score2 = _local_search(U, max_steps=300)
                if score2 > best_score:
                    best_score = score2
                    best_U = U2

    return sorted(best_U)
