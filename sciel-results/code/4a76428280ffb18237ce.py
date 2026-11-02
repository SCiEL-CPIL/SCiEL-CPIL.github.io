"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using FFT (autocorrelation approach)
    # The autocorrelation of the indicator gives the difference set
    diff_len = 2 * max_u + 1
    fft_len2 = _next_pow2(2 * diff_len)
    
    # Shift indicator so that differences range from 0 to 2*max_u
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    freq_d = fft(diff_vec)
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    # The autocorrelation peaks at lag 0, and the difference set
    # corresponds to non-zero entries in diff_result[max_u-2*max_u : max_u+2*max_u+1]
    # Actually, diff_result[k] for k in [0, fft_len2) gives count of pairs (a,b) with a-b = k - max_u
    # So differences range from -max_u to max_u, stored at indices 0 to 2*max_u
    diffset_size = int(np.sum(np.abs(diff_result[:diff_len]) > 1e-9))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_exact(U):
    """Exact O(n^2) score computation for small sets (high precision)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    if n <= 25:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _prime_step_construction(num_steps, max_prime=None):
    """Construct U using cumulative sums of distinct primes for additive independence."""
    if max_prime is None:
        max_prime = 500
    
    # Generate primes up to max_prime
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    # Greedy selection: at each step, pick the prime that maximizes score
    U = [0]
    current = 0
    best_score = _score_set(U)
    remaining_primes = set(primes)
    
    for step in range(num_steps):
        best_p = None
        best_s = best_score
        # Evaluate all remaining primes if few, otherwise a larger random subset
        candidates = list(remaining_primes)
        if len(candidates) <= 80:
            # Try all
            pass
        else:
            random.shuffle(candidates)
            candidates = candidates[:min(80, len(candidates))]
        
        # Sort by heuristic: prefer primes that are coprime to current differences
        # and spread out from existing elements
        def prime_heuristic(p):
            new_current = current + p
            # Score: prefer larger primes (more spread) but not too large
            spread_score = math.log(p + 1)
            # Penalize if new_current is close to existing elements
            min_dist = min(abs(new_current - u) for u in U) if U else p
            dist_score = math.log(min_dist + 1)
            return spread_score + dist_score
        
        candidates.sort(key=prime_heuristic, reverse=True)
        # Mix: take top by heuristic + random sample
        top_candidates = candidates[:min(30, len(candidates))]
        if len(candidates) > 30:
            random.shuffle(candidates[30:])
            top_candidates += candidates[30:50]
        
        for p in top_candidates:
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_p = p
        
        if best_p is not None:
            current += best_p
            U.append(current)
            best_score = best_s
            remaining_primes.discard(best_p)
        else:
            break
    
    return sorted(U)

def _greedy_prime_selection(U_base, max_prime=300, max_steps=50):
    """Greedy selection of prime steps to maximize score incrementally."""
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    U = [0]
    current = 0
    best_score = _score_set(U)
    
    used_primes = set()
    remaining = list(primes)
    
    for step in range(max_steps):
        best_p = None
        best_s = best_score
        # Try a larger subset of remaining primes with heuristic ordering
        candidates = [p for p in remaining if p not in used_primes]
        
        def prime_heuristic(p):
            new_current = current + p
            spread_score = math.log(p + 1)
            min_dist = min(abs(new_current - u) for u in U) if U else p
            dist_score = math.log(min_dist + 1)
            return spread_score + dist_score
        
        candidates.sort(key=prime_heuristic, reverse=True)
        top_candidates = candidates[:min(60, len(candidates))]
        if len(candidates) > 60:
            random.shuffle(candidates[60:])
            top_candidates += candidates[60:90]
        
        for p in top_candidates:
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_p = p
        
        if best_p is not None:
            current += best_p
            U.append(current)
            best_score = best_s
            used_primes.add(best_p)
        else:
            break
    
    return sorted(U)

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements, two-swap moves)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 500
    
    no_improve_count = 0
    max_no_improve = 80
    
    # Precompute a fast scoring function for the current max_u range
    # Use a fixed FFT size for speed during local search
    search_fft_len = _next_pow2(2 * (upper_bound + 1) * 2)
    
    def _fast_score(new_U):
        """Fast score using precomputed FFT size."""
        if not new_U or 0 not in new_U:
            return 0.0
        new_U = sorted(set(new_U))
        n = len(new_U)
        if n == 0:
            return 0.0
        mu = max(new_U)
        if mu == 0:
            return 1.0
        
        sum_len = 2 * mu + 1
        if sum_len > search_fft_len // 2:
            return _score_set_fft(new_U)
        
        indicator = np.zeros(search_fft_len, dtype=np.float64)
        for u in new_U:
            indicator[u] = 1.0
        
        freq = fft(indicator)
        conv = ifft(freq * freq).real
        sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
        
        # Difference set via autocorrelation
        diff_len = 2 * mu + 1
        diff_vec = np.zeros(search_fft_len, dtype=np.float64)
        for u in new_U:
            diff_vec[u + mu] = 1.0
        freq_d = fft(diff_vec)
        diff_result = ifft(freq_d * np.conj(freq_d)).real
        diffset_size = int(np.sum(np.abs(diff_result[:diff_len]) > 1e-9))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * mu + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            # Local neighborhood - tighter for speed
            for offset in range(-40, 41):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Some random candidates
            for _ in range(20):
                candidates.add(random.randint(1, upper_bound))
            # Candidates near other elements (to create structured patterns)
            for j in range(len(U)):
                if i != j:
                    candidates.add(max(1, U[j] + random.randint(-15, 15)))
                    candidates.add(max(1, 2 * U[j] - old_val + random.randint(-10, 10)))
            
            for new_val in sorted(candidates, key=lambda x: abs(x - old_val)):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            
            # Try two-swap move: change two elements simultaneously
            if len(U) >= 3 and no_improve_count < max_no_improve // 2:
                i1, i2 = random.sample(range(1, len(U)), 2)
                for _ in range(15):
                    nv1 = max(1, U[i1] + random.randint(-30, 30))
                    nv2 = max(1, U[i2] + random.randint(-30, 30))
                    if nv1 == U[i1] and nv2 == U[i2]:
                        continue
                    new_U = U[:]
                    new_U[i1] = nv1
                    new_U[i2] = nv2
                    new_U_set = set(new_U)
                    if len(new_U_set) != len(new_U):
                        continue
                    s = _fast_score(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved and len(U) < 300:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _fast_score(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _fast_score(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            if random.random() < 0.3:
                new_U = U[:]
                num_changes = random.randint(2, min(12, len(new_U) - 1))
                for _ in range(num_changes):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _fast_score(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
            
            if not improved and no_improve_count >= max_no_improve:
                break
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    """High-precision polish pass using exact scoring for small sets."""
    U = sorted(set(U))
    n = len(U)
    if n > 60:
        return _local_search(U, max_iters=3000, deadline=deadline)
    
    best_score = _score_set_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200
    
    no_improve = 0
    max_no_improve = 50
    
    for it in range(8000):
        if deadline and time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(1, len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            for offset in range(-35, 36):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_exact(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if len(U) < 80:
                for _ in range(35):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 3:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i+1:]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            # Add a small random perturbation to escape local optima
            if not improved and no_improve % 10 == 0 and no_improve < max_no_improve:
                for _ in range(5):
                    idx = random.randint(1, len(U) - 1)
                    U[idx] = max(1, U[idx] + random.randint(-20, 20))
                    U = sorted(set(U))
                    if 0 not in U:
                        U = [0] + U
                    s = _score_set_exact(U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = U[:]
                        improved = True
                        no_improve = 0
                        break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    random.seed(42)
    
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 40):
        if time.time() > deadline - 40:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Prime-step constructions with greedy selection (more varied parameters)
    for num_steps in [5, 8, 12, 15, 20, 25, 30, 35, 40, 45]:
        if time.time() > deadline - 40:
            break
        for max_p in [100, 200, 300, 500, 800, 1200]:
            if time.time() > deadline - 40:
                break
            ps = _prime_step_construction(num_steps, max_prime=max_p)
            candidates.append(ps)
    
    # Greedy prime selection (improved, more trials)
    for _ in range(20):
        if time.time() > deadline - 40:
            break
        for mp in [200, 400, 600]:
            if time.time() > deadline - 40:
                break
            gp = _greedy_prime_selection([0], max_prime=mp, max_steps=45)
            candidates.append(gp)
    
    # Random sets with various sizes
    for trial in range(50):
        if time.time() > deadline - 40:
            break
        n = random.randint(3, 60)
        max_val = random.randint(30, 1500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U[:min(len(U), 30)])
    
    # Powers-based sets
    for p in [2, 3, 4, 5]:
        if time.time() > deadline - 40:
            break
        U = [0]
        for i in range(1, 25):
            U.append(p ** i)
            if U[-1] > 50000:
                break
        candidates.append(U)
    
    # Sparse random sets
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        n = random.randint(5, 50)
        max_val = random.randint(100, 3000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Hybrid: start with Golomb ruler prefix, extend with primes
    for n_prefix in [5, 8, 10, 12]:
        if time.time() > deadline - 40:
            break
        gr = _golomb_ruler(n_prefix)
        current = gr[-1]
        U = gr[:]
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            for i in range(3, int(math.sqrt(x)) + 1, 2):
                if x % i == 0:
                    return False
            return True
        primes = [p for p in range(2, 400) if is_prime(p)]
        for p in primes[:20]:
            if time.time() > deadline - 40:
                break
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > _score_set(U):
                U = new_U
                current = new_current
        candidates.append(U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate
    if time.time() < deadline - 20:
        best_U, best_score = _local_search(best_U, max_iters=15000, deadline=deadline - 20)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:15]:
        if time.time() > deadline - 20:
            break
        if U == best_U:
            continue
        if s > best_score - 0.05:
            improved_U, improved_score = _local_search(U, max_iters=5000, deadline=deadline - 20)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final high-precision polish pass
    if time.time() < deadline - 5:
        best_U, best_score = _polish(best_U, deadline=deadline - 2)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
