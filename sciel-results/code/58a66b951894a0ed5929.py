"""Model-and-optimize approach for sumset/difference-set exponent lower bound.
Combines prime-step sparse constructions with incremental scoring for local search."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy where beneficial for moderate sizes."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    arr = np.array(sorted(U_set), dtype=np.int64)
    if n <= 200:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_incremental(U, new_val, sumset_cache, diffset_cache):
    """Fast incremental score when adding one element to U.
    U is sorted list, new_val is the new element.
    Returns (score, updated_sumset, updated_diffset) or (None, None, None) if invalid."""
    U_set = set(U)
    U_set.add(new_val)
    n = len(U_set)
    max_u = max(U_set)
    
    # New sums: new_val + each existing element
    new_sums = set()
    for a in U:
        s = new_val + a
        new_sums.add(s)
    # New diffs: new_val - each existing, and each existing - new_val
    new_diffs = set()
    for a in U:
        new_diffs.add(new_val - a)
        new_diffs.add(a - new_val)
    
    # Count how many new sums/diffs are actually new
    new_sum_count = sum(1 for s in new_sums if s not in sumset_cache)
    new_diff_count = sum(1 for d in new_diffs if d not in diffset_cache)
    
    sumset_size = len(sumset_cache) + new_sum_count
    diffset_size = len(diffset_cache) + new_diff_count
    
    if sumset_size == 0:
        return None, None, None
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return None, None, None
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0, None, None
    score = 1.0 + log_ratio / log_denom
    return score, new_sums, new_diffs

def _construct_golomb(n):
    """Construct a Golomb ruler of order n."""
    if n <= 1:
        return [0]
    known = {
        1: [0], 2: [0, 1], 3: [0, 1, 3], 4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11], 6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22], 8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36], 10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
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

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_below(n):
    return [x for x in range(2, n) if _is_prime(x)]

def _construct_prime_step(n, primes=None, max_val=None):
    """Construct U = {0, p1, p1+p2, p1+p2+p3, ...} using distinct primes.
    Exploits additive independence of primes to minimize sumset collisions."""
    if primes is None:
        primes = _primes_below(500)
    if max_val is None:
        max_val = 5000
    
    U = [0]
    current = 0
    for i in range(n - 1):
        if i >= len(primes):
            break
        p = primes[i]
        new_val = current + p
        if new_val > max_val:
            break
        U.append(new_val)
        current = new_val
    return sorted(set(U))

def _construct_prime_step_greedy(n, max_val=5000, deadline=None):
    """Greedy prime-step construction: at each step, pick the prime that maximizes
    the incremental score contribution (minimizes new sum collisions)."""
    U = [0]
    current = 0
    primes = _primes_below(2000)
    
    for i in range(n - 1):
        if deadline and time.time() > deadline:
            break
        best_p = None
        best_score = -1.0
        
        # Compute current sumset and diffset
        U_set = set(U)
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        
        # Try candidate primes (prefer smaller primes for compactness, but check score)
        candidates = []
        for p in primes:
            new_val = current + p
            if new_val > max_val:
                break
            candidates.append((p, new_val))
        
        # Sample if too many candidates
        if len(candidates) > 30:
            candidates = candidates[:15] + random.sample(candidates[15:], 15)
        
        for p, new_val in candidates:
            # Count new sums and diffs
            new_sum_count = 0
            new_diff_count = 0
            for a in U:
                s = new_val + a
                if s not in sumset:
                    new_sum_count += 1
                d1 = new_val - a
                if d1 not in diffset:
                    new_diff_count += 1
                d2 = a - new_val
                if d2 not in diffset:
                    new_diff_count += 1
            
            sumset_size = len(sumset) + new_sum_count
            diffset_size = len(diffset) + new_diff_count
            if sumset_size == 0:
                continue
            ratio = diffset_size / sumset_size
            max_u = max(new_val, max(U))
            log_denom = math.log(2 * max_u + 1)
            if log_denom == 0:
                continue
            score = 1.0 + math.log(ratio) / log_denom
            if score > best_score:
                best_score = score
                best_p = new_val
        
        if best_p is not None:
            U.append(best_p)
            current = best_p
        else:
            break
    
    return sorted(set(U))

def _construct_mixed_radix(n, deadline=None):
    """Mixed-radix construction: U[i] = sum of digits in a mixed-radix system.
    This gives good additive independence properties."""
    # Use radices that are coprime or have few common factors
    radices = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    U = [0]
    current = 0
    for i in range(1, n):
        if deadline and time.time() > deadline:
            break
        digit = i % radices[i % len(radices)]
        current += digit * radices[i % len(radices)]
        U.append(current)
    return sorted(set(U))

def _local_search_fast(U, max_iters=5000, deadline=None, max_size=40):
    """Local search using incremental scoring for speed."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 200
    
    # Precompute sumset and diffset for incremental updates
    U_set = set(U)
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            candidates = []
            for delta in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            for _ in range(5):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = _score_set_fast(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    # Update caches
                    U_set = set(best_U)
                    sumset = set()
                    for a in U_set:
                        for b in U_set:
                            sumset.add(a + b)
                    diffset = set()
                    for a in U_set:
                        for b in U_set:
                            diffset.add(a - b)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element using incremental scoring
            if len(U) < max_size:
                U_set = set(U)
                best_add_score = best_score
                best_add_val = None
                # Sample candidate values
                add_candidates = set()
                for _ in range(20):
                    add_candidates.add(random.randint(1, upper_bound))
                for d in [1, 2, 3, 5, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47]:
                    add_candidates.add(d)
                    add_candidates.add(max_u + d)
                
                for new_val in add_candidates:
                    if new_val in U_set:
                        continue
                    # Incremental scoring
                    new_sum_count = 0
                    new_diff_count = 0
                    for a in U:
                        s = new_val + a
                        if s not in sumset:
                            new_sum_count += 1
                        d1 = new_val - a
                        if d1 not in diffset:
                            new_diff_count += 1
                        d2 = a - new_val
                        if d2 not in diffset:
                            new_diff_count += 1
                    
                    sumset_size = len(sumset) + new_sum_count
                    diffset_size = len(diffset) + new_diff_count
                    if sumset_size == 0:
                        continue
                    ratio = diffset_size / sumset_size
                    max_u_new = max(new_val, max_u)
                    log_denom = math.log(2 * max_u_new + 1)
                    if log_denom == 0:
                        continue
                    s = 1.0 + math.log(ratio) / log_denom
                    if s > best_add_score + 1e-9:
                        best_add_score = s
                        best_add_val = new_val
                
                if best_add_val is not None:
                    best_score = best_add_score
                    best_U = sorted(U + [best_add_val])
                    U_set = set(best_U)
                    sumset = set()
                    for a in U_set:
                        for b in U_set:
                            sumset.add(a + b)
                    diffset = set()
                    for a in U_set:
                        for b in U_set:
                            diffset.add(a - b)
                    improved = True
        
        if not improved:
            # Try removing an element
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        U_set = set(best_U)
                        sumset = set()
                        for a in U_set:
                            for b in U_set:
                                sumset.add(a + b)
                        diffset = set()
                        for a in U_set:
                            for b in U_set:
                                diffset.add(a - b)
                        improved = True
                        break
        
        if not improved and it > 200:
            upper_bound += 100
            U = best_U[:]
            U_set = set(U)
            sumset = set()
            for a in U_set:
                for b in U_set:
                    sumset.add(a + b)
            diffset = set()
            for a in U_set:
                for b in U_set:
                    diffset.add(a - b)
    
    return sorted(set(best_U)), best_score

def _optimize_parameters(start_time, deadline):
    """Optimize construction parameters directly against the objective."""
    best_score = -1.0
    best_U = [0]
    
    # Strategy 1: Golomb rulers
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _construct_golomb(n)
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Strategy 2: Prime-step constructions
    for n in range(3, 25):
        if time.time() > deadline:
            break
        ps = _construct_prime_step(n, max_val=5000)
        if len(ps) >= 2:
            s = _score_set(ps)
            if s > best_score:
                best_score = s
                best_U = ps[:]
    
    # Strategy 3: Greedy prime-step
    for n in range(4, 20):
        if time.time() > deadline:
            break
        gps = _construct_prime_step_greedy(n, max_val=3000, deadline=deadline)
        if len(gps) >= 2:
            s = _score_set(gps)
            if s > best_score:
                best_score = s
                best_U = gps[:]
    
    # Strategy 4: Mixed radix
    for n in range(3, 20):
        if time.time() > deadline:
            break
        mr = _construct_mixed_radix(n, deadline=deadline)
        if len(mr) >= 2:
            s = _score_set(mr)
            if s > best_score:
                best_score = s
                best_U = mr[:]
    
    # Strategy 5: Parametric arithmetic with perturbations
    for n in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, 12):
            base = [i * a for i in range(n)]
            for pert in [0, 1, 2]:
                if pert == 0:
                    candidate = base[:]
                else:
                    candidate = []
                    for i, v in enumerate(base):
                        if i == 0:
                            candidate.append(0)
                        else:
                            candidate.append(v + (pert if i % 2 == 0 else -pert))
                    candidate = sorted(set(x for x in candidate if x >= 0))
                    if 0 not in candidate:
                        continue
                if len(candidate) >= 2:
                    s = _score_set(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate[:]
    
    # Strategy 6: Power-of-prime
    for p in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        for n in range(3, 8):
            candidate = [0]
            val = 1
            for i in range(n - 1):
                val *= p
                candidate.append(val)
            s = _score_set(candidate)
            if s > best_score:
                best_score = s
                best_U = candidate[:]
    
    # Strategy 7: Quadratic
    for n in range(3, 14):
        if time.time() > deadline:
            break
        for c in range(1, 8):
            candidate = sorted(set(i * i * c for i in range(n)))
            if 0 not in candidate:
                candidate = [0] + [x for x in candidate if x > 0]
            if len(candidate) >= 2:
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Parameter optimization
    opt_U, opt_score = _optimize_parameters(start_time, deadline)
    if opt_score > best_score:
        best_score = opt_score
        best_U = opt_U[:]
    
    # Phase 2: Local search refinement with fast scoring
    if time.time() < deadline:
        improved_U, improved_score = _local_search_fast(best_U, max_iters=10000, deadline=deadline, max_size=50)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Try local search on diverse candidates
    if time.time() < deadline:
        candidates = []
        
        for n in [5, 8, 12, 15, 18, 20]:
            if time.time() > deadline:
                break
            gr = _construct_golomb(n)
            candidates.append(gr)
        
        for n in [5, 8, 10, 12, 15]:
            if time.time() > deadline:
                break
            ps = _construct_prime_step(n, max_val=3000)
            if len(ps) >= 2:
                candidates.append(ps)
        
        for n in [6, 8, 10, 12]:
            if time.time() > deadline:
                break
            gps = _construct_prime_step_greedy(n, max_val=2000, deadline=deadline)
            if len(gps) >= 2:
                candidates.append(gps)
        
        for n in [6, 8, 10, 12]:
            if time.time() > deadline:
                break
            mr = _construct_mixed_radix(n, deadline=deadline)
            if len(mr) >= 2:
                candidates.append(mr)
        
        for c in [1, 2, 3]:
            if time.time() > deadline:
                break
            for n in [8, 12, 16]:
                candidate = sorted(set(i * i * c for i in range(n)))
                if len(candidate) >= 2 and 0 in candidate:
                    candidates.append(candidate)
        
        random.seed(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            n = random.randint(4, 18)
            max_val = random.randint(20, 200)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            candidates.append(U)
        
        for U in candidates:
            if time.time() > deadline:
                break
            s = _score_set_fast(U)
            if s > best_score - 0.08:
                improved_U, improved_score = _local_search_fast(U, max_iters=3000, deadline=deadline, max_size=40)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    # Phase 4: Final polish - tighter local search on best
    if time.time() < deadline - 5:
        improved_U, improved_score = _local_search_fast(best_U, max_iters=3000, deadline=deadline, max_size=len(best_U) + 5)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
