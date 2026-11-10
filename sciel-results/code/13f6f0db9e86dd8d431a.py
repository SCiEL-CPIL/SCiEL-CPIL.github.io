"""Erdos #689 n=1000: Greedy coverage-deficit construction + targeted simulated annealing.

Structural approach:
- Build residues greedily by processing x=1,2,... and for each x with deficit,
  assign the prime that maximizes coverage of future deficient integers.
- Then run simulated annealing with incremental coverage tracking,
  focusing mutations on primes covering the bottleneck.
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 145.0
MAX_L = 100000


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)
_N_PRIMES = len(_PRIMES)
_BY_P = {p: i for i, p in enumerate(_PRIMES)}


def _compute_score(residues, limit):
    """Return largest L such that every x in 1..L has >= 2 covering primes."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
        start = rr
        if start < 1:
            start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    ok = mark[1:] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _compute_coverage(residues, limit):
    """Return coverage array where cov[x] = number of primes covering x."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
        start = rr
        if start < 1:
            start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    return mark


def _greedy_construct(target_L, seed=0):
    """Greedy: process x=1..target_L, assign primes to cover deficient x.
    
    For each x with deficit (fewer than 2 covering primes), pick the unassigned
    prime p that, when set to r_p = x % p, covers the most future deficient integers.
    """
    rng = random.Random(seed)
    residues = [-1] * _N_PRIMES
    # Track which small primes are assigned
    small_primes = [(i, p) for i, p in enumerate(_PRIMES) if p <= target_L + 200]
    
    # Coverage tracking: cov[x] = count of primes covering x
    cov = [0] * (target_L + 200)
    
    for x in range(1, target_L + 1):
        # Count current coverage of x
        count = 0
        for pi, p in small_primes:
            if residues[pi] >= 0 and x % p == residues[pi]:
                count += 1
        
        if count >= 2:
            continue
        
        need = 2 - count
        
        # Find unassigned small primes
        candidates = [(pi, p) for pi, p in small_primes if residues[pi] < 0]
        if len(candidates) < need:
            break
        
        # Score each candidate by how many future deficient integers it would cover
        # For prime p with r = x % p, it covers x, x+p, x+2p, ...
        # Score = number of y in [x+1, target_L] where y % p == x % p AND cov[y] < 2
        best_candidates = []
        for pi, p in candidates:
            r = x % p
            # Count future integers covered
            score = 0
            for y in range(x + p, target_L + 1, p):
                if cov[y] < 2:
                    score += 1
            # Also weight by frequency: smaller p covers more often
            score += target_L // p * 0.1
            best_candidates.append((score, pi, p, r))
        
        best_candidates.sort(key=lambda t: -t[0])
        
        for _, pi, p, r in best_candidates[:need]:
            residues[pi] = r
            # Update coverage
            for y in range(x, target_L + 200, p):
                if y <= target_L + 200:
                    cov[y] += 1
    
    # Fill unassigned primes
    full_res = [0] * _N_PRIMES
    for i in range(_N_PRIMES):
        if residues[i] >= 0:
            full_res[i] = residues[i]
        else:
            full_res[i] = 0
    
    return full_res


def _incremental_score_update(cov, limit, old_pi, new_r):
    """Compute new score after changing residue of prime old_pi to new_r.
    Returns (new_score, updated_cov)."""
    p = _PRIMES[old_pi]
    
    # Remove old contribution
    old_r = None  # We need to track this; for now, recompute
    # Actually, let's just recompute the score from cov after updating
    
    # Find old residue - we need to pass it in
    # For efficiency, let's do a full recompute of just the affected positions
    # But that's complex. Let's use a different approach.
    
    # Update cov: subtract old prime's contribution, add new
    # We need the old residue. Let's handle this in the caller.
    raise NotImplementedError


def _anneal(residues, base_score, budget, seed=42):
    """Simulated annealing with incremental coverage.
    
    Mutate one prime at a time. Accept if score improves, or with probability
    exp(-delta/T) if it worsens. Focus mutations on primes covering the bottleneck.
    """
    rng = random.Random(seed)
    best_score = base_score
    best_res = list(residues)
    cur_res = list(residues)
    cur_score = base_score
    
    # Determine working limit for coverage array
    work_limit = min(base_score + 5000, MAX_L)
    
    # Build initial coverage
    cov = np.zeros(work_limit + 1, dtype=np.int16)
    for pi, p in enumerate(_PRIMES):
        r = cur_res[pi]
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= work_limit:
            cov[start::p] += 1
    
    # Identify bottleneck: first x where cov[x] < 2
    def find_bottleneck(cov_arr, limit):
        for x in range(1, limit + 1):
            if cov_arr[x] < 2:
                return x
        return limit + 1
    
    # Primes that cover the bottleneck region
    def get_bottleneck_primes(cov_arr, bottleneck_x, limit):
        """Primes p where changing r_p could fix the bottleneck."""
        primes_list = []
        for pi, p in enumerate(_PRIMES):
            if p > limit:
                continue
            # If we set r_p = bottleneck_x % p, would it help?
            # It covers bottleneck_x, bottleneck_x + p, etc.
            primes_list.append(pi)
        return primes_list
    
    T_init = max(base_score, 100) * 0.1
    T_min = 1.0
    end_time = time.time() + budget
    steps = 0
    no_improve = 0
    
    while time.time() < end_time:
        steps += 1
        # Cooling
        T = max(T_min, T_init * (0.9995 ** steps))
        
        # Choose which prime to mutate
        # Bias toward primes that cover the current bottleneck
        bottleneck = find_bottleneck(cov, min(cur_score + 100, work_limit))
        
        if rng.random() < 0.6 and bottleneck <= work_limit:
            # Mutate a prime that covers the bottleneck
            # Primes p where bottleneck % p gives a valid residue
            candidates = [pi for pi, p in enumerate(_PRIMES) if p <= work_limit + 100]
            if candidates:
                pi = rng.choice(candidates)
            else:
                pi = rng.randint(0, _N_PRIMES - 1)
        else:
            pi = rng.randint(0, _N_PRIMES - 1)
        
        p = _PRIMES[pi]
        old_r = cur_res[pi]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Compute score change
        # Remove old prime contribution from cov
        old_start = old_r
        if old_start < 1:
            old_start += ((1 - old_start + p - 1) // p) * p
        if old_start <= work_limit:
            cov[old_start::p] -= 1
        
        # Add new prime contribution
        new_start = new_r
        if new_start < 1:
            new_start += ((1 - new_start + p - 1) // p) * p
        if new_start <= work_limit:
            cov[new_start::p] += 1
        
        # Compute new score (find first x with cov < 2)
        # Only need to check up to work_limit
        new_score = work_limit
        for x in range(1, work_limit + 1):
            if cov[x] < 2:
                new_score = x - 1
                break
        
        # Accept/reject
        if new_score >= cur_score:
            cur_res[pi] = new_r
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_res = list(cur_res)
                no_improve = 0
            else:
                no_improve += 1
        else:
            delta = cur_score - new_score
            if rng.random() < np.exp(-delta / T):
                cur_res[pi] = new_r
                cur_score = new_score
            else:
                # Revert
                # Remove new, restore old
                if new_start <= work_limit:
                    cov[new_start::p] -= 1
                if old_start <= work_limit:
                    cov[old_start::p] += 1
        
        # Periodic verification against full limit
        if steps % 500 == 0 and time.time() < end_time - 5:
            full_s = _compute_score(cur_res, min(cur_score + 2000, MAX_L))
            if full_s > best_score:
                best_score = full_s
                best_res = list(cur_res)
    
    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)
    
    best_score = 0
    best_res = [0] * _N_PRIMES
    
    # Phase 1: Greedy construction for increasing targets
    for target_L in [200, 500, 1000, 2000, 5000, 10000]:
        if time.time() > DEADLINE - 30:
            break
        
        for seed in range(3):
            if time.time() > DEADLINE - 30:
                break
            res = _greedy_construct(target_L, seed=seed)
            score = _compute_score(res, min(target_L + 5000, MAX_L))
            if score > best_score:
                best_score = score
                best_res = res
                print(f"Greedy L={target_L} seed={seed}: score={score}")
    
    # Phase 2: Simulated annealing from best greedy solution
    if time.time() < DEADLINE - 20:
        budget = min(60.0, DEADLINE - time.time() - 20)
        s, r = _anneal(best_res, best_score, budget, seed=12345)
        if s > best_score:
            best_score = s
            best_res = r
            print(f"After anneal 1: score={best_score}")
    
    # Phase 3: Another annealing round with different seed
    if time.time() < DEADLINE - 15:
        budget = min(40.0, DEADLINE - time.time() - 15)
        s, r = _anneal(best_res, best_score, budget, seed=99999)
        if s > best_score:
            best_score = s
            best_res = r
            print(f"After anneal 2: score={best_score}")
    
    # Phase 4: Final targeted repair
    if time.time() < DEADLINE - 5:
        # Try to extend by fixing the first uncovered integer
        work_limit = min(best_score + 3000, MAX_L)
        cov = _compute_coverage(best_res, work_limit)
        # Find first deficient
        for x in range(1, work_limit + 1):
            if cov[x] < 2:
                # Try to fix by adjusting a prime
                for pi in range(_N_PRIMES):
                    p = _PRIMES[pi]
                    if p > work_limit:
                        break
                    old_r = best_res[pi]
                    new_r = x % p
                    if new_r == old_r:
                        continue
                    trial = list(best_res)
                    trial[pi] = new_r
                    s = _compute_score(trial, min(best_score + 2000, MAX_L))
                    if s > best_score:
                        best_score = s
                        best_res = trial
                        break
                break
    
    # Final verification
    final_score = _compute_score(best_res, MAX_L)
    if final_score > best_score:
        best_score = final_score
    
    return best_res
