import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    num_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    MAX_EVAL = 20000
    xs = np.arange(1, MAX_EVAL + 1, dtype=np.int64)

    def compute_coverage(residues, limit):
        """Compute coverage array for x in 1..limit."""
        xs_sub = xs[:limit]
        diff = xs_sub[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        return np.sum(hits, axis=0)

    def evaluate(residues, limit=None):
        """Compute largest L such that 1..L has coverage >= 2."""
        if limit is None:
            limit = MAX_EVAL
        coverage = compute_coverage(residues, limit)
        mask = coverage < 2
        if not np.any(mask):
            return limit
        first_fail_idx = np.argmax(mask)
        return int(xs[first_fail_idx] - 1)

    def get_criticality(residues, L):
        """
        For each prime i, count how many x in 1..L have coverage exactly 2
        and are covered by prime i. These are the x that would break if we
        changed prime i's residue.
        """
        if L <= 0:
            return np.zeros(num_primes, dtype=np.int32)
        
        coverage = compute_coverage(residues, L)
        criticality = np.zeros(num_primes, dtype=np.int32)
        for i in range(num_primes):
            p = primes[i]
            r = residues[i]
            x_mod = xs[:L] % p
            hits_i = (x_mod == r)
            criticality[i] = np.sum(hits_i & (coverage == 2))
        
        return criticality

    random.seed(42)
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    current_L = evaluate(residues)
    best_residues = residues.copy()
    best_L = current_L

    stuck_count = 0
    MAX_STUCK = 30

    while time.time() < deadline:
        current_L = evaluate(residues)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            stuck_count = 0

        if current_L >= MAX_EVAL:
            break

        target_x = current_L + 1

        covers_target = np.array([(target_x - residues[i]) % primes[i] == 0 for i in range(num_primes)])
        num_covering = np.sum(covers_target)

        if num_covering >= 2:
            continue

        candidates = [i for i in range(num_primes) if not covers_target[i]]

        if not candidates:
            break

        if current_L > 0:
            criticality = get_criticality(residues, current_L)
            candidates.sort(key=lambda i: criticality[i])
        else:
            random.shuffle(candidates)

        found_repair = False

        for idx in candidates:
            if time.time() > deadline - 2.0:
                break
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            
            if old_res == new_res:
                continue
                
            residues[idx] = new_res
            
            coverage = compute_coverage(residues, current_L)
            if np.all(coverage >= 2):
                new_L = evaluate(residues)
                if new_L > current_L:
                    found_repair = True
                    stuck_count = 0
                    break
            else:
                residues[idx] = old_res

        if not found_repair and current_L > 0 and time.time() < deadline - 5.0:
            if current_L > 0:
                criticality = get_criticality(residues, current_L)
            candidates_sorted = sorted(candidates, key=lambda i: criticality[i] if current_L > 0 else 0)
            
            tried_pairs = set()
            max_pairs = min(len(candidates_sorted), 15)
            for i_idx in range(max_pairs):
                for j_idx in range(i_idx + 1, max_pairs):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = candidates_sorted[i_idx], candidates_sorted[j_idx]
                    if (i, j) in tried_pairs:
                        continue
                    tried_pairs.add((i, j))
                    
                    old_res_i = residues[i]
                    old_res_j = residues[j]
                    new_res_i = target_x % primes[i]
                    new_res_j = target_x % primes[j]
                    
                    if old_res_i == new_res_i and old_res_j == new_res_j:
                        continue
                    
                    residues[i] = new_res_i
                    residues[j] = new_res_j
                    
                    coverage = compute_coverage(residues, current_L)
                    if np.all(coverage >= 2):
                        new_L = evaluate(residues)
                        if new_L > current_L:
                            found_repair = True
                            stuck_count = 0
                            break
                    else:
                        residues[i] = old_res_i
                        residues[j] = old_res_j
                
                if found_repair:
                    break

        if not found_repair:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                num_perturb = random.randint(3, 8)
                for _ in range(num_perturb):
                    i = random.randint(0, num_primes - 1)
                    p = primes[i]
                    residues[i] = random.randint(0, p - 1)
                stuck_count = 0
                current_L = evaluate(residues)
                if current_L < best_L:
                    residues = best_residues.copy()
                    current_L = best_L

    return [int(r) for r in best_residues]
