import random
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero = (0,) * num_primes
    candidates = list(range(2, 41))
    n_cand = len(candidates)
    
    def try_add(sums, v):
        new_items = []
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False, sums
            new_items.append(ns)
        seen = set(sums)
        for ns in new_items:
            if ns in seen:
                return False, sums
            seen.add(ns)
        return True, list(sums) + new_items
    
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with primes + squares
    initial = primes + [4, 9, 16, 25, 36]
    sums = [zero]
    valid = True
    for n in initial:
        v = vecs[n]
        ok, new_sums = try_add(sums, v)
        if not ok:
            valid = False
            break
        sums = new_sums
    
    if valid:
        best_set = initial[:]
        best_sums = sums
    else:
        best_set = primes[:]
        best_sums = [zero]
        for n in primes:
            v = vecs[n]
            _, best_sums = try_add(best_sums, v)
    
    best_len = len(best_set)
    
    # Try to add more elements
    if best_sums is not None:
        for n in candidates:
            if n in best_set:
                continue
            v = vecs[n]
            ok, new_sums = try_add(best_sums, v)
            if ok:
                best_set.append(n)
                best_sums = new_sums
                best_len = len(best_set)
                if best_len >= 18:
                    break
    
    # Local search
    if best_len < 18:
        rng = random.Random(42)
        while time.time() < deadline:
            current = best_set[:]
            in_set = set(current)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for n_remove in list(current):
                    sums_wo = [zero]
                    valid = True
                    for n in current:
                        if n == n_remove:
                            continue
                        v = vecs[n]
                        ok, new_sums = try_add(sums_wo, v)
                        if not ok:
                            valid = False
                            break
                        sums_wo = new_sums
                    if not valid:
                        continue
                    for n_add in candidates:
                        if n_add in in_set:
                            continue
                        v_add = vecs[n_add]
                        ok, new_sums = try_add(sums_wo, v_add)
                        if ok:
                            current.remove(n_remove)
                            current.append(n_add)
                            in_set.remove(n_remove)
                            in_set.add(n_add)
                            improved = True
                            break
                    if improved:
                        break
            if len(current) > best_len:
                best_len = len(current)
                best_set = current[:]
            if best_len >= 18:
                break
    
    return best_set
