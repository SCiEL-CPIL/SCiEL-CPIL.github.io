import time
import math
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for 90s limit
    n = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL

    # 1. Precompute SPF and Prime Factors
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors(x):
        if x <= 1:
            return frozenset()
        s = set()
        while x > 1:
            p = spf[x]
            s.add(p)
            while x % p == 0:
                x //= p
        return frozenset(s)

    # Precompute factors for all possible sums
    pf = [None] * (MAX_SUM + 1)
    pf[0] = frozenset()
    pf[1] = frozenset()
    for i in range(2, MAX_SUM + 1):
        pf[i] = prime_factors(i)

    def get_score(arr):
        """Returns the set of distinct prime factors and its size."""
        u = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                u |= pf[ai + arr[j]]
        return len(u), u

    # 2. Initialization Strategies
    def smooth_pool(primes, limit=10000):
        pool = set([1])
        for p in primes:
            new = set()
            for x in list(pool):
                v = x
                while v <= limit:
                    new.add(v)
                    v *= p
            pool |= new
        return sorted(pool)

    # Pools for initialization
    pool_7 = smooth_pool([2, 3, 5, 7])
    pool_5 = smooth_pool([2, 3, 5])
    
    seeds = []
    # Seed 1: Small 7-smooth
    seeds.append(pool_7[:n])
    # Seed 2: Small 5-smooth
    seeds.append(pool_5[:n])
    # Seed 3: Multiples of 30
    seeds.append([30 * i for i in range(1, n + 1)])
    # Seed 4: Multiples of 210
    seeds.append([210 * i for i in range(1, n + 1)])
    # Seed 5: Random from 7-smooth
    random.seed(42)
    for _ in range(10):
        s = random.sample(pool_7, n)
        seeds.append(s)
        
    # Validate and deduplicate seeds
    valid_seeds = []
    seen = set()
    for s in seeds:
        if len(set(s)) != n or any(x > MAX_VAL for x in s) or any(x <= 0 for x in s):
            continue
        key = tuple(sorted(s))
        if key in seen:
            continue
        seen.add(key)
        valid_seeds.append(list(s))

    if not valid_seeds:
        valid_seeds = [[i for i in range(1, n + 1)]]

    best_arr = valid_seeds[0]
    best_score, _ = get_score(best_arr)

    # 3. Improvement Algorithm
    
    def try_replace(arr, idx, candidates):
        """Try to replace arr[idx] with a candidate from candidates to improve score."""
        old_val = arr[idx]
        old_score, old_u = get_score(arr)
        
        best_cand = None
        best_new_score = old_score
        best_new_u = old_u

        # Filter candidates: must be distinct from other elements
        others = set(arr)
        others.discard(old_val)

        for cand in candidates:
            if cand in others:
                continue
            if cand == old_val:
                continue
            if cand < 1 or cand > MAX_VAL:
                continue
            
            # Quick check: does this cand introduce new primes?
            # We compute the new union
            new_u = set()
            for j in range(n):
                if j == idx:
                    continue
                s = arr[j] + cand
                new_u |= pf[s]
            
            # Add sums among other elements (unchanged)
            # Optimization: We can just recompute fully if we want, 
            # but since n=12, full recompute is 66 pairs. 
            # However, we can be smarter.
            # Actually, just calling get_score is simpler and fast enough.
            
            # Let's just use get_score for simplicity and correctness
            arr[idx] = cand
            new_score, u = get_score(arr)
            arr[idx] = old_val
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_new_u = u
                best_cand = cand
            elif new_score == best_new_score and best_cand is None:
                # Tie-breaking: prefer smaller numbers? Or random?
                # Let's stick with first found or smallest
                if best_cand is None or cand < best_cand:
                     best_cand = cand
                     best_new_u = u

        if best_cand is not None:
            arr[idx] = best_cand
            return True, best_new_score, best_new_u
        return False, old_score, old_u

    def targeted_elimination(arr, max_iter=1000):
        """Try to eliminate primes from the set of factors."""
        cur_score, cur_u = get_score(arr)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline - 5.0:
            improved = False
            it += 1
            if not cur_u:
                break
                
            # Pick a prime to eliminate. 
            # Strategy: Pick the largest prime, or one that appears in fewest sums?
            # Let's pick the largest prime first, as it's "heavier" in terms of value constraints usually.
            # Or simply iterate through primes.
            # Let's try to eliminate the largest prime first.
            target_p = max(cur_u)
            
            # Try to replace each element to remove target_p
            for idx in range(n):
                if time.time() > deadline - 5.0:
                    return cur_score, cur_u
                
                old_val = arr[idx]
                others = set(arr)
                others.discard(old_val)
                
                # Find candidates c such that for all o in others, target_p does not divide (o + c)
                # i.e., c % target_p != -o % target_p for all o in others.
                # Let forbidden_residues = { (-o) % target_p for o in others }
                forbidden = set()
                for o in others:
                    forbidden.add((-o) % target_p)
                
                # We want c such that c % target_p NOT IN forbidden.
                # Also c must be in [1, 10000] and distinct from others.
                
                # Generate candidates efficiently
                # Instead of scanning 1..10000, we can scan a pool or specific ranges.
                # Let's scan a random sample or a smooth pool.
                # Using pool_7 is good.
                
                found = False
                # Try a few random candidates from pool_7
                # And some random numbers
                candidates = []
                # Add some random numbers
                for _ in range(50):
                    c = random.randint(1, MAX_VAL)
                    candidates.append(c)
                # Add some from pool_7
                for c in random.sample(pool_7, min(50, len(pool_7))):
                    candidates.append(c)
                
                # Filter candidates by residue
                valid_cands = []
                for c in candidates:
                    if c in others:
                        continue
                    if c == old_val:
                        continue
                    if c % target_p in forbidden:
                        continue
                    valid_cands.append(c)
                
                if not valid_cands:
                    continue
                    
                # Try the valid candidates
                best_cand = None
                best_score = cur_score
                best_u = cur_u
                
                for c in valid_cands:
                    arr[idx] = c
                    ns, nu = get_score(arr)
                    arr[idx] = old_val
                    
                    if ns < best_score:
                        best_score = ns
                        best_u = nu
                        best_cand = c
                    elif ns == best_score and best_cand is None:
                        # Tie break: prefer if it removes more primes?
                        # Or just take it
                        if len(nu) < len(best_u):
                            best_u = nu
                            best_cand = c
                
                if best_cand is not None:
                    arr[idx] = best_cand
                    cur_score = best_score
                    cur_u = best_u
                    improved = True
                    # Restart the loop to re-evaluate primes
                    break 
                    
        return cur_score, cur_u

    def simulated_annealing(arr, max_iter=2000):
        """General SA to escape local minima."""
        cur_score, cur_u = get_score(arr)
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if T < T_min:
                T = T_min
                
            idx = random.randint(0, n - 1)
            old_val = arr[idx]
            
            # Pick a candidate
            # Mix of smooth and random
            if random.random() < 0.5:
                cand = random.choice(pool_7)
            else:
                cand = random.randint(1, MAX_VAL)
                
            if cand == old_val:
                continue
                
            others = set(arr)
            others.discard(old_val)
            if cand in others:
                # Try to find a distinct one quickly
                for _ in range(5):
                    cand = random.randint(1, MAX_VAL)
                    if cand not in others and cand != old_val:
                        break
                else:
                    continue
            
            arr[idx] = cand
            new_score, new_u = get_score(arr)
            
            delta = new_score - cur_score
            if delta <= 0 or random.random() < math.exp(-delta / T):
                cur_score = new_score
                cur_u = new_u
                # Update best if needed (handled outside)
            else:
                arr[idx] = old_val
                
            T *= cooling
            
        return cur_score, cur_u

    # Main Loop
    # Start with the best seed
    cur_arr = list(valid_seeds[0])
    cur_score, _ = get_score(cur_arr)
    if cur_score < best_score:
        best_score = cur_score
        best_arr = cur_arr

    for seed in valid_seeds[1:]:
        if time.time() > deadline - 10.0:
            break
        cur_arr = list(seed)
        cur_score, _ = get_score(cur_arr)
        
        # Improve
        cur_score, _ = targeted_elimination(cur_arr)
        if cur_score < best_score:
            best_score = cur_score
            best_arr = cur_arr
            
        # SA polish
        cur_score, _ = simulated_annealing(cur_arr, max_iter=500)
        if cur_score < best_score:
            best_score = cur_score
            best_arr = cur_arr

    # Final polish on the best found
    if time.time() < deadline - 5.0:
        cur_arr = list(best_arr)
        cur_score, _ = get_score(cur_arr)
        # Run targeted elimination again
        cur_score, _ = targeted_elimination(cur_arr)
        if cur_score < best_score:
            best_score = cur_score
            best_arr = cur_arr
            
        # Run SA again
        cur_score, _ = simulated_annealing(cur_arr, max_iter=1000)
        if cur_score < best_score:
            best_score = cur_score
            best_arr = cur_arr

    return best_arr
