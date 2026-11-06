import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate_set(nums):
        """Calculate the number of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes)

    def get_prime_factors_fast(n):
        """Return frozenset of prime factors for faster set operations."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Strategy 1: Clique Search for Prime Set {2, 3}
    # x_i = 6*k_i + 3, need k_i + k_j + 1 = 2^a * 3^b
    max_k = 10000 // 6
    smooth_sums = set()
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 2 * max_k + 1:
            break
        for b in range(15):
            val = pow2 * (3 ** b)
            if val > 2 * max_k + 1:
                break
            smooth_sums.add(val)
            
    found_clique = False
    best_clique = None
    
    # Heuristic Clique Search with multiple restarts
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        start_k = random.randint(0, 300)
        clique = [start_k]
        for _ in range(11):
            candidates = []
            for k_prime in range(0, 1667):
                if k_prime in clique:
                    continue
                ok = True
                for k in clique:
                    if (k + k_prime + 1) not in smooth_sums:
                        ok = False
                        break
                if ok:
                    candidates.append(k_prime)
            
            if not candidates:
                break
            next_k = random.choice(candidates)
            clique.append(next_k)
            
        if len(clique) == 12:
            found_clique = True
            best_clique = [6 * k + 3 for k in clique]
            break

    if found_clique:
        if len(set(best_clique)) == 12 and all(1 <= x <= 10000 for x in best_clique):
            if evaluate_set(best_clique) == 2:
                return best_clique

    # Strategy 2: Enhanced Local Search with targeted moves
    best_set = None
    best_score = float('inf')
    
    # Generate diverse initializations
    inits = []
    # Multiples of 6
    inits.append([6 * i for i in range(1, 13)])
    # 6k+3
    inits.append([6 * i + 3 for i in range(0, 12)])
    # 6k+1
    inits.append([6 * i + 1 for i in range(0, 12)])
    # 6k+5
    inits.append([6 * i + 5 for i in range(0, 12)])
    # Random
    inits.append(random.sample(range(1, 10001), 12))
    # Small random
    inits.append(random.sample(range(1, 500), 12))
    # Powers of 2 based
    inits.append([2**i for i in range(1, 13) if 2**i <= 10000][:12])
    # Mixed smooth
    smooth_vals = []
    for a in range(14):
        for b in range(14):
            v = (1 << a) * (3 ** b)
            if 1 <= v <= 10000:
                smooth_vals.append(v)
    if len(smooth_vals) >= 12:
        inits.append(random.sample(smooth_vals, 12))
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        curr_set = init.copy()
        curr_score = evaluate_set(curr_set)
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            if best_score <= 2:
                return best_set

        # Enhanced local search
        improved = True
        stall_count = 0
        max_stalls = 50
        while improved and time.time() < deadline - 3:
            improved = False
            stall_count += 1
            if stall_count > max_stalls:
                break
            
            # Try multiple random indices per iteration
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                idx = random.randint(0, 11)
                old_val = curr_set[idx]
                
                # Generate targeted candidates
                candidates = set()
                
                # 1. Small perturbations
                for delta in range(-20, 21):
                    nv = old_val + delta
                    if 1 <= nv <= 10000 and nv != old_val:
                        candidates.add(nv)
                
                # 2. Targeted: make sums with neighbors have few primes
                # For each neighbor, find values v such that v + neighbor has few prime factors
                for j in range(12):
                    if j == idx:
                        continue
                    neighbor = curr_set[j]
                    # Try to make v + neighbor = 2^a * 3^b or 2^a * 5^b, etc.
                    for target_prime_set in [(2, 3), (2, 5), (2, 7), (3, 5)]:
                        p1, p2 = target_prime_set
                        # v + neighbor should be of form p1^a * p2^b
                        for a in range(1, 12):
                            for b in range(1, 12):
                                target = (p1 ** a) * (p2 ** b)
                                if target <= neighbor + 2:
                                    continue
                                v = target - neighbor
                                if 1 <= v <= 10000:
                                    candidates.add(v)
                        
                        # Also try pure powers
                        for p in target_prime_set:
                            for a in range(1, 14):
                                target = p ** a
                                if target <= neighbor + 2:
                                    continue
                                v = target - neighbor
                                if 1 <= v <= 10000:
                                    candidates.add(v)
                
                # 3. Random structured candidates
                for _ in range(10):
                    r = random.randint(0, 5)
                    k = random.randint(1, 1666)
                    candidates.add(k * 6 + r)
                
                # 4. Pure random
                candidates.add(random.randint(1, 10000))
                
                # Evaluate candidates
                best_candidate = None
                best_candidate_score = curr_score
                
                # Sort candidates by heuristic: prefer those that make sums with neighbors have fewer primes
                scored_candidates = []
                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv in curr_set:
                        continue
                    # Quick heuristic: count prime factors of sums with a few neighbors
                    heuristic_score = 0
                    for j in range(12):
                        if j == idx:
                            continue
                        s = nv + curr_set[j]
                        heuristic_score += len(get_prime_factors(s))
                    scored_candidates.append((heuristic_score, nv))
                
                scored_candidates.sort()
                # Try top candidates
                for _, nv in scored_candidates[:15]:
                    new_set = curr_set.copy()
                    new_set[idx] = nv
                    new_score = evaluate_set(new_set)
                    if new_score < best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = nv
                
                if best_candidate is not None:
                    curr_set[idx] = best_candidate
                    curr_score = best_candidate_score
                    improved = True
                    stall_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                        if best_score <= 2:
                            return best_set
                    break  # Restart while loop
                
                # Also try swap moves
                if random.random() < 0.3:
                    idx2 = random.randint(0, 11)
                    if idx2 != idx:
                        new_set = curr_set.copy()
                        new_set[idx], new_set[idx2] = curr_set[idx2], curr_set[idx]
                        new_score = evaluate_set(new_set)
                        if new_score < curr_score:
                            curr_set = new_set
                            curr_score = new_score
                            improved = True
                            stall_count = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_set = curr_set.copy()
                                if best_score <= 2:
                                    return best_set
                            break
    
    # Final polish: try to reduce prime count further with focused search
    if best_set is not None and time.time() < deadline - 2:
        curr_set = best_set.copy()
        curr_score = best_score
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates targeting the current prime set
            # Get the current set of primes
            current_primes = set()
            for i in range(12):
                for j in range(i + 1, 12):
                    current_primes.update(get_prime_factors(curr_set[i] + curr_set[j]))
            
            candidates = set()
            # Try values that keep sums within current prime set
            for _ in range(50):
                nv = random.randint(1, 10000)
                if nv in curr_set:
                    continue
                # Check if all sums with neighbors use only current primes
                ok = True
                for j in range(12):
                    if j == idx:
                        continue
                    pf = get_prime_factors(nv + curr_set[j])
                    if not pf.issubset(current_primes):
                        ok = False
                        break
                if ok:
                    candidates.add(nv)
            
            # Also try small perturbations
            for delta in range(-50, 51):
                nv = old_val + delta
                if 1 <= nv <= 10000 and nv != old_val:
                    candidates.add(nv)
            
            best_candidate = None
            best_candidate_score = curr_score
            for nv in candidates:
                new_set = curr_set.copy()
                new_set[idx] = nv
                new_score = evaluate_set(new_set)
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate = nv
            
            if best_candidate is not None:
                curr_set[idx] = best_candidate
                curr_score = best_candidate_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_set = curr_set.copy()
                    if best_score <= 2:
                        return best_set
            else:
                break
    
    return best_set
# EVOLVE-BLOCK-END
