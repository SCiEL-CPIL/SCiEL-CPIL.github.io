import time
import random
import sys
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin, evaluator kills at 90s
    
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

    def get_prime_factors_set(nums):
        """Return the set of prime factors."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return all_primes

    # Strategy 1: Clique Search for Small Prime Sets
    def try_find_clique_for_primes(P, max_attempts=500):
        # Generate P-smooth numbers up to 20000
        smooth_numbers = set()
        ps = list(P)
        def gen_smooth(val, idx):
            if val > 20000:
                return
            smooth_numbers.add(val)
            for i in range(idx, len(ps)):
                gen_smooth(val * ps[i], i)
        gen_smooth(1, 0)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            clique = [random.randint(1, 10000)]
            
            for _ in range(11):
                # Find candidates for next element
                candidate_sets = []
                for x_k in clique:
                    S_k = set()
                    for s in smooth_numbers:
                        v = s - x_k
                        if 1 <= v <= 10000:
                            S_k.add(v)
                    candidate_sets.append(S_k)
                
                candidates = candidate_sets[0]
                for i in range(1, len(candidate_sets)):
                    candidates = candidates.intersection(candidate_sets[i])
                
                candidates -= set(clique)
                
                if not candidates:
                    break
                
                next_k = random.choice(list(candidates))
                clique.append(next_k)
                
            if len(clique) == 12:
                if len(set(clique)) == 12 and all(1 <= x <= 10000 for x in clique):
                    primes_used = get_prime_factors_set(clique)
                    if primes_used.issubset(P):
                        return clique
        return None

    target_prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7}
    ]
    
    best_set = None
    best_score = float('inf')
    
    for P in target_prime_sets:
        if time.time() > deadline - 20:
            break
        clique = try_find_clique_for_primes(P, max_attempts=1000)
        if clique:
            score = evaluate_set(clique)
            if score < best_score:
                best_score = score
                best_set = clique
            if score <= 3:
                return best_set

    # Strategy 2: Enhanced Local Search with Simulated Annealing
    if best_set is None:
        best_set = random.sample(range(1, 10001), 12)
        best_score = evaluate_set(best_set)

    # Generate diverse initializations
    inits = []
    inits.append([6 * i for i in range(1, 13)])
    inits.append([6 * i + 3 for i in range(0, 12)])
    inits.append([6 * i + 1 for i in range(0, 12)])
    inits.append([6 * i + 5 for i in range(0, 12)])
    inits.append(random.sample(range(1, 10001), 12))
    inits.append(random.sample(range(1, 500), 12))
    inits.append(best_set.copy())
    
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        curr_set = init.copy()
        curr_score = evaluate_set(curr_set)
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr_set.copy()
            if best_score <= 4:
                return best_set

        T = 10.0
        cooling = 0.99
        min_T = 0.1
        improved = True
        stall_count = 0
        
        while T > min_T and time.time() < deadline - 2:
            improved = False
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, 11)
                old_val = curr_set[idx]
                
                candidates = set()
                for j in range(12):
                    if j == idx:
                        continue
                    neighbor = curr_set[j]
                    for a in range(1, 15):
                        for b in range(1, 15):
                            target = (1 << a) * (3 ** b)
                            if target <= neighbor + 2:
                                continue
                            v = target - neighbor
                            if 1 <= v <= 10000:
                                candidates.add(v)
                    for p in [2, 3, 5]:
                        for a in range(1, 15):
                            target = p ** a
                            if target <= neighbor + 2:
                                continue
                            v = target - neighbor
                            if 1 <= v <= 10000:
                                candidates.add(v)
                
                for _ in range(5):
                    candidates.add(random.randint(1, 10000))
                
                valid_candidates = [c for c in candidates if 1 <= c <= 10000 and c != old_val and c not in curr_set]
                
                if not valid_candidates:
                    continue
                
                new_val = random.choice(valid_candidates)
                new_set = curr_set.copy()
                new_set[idx] = new_val
                new_score = evaluate_set(new_set)
                
                delta = new_score - curr_score
                if delta < 0:
                    curr_set = new_set
                    curr_score = new_score
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_set = curr_set.copy()
                        if best_score <= 4:
                            return best_set
                else:
                    if random.random() < math.exp(-delta / T):
                        curr_set = new_set
                        curr_score = new_score
                        improved = True
            
            T *= cooling
            if not improved:
                stall_count += 1
                if stall_count > 10:
                    break
            else:
                stall_count = 0

    return best_set
# EVOLVE-BLOCK-END
