"""Combined program: conflict-aware greedy + improved local search with merge moves."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending for fast lookup of best add
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def valid(s):
        used = set()
        for a in s:
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True
    
    def find_best_add(used_primes, current_set, exclude=None):
        """Find the best number to add that is coprime to used_primes and not in current_set."""
        if exclude is None:
            exclude = set()
        for a in candidates_by_score:
            if a in current_set or a in exclude:
                continue
            if not (prime_factors[a] & used_primes):
                return a
        return None

    def hill_climb(initial_set, max_iters=5000):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue

            # 2. Try 1-for-1 swap: remove one, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                # Find best pair to add
                best_pair = None
                best_pair_score = 0
                
                # Get list of valid candidates
                valid_cands = []
                for a in candidates_by_score:
                    if a in rem_set:
                        continue
                    if prime_factors[a] & rem_used:
                        continue
                    valid_cands.append(a)
                    if len(valid_cands) > 50: # Limit search space for pairs
                        break
                
                for idx_a in range(len(valid_cands)):
                    a = valid_cands[idx_a]
                    for idx_b in range(idx_a + 1, len(valid_cands)):
                        b = valid_cands[idx_b]
                        if prime_factors[a] & prime_factors[b]:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_set = rem_set | set(best_pair)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if improved:
                continue

            # 4. Try 2-for-1 swap: remove two, add one (MERGE)
            # This is crucial for combining primes into composites
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = set()
                    for a in remaining:
                        rem_used |= prime_factors[a]
                    
                    best_add = find_best_add(rem_used, rem_set)
                    if best_add is not None:
                        new_score = rem_score + scores[best_add]
                        if new_score > current_score + 1e-15:
                            current = remaining + [best_add]
                            current_set = rem_set | {best_add}
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return current, current_score
    
    # Initial strategy: Start with all primes
    # This ensures we have a high baseline and allows merges to improve it
    initial_set = [p for p in range(2, 512) if is_prime[p]]
    
    # Hill climb from all primes
    best_set, best_score = hill_climb(initial_set, max_iters=2000)
    
    # Try a few random restarts with perturbation if time permits
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Perturb best set
        perturbed = best_set[:]
        k = rng.randint(1, min(3, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        result, score = hill_climb(perturbed, max_iters=1000)
        
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Final polish
    if time.time() < deadline - 5:
        result, score = hill_climb(best_set, max_iters=5000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
            
    return best_set
# EVOLVE-BLOCK-END
