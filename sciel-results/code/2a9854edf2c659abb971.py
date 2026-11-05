import time
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
    
    # Assign each prime a bit position
    primes_list = [p for p in range(2, 512) if is_prime[p]]
    prime_to_bit = {p: i for i, p in enumerate(primes_list)}
    num_primes = len(primes_list)
    
    # For each number 1..511, compute its prime factor bitmask
    prime_factor_mask = [0] * 512
    for p in primes_list:
        bit = 1 << prime_to_bit[p]
        for multiple in range(p, 512, p):
            prime_factor_mask[multiple] |= bit
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def find_best_add(used_mask, current_set):
        """Find the best number to add that is coprime to used_mask and not in current_set."""
        for a in candidates_by_score:
            if a in current_set:
                continue
            if (prime_factor_mask[a] & used_mask) == 0:
                return a
        return None

    def find_best_pair(used_mask, current_set, limit=100):
        """Find the best pair (a,b) to add that are both coprime to used_mask, to each other, and not in current_set."""
        valid_cands = []
        for a in candidates_by_score:
            if a in current_set:
                continue
            if prime_factor_mask[a] & used_mask:
                continue
            valid_cands.append(a)
            if len(valid_cands) > limit:
                break
        
        best_pair = None
        best_pair_score = -1.0
        for i in range(len(valid_cands)):
            a = valid_cands[i]
            mask_a = prime_factor_mask[a]
            for j in range(i + 1, len(valid_cands)):
                b = valid_cands[j]
                if mask_a & prime_factor_mask[b]:
                    continue
                s = scores[a] + scores[b]
                if s > best_pair_score:
                    best_pair_score = s
                    best_pair = (a, b)
        return best_pair

    def hill_climb(initial_set, max_iters=5000, aggressive=True):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_mask = 0
            for a in current:
                used_mask |= prime_factor_mask[a]
            
            best_add = find_best_add(used_mask, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue

            # 2. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_mask = 0
                for a in remaining:
                    rem_mask |= prime_factor_mask[a]
                
                best_pair = find_best_pair(rem_mask, rem_set)
                if best_pair is not None:
                    new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                    if new_score > current_score + 1e-15:
                        current = remaining + list(best_pair)
                        current_set = rem_set | set(best_pair)
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # 3. Try 1-for-1 swap
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_mask = 0
                for a in remaining:
                    rem_mask |= prime_factor_mask[a]
                
                best_add = find_best_add(rem_mask, rem_set)
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

            # 4. Try 2-for-1 swap (MERGE): remove two, add one
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
                    
                    rem_mask = 0
                    for a in remaining:
                        rem_mask |= prime_factor_mask[a]
                    
                    best_add = find_best_add(rem_mask, rem_set)
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
            
            # 5. Try 2-for-2 swap
            if aggressive and not improved:
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
                        
                        rem_mask = 0
                        for a in remaining:
                            rem_mask |= prime_factor_mask[a]
                        
                        # Find best pair to add
                        best_pair = find_best_pair(rem_mask, rem_set)
                        if best_pair is not None:
                            new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                            if new_score > current_score + 1e-15:
                                current = remaining + list(best_pair)
                                current_set = rem_set | set(best_pair)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                break

        return current, current_score

    # 1. Greedy Initial Solution
    def greedy_solve():
        used_mask = 0
        sol = []
        for a in candidates_by_score:
            if (prime_factor_mask[a] & used_mask) == 0:
                sol.append(a)
                used_mask |= prime_factor_mask[a]
        return sol, get_score(sol)

    initial_sol, initial_score = greedy_solve()
    best_score = initial_score
    best_sol = initial_sol[:]
    
    # 2. Hill Climb on initial
    sol, score = hill_climb(initial_sol, max_iters=5000, aggressive=True)
    if score > best_score:
        best_score = score
        best_sol = sol[:]

    # 3. Diversified Restarts
    rng = random.Random(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Create a perturbed solution
        perturbed_sol = best_sol[:]
        # Remove 2-6 random elements
        num_remove = min(2 + rng.randint(0, 4), len(perturbed_sol))
        if num_remove > 0:
            to_remove = rng.sample(perturbed_sol, num_remove)
            remove_set = set(to_remove)
            perturbed_sol = [x for x in perturbed_sol if x not in remove_set]
        
        # Run hill climb
        new_sol, new_score = hill_climb(perturbed_sol, max_iters=3000, aggressive=True)
        if new_score > best_score:
            best_score = new_score
            best_sol = new_sol[:]

    # 4. Final Polish
    final_sol, final_score = hill_climb(best_sol, max_iters=5000, aggressive=True)
    if final_score > best_score:
        best_score = final_score
        best_sol = final_sol[:]

    return best_sol
