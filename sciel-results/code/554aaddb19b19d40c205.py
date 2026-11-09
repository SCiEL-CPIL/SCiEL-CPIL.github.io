import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
        
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
        
    # Initialize with 1 and highest power of each prime
    initial_set = {1}
    for p in primes:
        power = p
        while power * p < N:
            power *= p
        initial_set.add(power)
        
    # Convert to list for easier manipulation, but keep set for O(1) lookup
    # We will maintain:
    # chosen: list of elements
    # chosen_set: set of elements
    # chosen_mask: OR of factor_masks of elements
    
    def compute_mask(chosen_set):
        mask = 0
        for a in chosen_set:
            mask |= factor_masks[a]
        return mask
        
    best_score = -1
    best_chosen_set = set()
    
    def evaluate(chosen_set):
        return sum(scores[a] for a in chosen_set)
        
    # Local Search
    def local_search(chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Merge Move: Replace a, b with a*b
            # Iterate over pairs
            chosen_list = list(chosen_set)
            n = len(chosen_list)
            for i in range(n):
                a = chosen_list[i]
                if a == 1: continue # Merging 1 doesn't change value, just removes 1. w(1)=1/511. w(1*b)=w(b). w(b) > w(1). So we should merge 1 with b?
                # If we merge 1 and b, we remove 1 and b, add b. Net change: remove 1. Score decreases.
                # So we should NOT merge 1 with others. We should keep 1 separate.
                # Actually, if we merge 1 and b, the set becomes S \ {1, b} U {b} = S \ {1}.
                # Score change: -w(1). Negative.
                # So skip 1 in merges.
                
                for j in range(i+1, n):
                    b = chosen_list[j]
                    if b == 1: continue
                    prod = a * b
                    if prod >= N:
                        continue
                    # Check if prod is coprime to rest
                    # Rest mask = chosen_mask ^ mask(a) ^ mask(b)
                    # Note: mask(a) and mask(b) are disjoint if gcd(a,b)=1, which is true for distinct primes powers.
                    # But after merges, they might not be?
                    # No, if a and b are in S, they are coprime. So masks are disjoint.
                    rest_mask = chosen_mask & ~factor_masks[a] & ~factor_masks[b]
                    if factor_masks[prod] & rest_mask:
                        continue
                    
                    gain = scores[prod] - scores[a] - scores[b]
                    if gain > 1e-12:
                        # Perform merge
                        chosen_set.remove(a)
                        chosen_set.remove(b)
                        chosen_set.add(prod)
                        chosen_mask = (chosen_mask & ~factor_masks[a] & ~factor_masks[b]) | factor_masks[prod]
                        current_score += gain
                        improved = True
                        # Restart loop to re-evaluate pairs
                        break
                if improved:
                    break
            
            if improved:
                continue
                
            # 2. 1-Swap Move: Replace a with b
            for a in list(chosen_set):
                best_b = None
                best_gain = 0
                # Try all b not in chosen_set
                # Optimization: Only try b with score > score(a)
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if scores[b] <= scores[a]:
                        continue
                    rest_mask = chosen_mask & ~factor_masks[a]
                    if factor_masks[b] & rest_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                if best_b is not None:
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[best_b]
                    current_score += best_gain
                    improved = True
                    break
            if improved:
                continue
                
            # 3. 2-Swap Move (Remove 2, Add 2)
            # This is expensive, but we can try a limited version
            # Or just rely on SA for this.
            
        return chosen_set, chosen_mask, current_score

    # Run local search on initial set
    current_set = set(initial_set)
    current_mask = compute_mask(current_set)
    current_score = evaluate(current_set)
    
    current_set, current_mask, current_score = local_search(current_set, current_mask, current_score)
    
    if current_score > best_score:
        best_score = current_score
        best_chosen_set = set(current_set)
        
    # Simulated Annealing to find better solutions
    # We might need multiple restarts or a long SA
    def simulated_annealing(start_set, start_mask, start_score):
        chosen_set = set(start_set)
        chosen_mask = start_mask
        current_score = start_score
        
        temperature = 0.01
        cooling_rate = 0.99
        min_temperature = 0.0001
        
        while temperature > min_temperature and time.time() < deadline:
            move_type = random.random()
            if move_type < 0.7:
                # 1-Swap
                a = random.choice(list(chosen_set))
                # Find candidates
                rest_mask = chosen_mask & ~factor_masks[a]
                candidates = [b for b in range(1, N) if b not in chosen_set and (factor_masks[b] & rest_mask) == 0]
                if not candidates:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates)
                delta = scores[b] - scores[a]
                if delta > 0 or random.random() < pow(2.71828, delta / temperature):
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    current_score += delta
            else:
                # Merge/Unmerge?
                # Merge is hard to reverse in SA without tracking history.
                # Let's stick to 1-swap and 2-swap in SA.
                if len(chosen_set) >= 2:
                    a, b = random.sample(list(chosen_set), 2)
                    rest_mask = chosen_mask & ~factor_masks[a] & ~factor_masks[b]
                    candidates = [c for c in range(1, N) if c not in chosen_set and (factor_masks[c] & rest_mask) == 0]
                    if len(candidates) >= 2:
                        c, d = random.sample(candidates, 2)
                        if (factor_masks[c] & factor_masks[d]) == 0:
                            delta = scores[c] + scores[d] - scores[a] - scores[b]
                            if delta > 0 or random.random() < pow(2.71828, delta / temperature):
                                chosen_set.remove(a)
                                chosen_set.remove(b)
                                chosen_set.add(c)
                                chosen_set.add(d)
                                chosen_mask = (chosen_mask & ~factor_masks[a] & ~factor_masks[b]) | factor_masks[c] | factor_masks[d]
                                current_score += delta
            
            temperature *= cooling_rate
            
        return chosen_set, chosen_mask, current_score

    # Run SA from best solution
    sa_set, sa_mask, sa_score = simulated_annealing(best_chosen_set, compute_mask(best_chosen_set), best_score)
    
    # Polish with local search
    sa_set, sa_mask, sa_score = local_search(sa_set, sa_mask, sa_score)
    
    if sa_score > best_score:
        best_score = sa_score
        best_chosen_set = set(sa_set)
        
    return list(best_chosen_set)
