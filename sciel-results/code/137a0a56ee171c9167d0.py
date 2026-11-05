import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Precompute weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    # Precompute prime factors for each number
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    prime_factors = [set() for _ in range(512)]
    for a in range(1, 512):
        prime_factors[a] = get_prime_factors(a)
        
    # Sort candidates by weight descending (largest a first)
    candidates = list(range(511, 0, -1))
    
    # Global best
    best_score = 0.0
    best_set = []
    
    # Branch and Bound
    # To make B&B efficient, we need a good upper bound.
    # Upper bound for a state (current_set, idx):
    # current_score + sum(weights[c] for c in candidates[idx:] if coprime(c, current_set))
    # This is expensive to compute every time.
    # Optimization: Precompute coprime masks? 
    # Since N is small, we can just compute it.
    
    # Let's use a recursive search with memoization or just simple B&B.
    # Given the tree size, simple B&B with pruning is good.
    
    def is_coprime_to_set(a, used_primes):
        return prime_factors[a].isdisjoint(used_primes)
        
    def search(idx, current_set, current_score, used_primes):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound
        # Calculate optimistic upper bound
        # Sum of weights of all remaining candidates that are coprime to current_set
        # Note: This ignores that remaining candidates must be pairwise coprime, so it's an overestimate.
        ub = current_score
        for k in range(idx, len(candidates)):
            c = candidates[k]
            if is_coprime_to_set(c, used_primes):
                ub += weights[c]
                
        if ub <= best_score + 1e-12:
            return
            
        if idx == len(candidates):
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            return
            
        c = candidates[idx]
        
        # Option 1: Include c
        if is_coprime_to_set(c, used_primes):
            new_used = used_primes | prime_factors[c]
            search(idx + 1, current_set + [c], current_score + weights[c], new_used)
            
        # Option 2: Exclude c
        search(idx + 1, current_set, current_score, used_primes)
        
    # Initial call
    search(0, [], 0.0, set())
    
    # Local Search Polish
    # The B&B might find the global optimum, but if it times out or prunes too aggressively due to floating point issues, 
    # a local search on the best found set can help.
    
    if best_set:
        # Try to improve by swapping or replacing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Move 1: Swap one element for another
            for i, a in enumerate(best_set):
                # Calculate score without a
                score_without_a = best_score - weights[a]
                used_primes_without_a = set().union(*[prime_factors[x] for x in best_set if x != a]) if len(best_set) > 1 else set()
                
                # Find best replacement b
                best_b = None
                best_gain = 0
                for b in range(511, 0, -1):
                    if b in best_set:
                        continue
                    if not prime_factors[b].isdisjoint(used_primes_without_a):
                        continue
                    gain = weights[b]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                        
                if best_b is not None and score_without_a + best_gain > best_score + 1e-12:
                    new_set = [x for j, x in enumerate(best_set) if j != i]
                    new_set.append(best_b)
                    best_set = new_set
                    best_score = score_without_a + best_gain
                    improved = True
                    break
                    
            # Move 2: Add one element if possible (shouldn't happen if B&B was exhaustive, but check just in case)
            # If B&B was exhaustive, we can't add. If it timed out, we might be able to.
            for b in range(511, 0, -1):
                if b in best_set:
                    continue
                if prime_factors[b].isdisjoint(set().union(*[prime_factors[x] for x in best_set])):
                    new_score = best_score + weights[b]
                    if new_score > best_score + 1e-12:
                        best_set.append(b)
                        best_score = new_score
                        improved = True
                        break
                        
            # Move 3: Merge two elements into their product if beneficial and < 512
            # This is less likely to improve if we are maximizing 1/(512-a), as merging reduces the count of large numbers.
            # w(a*b) vs w(a) + w(b). 
            # 1/(512-ab) vs 1/(512-a) + 1/(512-b).
            # Usually 1/(512-a) + 1/(512-b) > 1/(512-ab) for a,b > 1.
            # Example: a=2, b=3. w(2)=1/510, w(3)=1/509. w(6)=1/506.
            # 1/510 + 1/509 ~ 0.0039. 1/506 ~ 0.0019. So merging is bad.
            # So we skip merge.
            
    return best_set

# Test locally
if __name__ == "__main__":
    result = construct()
    score = sum(1.0 / (512 - a) for a in result)
    print(f"Score: {score}")
    print(f"Set: {sorted(result, reverse=True)}")
