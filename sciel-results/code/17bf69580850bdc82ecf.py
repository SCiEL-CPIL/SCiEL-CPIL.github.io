import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n) if is_prime[i]]
    
    primes = sieve(N)
    
    # Prime factors for each number
    prime_factors = [set() for _ in range(N)]
    for p in primes:
        for multiple in range(p, N, p):
            prime_factors[multiple].add(p)
    
    # Weight: 1/(512 - a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    def score_set(chosen):
        s = 0.0
        for a in chosen:
            s += weights[a]
        return s
    
    def is_coprime_to_all(a, chosen_set, chosen_factors):
        """Check if a is coprime to all elements in chosen_set using prime factors."""
        if a == 1:
            return True
        for p in prime_factors[a]:
            if p in chosen_factors:
                return False
        return True
    
    # Greedy initialization: pick in order of highest weight (largest a first)
    def greedy_init():
        chosen = []
        chosen_set = set()
        chosen_factors = set()
        
        for a in range(N - 1, 0, -1):
            if is_coprime_to_all(a, chosen_set, chosen_factors):
                chosen.append(a)
                chosen_set.add(a)
                if a > 1:
                    chosen_factors.update(prime_factors[a])
        
        return chosen, chosen_set, chosen_factors
    
    # Get the best greedy initialization
    best_chosen, best_set, best_factors = greedy_init()
    best_score = score_set(best_chosen)
    
    # Helper: update factors after adding/removing
    def add_element(chosen, chosen_set, chosen_factors, a):
        chosen.append(a)
        chosen_set.add(a)
        if a > 1:
            chosen_factors.update(prime_factors[a])
    
    def remove_element(chosen, chosen_set, chosen_factors, a):
        chosen.remove(a)
        chosen_set.remove(a)
        if a > 1:
            # Recompute factors from remaining elements
            new_factors = set()
            for b in chosen:
                if b > 1:
                    new_factors.update(prime_factors[b])
            chosen_factors.clear()
            chosen_factors.update(new_factors)
    
    # Single-swap local search
    def single_swap_search(chosen, chosen_set, chosen_factors, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(chosen)):
                if time.time() > time_limit:
                    break
                
                a_i = chosen[i]
                current_w = weights[a_i]
                
                # Compute factors of all other elements
                other_factors = set()
                for j in range(len(chosen)):
                    if j != i and chosen[j] > 1:
                        other_factors.update(prime_factors[chosen[j]])
                
                # Find best replacement
                best_new = -1
                best_new_w = current_w
                
                for b in range(N - 1, 0, -1):
                    if b in chosen_set:
                        continue
                    if weights[b] <= current_w:
                        break
                    # Check if b is coprime to all other chosen
                    if b == 1:
                        best_new = b
                        best_new_w = weights[b]
                        break
                    conflict = False
                    for p in prime_factors[b]:
                        if p in other_factors:
                            conflict = True
                            break
                    if not conflict:
                        best_new = b
                        best_new_w = weights[b]
                        break
                
                if best_new != -1 and best_new_w > current_w:
                    # Perform swap
                    remove_element(chosen, chosen_set, chosen_factors, a_i)
                    add_element(chosen, chosen_set, chosen_factors, best_new)
                    current_score += best_new_w - current_w
                    improved = True
                    break  # Restart search after modification
        
        return chosen, chosen_set, chosen_factors, current_score
    
    # Multi-swap: try replacing a subset of elements with one new element
    def multi_swap_search(chosen, chosen_set, chosen_factors, current_score, time_limit, max_k=3):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add a new element b by removing conflicting elements
            # For each candidate b not in chosen, find conflicts
            for b in range(N - 1, 0, -1):
                if time.time() > time_limit:
                    break
                if b in chosen_set:
                    continue
                if b == 1:
                    continue  # 1 is already likely in the set or has weight 1/511
                
                # Find which chosen elements conflict with b
                conflicts = []
                for a in chosen:
                    if a == 1:
                        continue
                    # Check if gcd(a, b) > 1
                    if gcd(a, b) > 1:
                        conflicts.append(a)
                
                if not conflicts:
                    # Can add b directly
                    add_element(chosen, chosen_set, chosen_factors, b)
                    current_score += weights[b]
                    improved = True
                    break
                
                # Try removing the conflicting elements and adding b
                # Only worthwhile if weight(b) > sum of weights of conflicts
                conflict_weight = sum(weights[a] for a in conflicts)
                if weights[b] > conflict_weight + 1e-12:
                    # Check that after removing conflicts, b is coprime to all remaining
                    remaining_factors = set()
                    for a in chosen:
                        if a not in conflicts and a > 1:
                            remaining_factors.update(prime_factors[a])
                    
                    b_conflict = False
                    for p in prime_factors[b]:
                        if p in remaining_factors:
                            b_conflict = True
                            break
                    
                    if not b_conflict:
                        # Perform the multi-swap
                        for a in conflicts:
                            remove_element(chosen, chosen_set, chosen_factors, a)
                        add_element(chosen, chosen_set, chosen_factors, b)
                        current_score += weights[b] - conflict_weight
                        improved = True
                        break
        
        return chosen, chosen_set, chosen_factors, current_score
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # Init 1: Standard greedy (largest a first)
    init1, set1, fac1 = greedy_init()
    initializations.append((init1, set1, fac1))
    
    # Init 2: Greedy with random perturbation
    import random
    random.seed(123)
    for _ in range(10):
        # Randomly shuffle the order of candidates
        candidates = list(range(N - 1, 0, -1))
        random.shuffle(candidates)
        # But keep large numbers more likely to be picked early
        # Weighted shuffle: higher numbers have higher probability
        chosen = []
        chosen_set = set()
        chosen_factors = set()
        
        for a in candidates:
            if is_coprime_to_all(a, chosen_set, chosen_factors):
                chosen.append(a)
                chosen_set.add(a)
                if a > 1:
                    chosen_factors.update(prime_factors[a])
        
        initializations.append((chosen, chosen_set, chosen_factors))
    
    # Evaluate all initializations and pick the best
    best_init = None
    best_init_score = -1
    for chosen, cset, cfacs in initializations:
        s = score_set(chosen)
        if s > best_init_score:
            best_init_score = s
            best_init = (chosen, cset, cfacs)
    
    best_chosen, best_set, best_factors = best_init
    best_score = best_init_score
    
    # Apply local search: alternate between single-swap and multi-swap
    search_time_limit = deadline - 2.0  # Leave 2s for final operations
    
    while time.time() < search_time_limit:
        # Single-swap pass
        best_chosen, best_set, best_factors, best_score = single_swap_search(
            best_chosen, best_set, best_factors, best_score, search_time_limit
        )
        
        if time.time() >= search_time_limit:
            break
        
        # Multi-swap pass
        best_chosen, best_set, best_factors, best_score = multi_swap_search(
            best_chosen, best_set, best_factors, best_score, search_time_limit, max_k=3
        )
        
        # Check if we can still improve
        if time.time() >= search_time_limit:
            break
    
    return best_chosen
