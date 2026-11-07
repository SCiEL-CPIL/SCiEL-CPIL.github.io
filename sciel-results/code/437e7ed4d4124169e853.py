import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    def is_valid_pairwise_coprime(selected):
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if gcd(selected[i], selected[j]) != 1:
                    return False
        return True
    
    def greedy_solve(prime_order):
        """Solve using greedy with given prime order."""
        selected = []
        for p in prime_order:
            best_val = 0
            max_k = N // p
            for k in range(max_k, 0, -1):
                candidate = p * k
                is_coprime = True
                for s in selected:
                    if gcd(candidate, s) != 1:
                        is_coprime = False
                        break
                if is_coprime:
                    best_val = candidate
                    break
            if best_val > 0:
                selected.append(best_val)
        selected.append(1)
        return selected
    
    def local_search(selected):
        """Try to improve via local swaps."""
        selected = list(selected)
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                if selected[i] == 1:
                    continue
                # Find the prime that selected[i] is a multiple of
                # (the "owner" prime in our construction)
                # Actually, let's just try replacing selected[i] with other values
                # that are coprime to all others
                orig = selected[i]
                # Try to find a better value coprime to all other selected
                others = [x for j, x in enumerate(selected) if j != i]
                # Try values from N down to 1, skipping those already in selected
                for candidate in range(N, 0, -1):
                    if candidate in others:
                        continue
                    # Check if candidate is coprime to all others
                    ok = True
                    for o in others:
                        if gcd(candidate, o) != 1:
                            ok = False
                            break
                    if ok and candidate > orig:
                        selected[i] = candidate
                        improved = True
                        break
        return selected
    
    # Try multiple orderings
    best_sum = -1
    best_selected = None
    
    # Ordering 1: descending
    order1 = sorted(primes, reverse=True)
    sel1 = greedy_solve(order1)
    sel1 = local_search(sel1)
    if is_valid_pairwise_coprime(sel1):
        s1 = sum(sel1)
        if s1 > best_sum:
            best_sum = s1
            best_selected = sel1
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 2: ascending
    order2 = sorted(primes)
    sel2 = greedy_solve(order2)
    sel2 = local_search(sel2)
    if is_valid_pairwise_coprime(sel2):
        s2 = sum(sel2)
        if s2 > best_sum:
            best_sum = s2
            best_selected = sel2
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 3: by number of multiples (ascending - most constrained first)
    def num_multiples(p):
        return N // p
    order3 = sorted(primes, key=num_multiples, reverse=True)  # Most multiples last, fewest first
    sel3 = greedy_solve(order3)
    sel3 = local_search(sel3)
    if is_valid_pairwise_coprime(sel3):
        s3 = sum(sel3)
        if s3 > best_sum:
            best_sum = s3
            best_selected = sel3
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 4: random-ish (use a fixed seed for reproducibility)
    import random
    rng = random.Random(42)
    order4 = primes[:]
    rng.shuffle(order4)
    sel4 = greedy_solve(order4)
    sel4 = local_search(sel4)
    if is_valid_pairwise_coprime(sel4):
        s4 = sum(sel4)
        if s4 > best_sum:
            best_sum = s4
            best_selected = sel4
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Try more random orderings if time permits
    for seed in range(1, 100):
        if time.time() > deadline - 3:
            break
        rng2 = random.Random(seed)
        order = primes[:]
        rng2.shuffle(order)
        sel = greedy_solve(order)
        sel = local_search(sel)
        if is_valid_pairwise_coprime(sel):
            s = sum(sel)
            if s > best_sum:
                best_sum = s
                best_selected = sel
    
    if best_selected is None:
        return [1] + primes
    
    # Final validation
    if len(set(best_selected)) != len(best_selected):
        return [1] + primes
    for x in best_selected:
        if x < 1 or x > N:
            return [1] + primes
    if not is_valid_pairwise_coprime(best_selected):
        return [1] + primes
    
    return best_selected
