import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
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
                if gcd_table[selected[i]][selected[j]] != 1:
                    return False
        return True
    
    def greedy_solve(prime_order):
        """Solve using greedy with given prime order."""
        selected = []
        selected_set = set()
        for p in prime_order:
            best_val = 0
            max_k = N // p
            for k in range(max_k, 0, -1):
                candidate = p * k
                if candidate in selected_set:
                    continue
                is_coprime = True
                for s in selected:
                    if gcd_table[candidate][s] != 1:
                        is_coprime = False
                        break
                if is_coprime:
                    best_val = candidate
                    break
            if best_val > 0:
                selected.append(best_val)
                selected_set.add(best_val)
        selected.append(1)
        selected_set.add(1)
        return selected, selected_set
    
    def try_merge(selected, selected_set):
        """Try to merge pairs of selected elements into their product if it fits and is coprime to all others."""
        improved = True
        while improved:
            improved = False
            n = len(selected)
            for i in range(n):
                if selected[i] == 1:
                    continue
                for j in range(i + 1, n):
                    if selected[j] == 1:
                        continue
                    a, b = selected[i], selected[j]
                    if gcd_table[a][b] != 1:
                        continue
                    product = a * b
                    if product > N:
                        continue
                    if product in selected_set:
                        continue
                    # Check if product is coprime to all other selected
                    ok = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if gcd_table[product][selected[k]] != 1:
                            ok = False
                            break
                    if ok:
                        # Remove i and j, add product
                        new_selected = [x for k, x in enumerate(selected) if k != i and k != j]
                        new_selected.append(product)
                        new_set = set(new_selected)
                        if sum(new_selected) > sum(selected):
                            selected = new_selected
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
        return selected, selected_set
    
    def local_search(selected, selected_set):
        """Try to improve via local swaps using precomputed gcd table."""
        improved = True
        while improved:
            improved = False
            n = len(selected)
            for i in range(n):
                if selected[i] == 1:
                    continue
                orig = selected[i]
                others = [x for j, x in enumerate(selected) if j != i]
                # Try values from N down to orig+1
                for candidate in range(N, orig, -1):
                    if candidate in selected_set:
                        continue
                    ok = True
                    for o in others:
                        if gcd_table[candidate][o] != 1:
                            ok = False
                            break
                    if ok:
                        selected[i] = candidate
                        selected_set.discard(orig)
                        selected_set.add(candidate)
                        improved = True
                        break
        return selected, selected_set
    
    best_sum = -1
    best_selected = None
    
    def evaluate(sel):
        nonlocal best_sum, best_selected
        if not is_valid_pairwise_coprime(sel):
            return
        s = sum(sel)
        if s > best_sum:
            best_sum = s
            best_selected = list(sel)
    
    # Try multiple orderings
    # Ordering 1: descending
    order1 = sorted(primes, reverse=True)
    sel1, set1 = greedy_solve(order1)
    sel1, set1 = try_merge(sel1, set1)
    sel1, set1 = local_search(sel1, set1)
    evaluate(sel1)
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 2: ascending
    order2 = sorted(primes)
    sel2, set2 = greedy_solve(order2)
    sel2, set2 = try_merge(sel2, set2)
    sel2, set2 = local_search(sel2, set2)
    evaluate(sel2)
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 3: by number of multiples (most constrained first - fewest multiples)
    order3 = sorted(primes, key=lambda p: N // p)
    sel3, set3 = greedy_solve(order3)
    sel3, set3 = try_merge(sel3, set3)
    sel3, set3 = local_search(sel3, set3)
    evaluate(sel3)
    
    if time.time() > deadline - 5:
        if best_selected is not None:
            return best_selected
        return [1] + primes
    
    # Ordering 4: random
    import random
    rng = random.Random(42)
    order4 = primes[:]
    rng.shuffle(order4)
    sel4, set4 = greedy_solve(order4)
    sel4, set4 = try_merge(sel4, set4)
    sel4, set4 = local_search(sel4, set4)
    evaluate(sel4)
    
    # Try more random orderings if time permits
    for seed in range(1, 500):
        if time.time() > deadline - 3:
            break
        rng2 = random.Random(seed)
        order = primes[:]
        rng2.shuffle(order)
        sel, selset = greedy_solve(order)
        sel, selset = try_merge(sel, selset)
        sel, selset = local_search(sel, selset)
        evaluate(sel)
    
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
