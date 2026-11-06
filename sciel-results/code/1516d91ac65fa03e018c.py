import math
import time
import random

def construct():
    deadline = time.time() + 145.0
    N = 511
    C = 512

    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Precompute prime factors for all numbers 1..511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(C)
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)

    # Precompute conflict masks as bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Greedy initial solution - pick from high a down
    S_mask = 0
    for a in range(N, 0, -1):
        if (S_mask & conflict_mask[a]) == 0:
            S_mask |= (1 << a)

    def mask_to_list(mask):
        result = []
        for a in range(1, N + 1):
            if mask & (1 << a):
                result.append(a)
        return result

    best_S_mask = S_mask
    best_score = sum(w[a] for a in range(1, N + 1) if best_S_mask & (1 << a))

    # Simple upper bound: sum of weights of all available items
    # (loose but fast) - we'll use a tighter one based on greedy selection
    def fast_upper_bound(available_mask, current_score):
        """Greedy upper bound: pick items greedily from available, no conflict check
        beyond what's already excluded"""
        greedy_score = 0.0
        temp_mask = available_mask
        # Limit iterations for speed
        count = 0
        while temp_mask and count < 200:
            best_a = 0
            best_w = -1.0
            # Scan from high a (high weight) down
            for a in range(N, 0, -1):
                if temp_mask & (1 << a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
                        break  # Since we go from high a, first found is best
            if best_a == 0:
                break
            greedy_score += best_w
            temp_mask &= ~(1 << best_a)
            temp_mask &= ~conflict_mask[best_a]
            count += 1
        return current_score + greedy_score

    # Branch and bound with tighter pruning
    nodes_visited = 0
    max_nodes = 8000000

    def branch_and_bound(current_mask, available_mask, current_score, depth):
        nonlocal best_S_mask, best_score, nodes_visited, deadline

        if time.time() > deadline - 5.0:
            return

        nodes_visited += 1
        if nodes_visited > max_nodes:
            return

        if not available_mask:
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_S_mask = current_mask
            return

        # Upper bound check
        ub = fast_upper_bound(available_mask, current_score)
        if ub <= best_score + 1e-12:
            return

        # Select pivot: highest weight available item
        pivot = 0
        for a in range(N, 0, -1):
            if available_mask & (1 << a):
                pivot = a
                break

        if pivot == 0:
            return

        # Branch 1: Include pivot
        new_mask_include = current_mask | (1 << pivot)
        new_available_include = available_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound(new_mask_include, new_available_include, current_score + w[pivot], depth + 1)

        # Branch 2: Exclude pivot
        new_available_exclude = available_mask & ~(1 << pivot)
        branch_and_bound(current_mask, new_available_exclude, current_score, depth + 1)

    # Run branch and bound
    initial_available = 0
    for a in range(1, N + 1):
        initial_available |= (1 << a)

    branch_and_bound(0, initial_available, 0.0, 0)

    # Local search refinement on the best found
    result = mask_to_list(best_S_mask)
    result_set = set(result)
    
    # Phase 1: Try to add any number that is coprime to all in result
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for a in range(N, 0, -1):
            if a in result_set:
                continue
            if (best_S_mask & conflict_mask[a]) == 0:
                result.append(a)
                result_set.add(a)
                best_S_mask |= (1 << a)
                best_score += w[a]
                improved = True
                break  # Restart to find more additions

    # Phase 2: 1-swap local search
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        result_set = set(result)
        
        for i in range(len(result)):
            if time.time() > deadline - 3.0:
                break
            current = result[i]
            current_score = w[current]
            
            # Find required prime factors from other chosen elements
            req_factors = set()
            for j in range(len(result)):
                if j != i:
                    req_factors.update(prime_factors[result[j]])
            
            # Find best replacement
            best_new = -1
            best_new_score = 0.0
            
            for a in range(N, 0, -1):
                if a in result_set:
                    continue
                if a == current:
                    continue
                if prime_factors[a] and prime_factors[a].intersection(req_factors):
                    continue
                if w[a] > current_score + 1e-15:
                    best_new = a
                    best_new_score = w[a]
                    break
            
            if best_new != -1:
                # Perform swap
                result[i] = best_new
                result_set.remove(current)
                result_set.add(best_new)
                best_score = best_score - current_score + best_new_score
                improved = True
                break
    
    # Phase 3: 2-swap local search (remove two, add two)
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        result_set = set(result)
        n = len(result)
        
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3.0:
                    break
                cur_i = result[i]
                cur_j = result[j]
                cur_score = w[cur_i] + w[cur_j]
                
                # Find required prime factors from remaining chosen elements
                req_factors = set()
                for k in range(n):
                    if k != i and k != j:
                        req_factors.update(prime_factors[result[k]])
                
                # Find best pair to replace with
                best_pair = None
                best_pair_score = 0.0
                
                # Try all pairs of candidates
                for a in range(N, 0, -1):
                    if time.time() > deadline - 3.0:
                        break
                    if a in result_set:
                        continue
                    if prime_factors[a] and prime_factors[a].intersection(req_factors):
                        continue
                    # Check a doesn't conflict with itself (always true)
                    # Now try to find b that works with a
                    req_factors_with_a = req_factors | (prime_factors[a] if prime_factors[a] else set())
                    for b in range(a - 1, 0, -1):
                        if b in result_set:
                            continue
                        if b == a:
                            continue
                        if prime_factors[b] and prime_factors[b].intersection(req_factors_with_a):
                            continue
                        # Check a and b are coprime to each other
                        if math.gcd(a, b) > 1:
                            continue
                        pair_score = w[a] + w[b]
                        if pair_score > best_pair_score:
                            best_pair_score = pair_score
                            best_pair = (a, b)
                        # Early termination: if we found a good pair, we can stop
                        # since we're going in descending order
                        if pair_score > cur_score + 1e-15:
                            break
                    if best_pair and best_pair_score > cur_score + 1e-15:
                        break
                
                if best_pair and best_pair_score > cur_score + 1e-15:
                    # Perform 2-swap
                    a_new, b_new = best_pair
                    result[i] = a_new
                    result[j] = b_new
                    result_set.remove(cur_i)
                    result_set.remove(cur_j)
                    result_set.add(a_new)
                    result_set.add(b_new)
                    best_score = best_score - cur_score + best_pair_score
                    improved = True
                    break
            if improved:
                break
    
    # Phase 4: Final greedy addition pass
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for a in range(N, 0, -1):
            if a in result_set:
                continue
            if (best_S_mask & conflict_mask[a]) == 0:
                result.append(a)
                result_set.add(a)
                best_S_mask |= (1 << a)
                best_score += w[a]
                improved = True
                break

    # Update best_S_mask based on final result
    final_S_mask = 0
    for a in result_set:
        final_S_mask |= (1 << a)
    
    return mask_to_list(final_S_mask)
