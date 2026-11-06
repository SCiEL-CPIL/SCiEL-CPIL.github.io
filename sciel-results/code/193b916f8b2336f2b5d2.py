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

    # Greedy initial solution
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

    def better_upper_bound(available_mask, current_score):
        """Better upper bound using greedy coloring heuristic on the subgraph"""
        greedy_score = 0.0
        temp_mask = available_mask
        while temp_mask:
            best_a = 0
            best_w = -1.0
            for a in range(N, 0, -1):
                if temp_mask & (1 << a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
            if best_a == 0:
                break
            greedy_score += best_w
            temp_mask &= ~(1 << best_a)
            temp_mask &= ~conflict_mask[best_a]
        return current_score + greedy_score

    # Branch and bound
    nodes_visited = 0
    max_nodes = 5000000

    def branch_and_bound(current_mask, available_mask, current_score, depth):
        nonlocal best_S_mask, best_score, nodes_visited, deadline

        if time.time() > deadline - 5.0:
            return

        nodes_visited += 1
        if nodes_visited > max_nodes:
            return

        if not available_mask:
            if current_score > best_score:
                best_score = current_score
                best_S_mask = current_mask
            return

        ub = better_upper_bound(available_mask, current_score)
        if ub <= best_score + 1e-12:
            return

        pivot = 0
        pivot_w = -1.0
        for a in range(N, 0, -1):
            if available_mask & (1 << a):
                if w[a] > pivot_w:
                    pivot_w = w[a]
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

    # Local search refinement
    # Convert best_S_mask to list
    result = mask_to_list(best_S_mask)
    result_set = set(result)
    
    # Try to add any number that is coprime to all in result
    for a in range(N, 0, -1):
        if a in result_set:
            continue
        if (best_S_mask & conflict_mask[a]) == 0:
            result.append(a)
            result_set.add(a)
            best_S_mask |= (1 << a)
            best_score += w[a]

    # Try swaps: for each element in result, try to replace with a better one
    # Sort by weight ascending (lowest first) to potentially swap out low-value items
    result_sorted = sorted(result, key=lambda x: w[x])
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        result_set = set(result)
        
        for i in range(len(result_sorted)):
            if time.time() > deadline - 2.0:
                break
            current = result_sorted[i]
            current_score = w[current]
            
            # Find required prime factors from other chosen elements
            req_factors = set()
            for j in range(len(result_sorted)):
                if j != i:
                    req_factors.update(prime_factors[result_sorted[j]])
            
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
                if w[a] > current_score:
                    best_new = a
                    best_new_score = w[a]
                    break
            
            if best_new != -1:
                # Perform swap
                result_sorted[i] = best_new
                result_set.remove(current)
                result_set.add(best_new)
                best_score = best_score - current_score + best_new_score
                improved = True
                # Re-sort and restart
                result_sorted = sorted(result_set, key=lambda x: w[x])
                break
    
    # Update best_S_mask based on result_set
    final_S_mask = 0
    for a in result_set:
        final_S_mask |= (1 << a)
    
    return mask_to_list(final_S_mask)
