import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    N = 512
    max_a = N - 1
    
    # Precompute prime factors for each number
    # p_factors[a] will be a set of prime factors of a
    p_factors = [set() for _ in range(max_a + 1)]
    for i in range(2, max_a + 1):
        if not p_factors[i]: # i is prime
            for j in range(i, max_a + 1, i):
                p_factors[j].add(i)
    # 1 has no prime factors, so p_factors[1] is empty set, which is correct for coprime logic (gcd(1, x)=1)

    # Weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)

    # Greedy Initialization
    # Sort by weight descending
    candidates = list(range(1, max_a + 1))
    candidates.sort(key=lambda x: -weights[x])
    
    selected = []
    selected_set = set()
    used_primes = set()
    
    for a in candidates:
        if a in selected_set:
            continue
        # Check if a's prime factors are disjoint from used_primes
        if p_factors[a].isdisjoint(used_primes):
            selected.append(a)
            selected_set.add(a)
            used_primes.update(p_factors[a])
            
    def get_score(lst):
        return sum(weights[a] for a in lst)
        
    current_score = get_score(selected)
    
    # Helper to check if a number can be added to a set of selected numbers
    # excluding a specific subset of selected numbers (for swaps)
    def is_compatible(a, selected_list, exclude_indices):
        # Check against all selected numbers except those at exclude_indices
        for i, b in enumerate(selected_list):
            if i in exclude_indices:
                continue
            if not p_factors[a].isdisjoint(p_factors[b]):
                return False
        return True

    # Local Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # 1. Try to add a new number (if not full)
        # This is less likely to help if greedy was good, but let's check
        # Actually, if we have a valid set, adding a number requires its primes to be unused.
        # Greedy already did this. But maybe a different order?
        # Let's stick to swaps which are more powerful.

        # 2. 1-Swap: Replace one selected a with one unselected b
        for i in range(len(selected)):
            a = selected[i]
            best_gain = 0
            best_b = None
            for b in range(1, max_a + 1):
                if b in selected_set:
                    continue
                # b must be compatible with all selected except a
                if not is_compatible(b, selected, {i}):
                    continue
                gain = weights[b] - weights[a]
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b is not None and best_gain > 0:
                # Perform swap
                selected_set.remove(a)
                selected_set.add(best_b)
                # Update used_primes: remove a's primes, add best_b's primes
                # Note: We must ensure no other selected number shares primes with best_b.
                # is_compatible already checked this.
                # But we need to update used_primes correctly.
                # Since it's a 1-for-1 swap, and we checked compatibility with others,
                # we can just update the set.
                # However, 'used_primes' is a global set. 
                # If a prime p is in p_factors[a] and p_factors[other], it would have been caught by greedy/previous steps.
                # So p_factors[a] and p_factors[other] are disjoint for all other.
                # Thus, removing p_factors[a] is safe.
                used_primes.difference_update(p_factors[a])
                used_primes.update(p_factors[best_b])
                
                selected[i] = best_b
                current_score += best_gain
                improved = True
                if (time.time() - start_time) > deadline:
                    return selected

        # 3. 2-Swap: Replace two selected {a1, a2} with two unselected {b1, b2}
        # This allows merging small primes into large composites.
        if (time.time() - start_time) > deadline:
            return selected
            
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a1 = selected[i]
                a2 = selected[j]
                exclude = {i, j}
                
                # Find best pair {b1, b2}
                # To keep it fast, we can iterate b1 and for each valid b1, find best b2
                # Or just iterate pairs if N is small enough.
                # N=511, selected size is small (~10-20).
                # Number of unselected is ~500. Pairs ~ 125k.
                # For each pair, check compatibility with ~10 others.
                # 125k * 10 * 2 (gcd checks or set disjoint) ~ 2.5M ops per outer loop iteration.
                # Outer loop is ~100. Total 250M ops. Might be too slow in Python.
                
                # Optimization: Pre-filter b candidates?
                # Let's try a smarter 2-swap:
                # For each unselected b1, check if it can replace a1 or a2 or both?
                # Actually, let's just iterate b1, then b2 > b1.
                
                best_gain = 0
                best_b1 = None
                best_b2 = None
                
                # Iterate b1
                for b1 in range(1, max_a + 1):
                    if b1 in selected_set:
                        continue
                    # b1 must be compatible with all selected except a1, a2
                    if not is_compatible(b1, selected, exclude):
                        continue
                    
                    # Now find best b2
                    # b2 must be:
                    # 1. Not in selected_set
                    # 2. b2 > b1 (to avoid duplicates)
                    # 3. Compatible with all selected except a1, a2
                    # 4. Compatible with b1 (p_factors[b1].isdisjoint(p_factors[b2]))
                    
                    w_b1 = weights[b1]
                    base_gain = w_b1 - weights[a1] - weights[a2] # This is not right, we need sum of weights of b1, b2 - sum of a1, a2
                    
                    # Let's just find the best b2 for this b1
                    best_b2_for_b1 = None
                    best_w_b2 = -1
                    
                    for b2 in range(b1 + 1, max_a + 1):
                        if b2 in selected_set:
                            continue
                        if not p_factors[b1].isdisjoint(p_factors[b2]):
                            continue
                        if not is_compatible(b2, selected, exclude):
                            continue
                        
                        if weights[b2] > best_w_b2:
                            best_w_b2 = weights[b2]
                            best_b2_for_b1 = b2
                            
                    if best_b2_for_b1 is not None:
                        gain = weights[b1] + weights[best_b2_for_b1] - weights[a1] - weights[a2]
                        if gain > best_gain:
                            best_gain = gain
                            best_b1 = b1
                            best_b2 = best_b2_for_b1

                if best_b1 is not None and best_gain > 0:
                    # Perform swap
                    # Remove a1, a2
                    selected_set.remove(a1)
                    selected_set.remove(a2)
                    used_primes.difference_update(p_factors[a1])
                    used_primes.difference_update(p_factors[a2])
                    
                    # Add b1, b2
                    selected_set.add(best_b1)
                    selected_set.add(best_b2)
                    used_primes.update(p_factors[best_b1])
                    used_primes.update(p_factors[best_b2])
                    
                    # Update selected list
                    # Replace indices i and j
                    selected[i] = best_b1
                    selected[j] = best_b2
                    # Note: The order in 'selected' doesn't matter for the final return, 
                    # but for the next iterations, it's just a list.
                    
                    current_score += best_gain
                    improved = True
                    if (time.time() - start_time) > deadline:
                        return selected

    return selected
