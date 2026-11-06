"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    from functools import lru_cache
    
    deadline = time.time() + 130  # safety margin before 150s limit
    
    N = 512
    # Precompute score contribution
    score_contrib = [0.0] * N
    for a in range(1, N):
        score_contrib[a] = 1.0 / (N - a)
    
    # Build conflict graph using bitsets for efficiency
    # conflict[a] is a set of integers that are NOT coprime with a
    conflict = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a+1, N):
            if gcd(a, b) != 1:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Convert to sorted lists for faster iteration
    conflict_list = [sorted(conflict[a]) for a in range(N)]
    
    def is_coprime_to_all(a, chosen_list):
        for b in chosen_list:
            if b in conflict[a]:
                return False
        return True
    
    # Phase 1: Greedy initialization - pick from high to low weight
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    def current_score():
        return sum(score_contrib[a] for a in chosen)
    
    best_score = current_score()
    best_chosen = chosen[:]
    
    # Phase 2: Local search - try to improve via swaps and additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any unchosen element
        for a in range(511, 0, -1):
            if a not in chosen_set:
                if is_coprime_to_all(a, chosen):
                    new_score = current_score() + score_contrib[a]
                    if new_score > best_score + 1e-15:
                        chosen.append(a)
                        chosen_set.add(a)
                        best_score = new_score
                        best_chosen = chosen[:]
                        improved = True
                    break
        
        # Try swaps: replace one chosen element with one unchosen element
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            old_a = chosen[i]
            for new_a in range(511, 0, -1):
                if new_a in chosen_set:
                    continue
                coprime = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    if gcd(new_a, chosen[j]) != 1:
                        coprime = False
                        break
                if not coprime:
                    continue
                delta = score_contrib[new_a] - score_contrib[old_a]
                if delta > 1e-15:
                    chosen[i] = new_a
                    chosen_set.remove(old_a)
                    chosen_set.add(new_a)
                    best_score += delta
                    best_chosen = chosen[:]
                    improved = True
                    break
    
    # Phase 3: Branch and bound with bitset-based pruning
    # Use a more sophisticated approach: try to find if we can beat best_score
    # by exploring the search space with upper bound pruning
    
    # Sort vertices by weight descending
    vertices = list(range(1, N))
    vertices.sort(key=lambda x: -score_contrib[x])
    
    # Precompute for each vertex, the set of vertices that can follow it
    # (i.e., are coprime to it) - but we'll compute on the fly with bitsets
    
    # Use a recursive branch and bound with memoization of upper bounds
    # Upper bound: sum of weights of all remaining vertices that are coprime to chosen
    
    def branch_and_bound(chosen_set, chosen_list, current_weight, remaining_vertices):
        """
        remaining_vertices: list of vertices not yet decided (can include or exclude)
        Returns the best additional weight achievable from remaining_vertices
        given that chosen_set is already selected.
        """
        nonlocal best_score, best_chosen, deadline
        
        if time.time() > deadline:
            return 0.0
        
        if not remaining_vertices:
            return 0.0
        
        # Upper bound: sum of all weights in remaining_vertices that are coprime to all chosen
        # This is an admissible upper bound (relaxation)
        ub = 0.0
        feasible_remaining = []
        for v in remaining_vertices:
            if v not in chosen_set:
                if all(gcd(v, c) == 1 for c in chosen_list):
                    ub += score_contrib[v]
                    feasible_remaining.append(v)
        
        if ub <= 1e-15:
            return 0.0
        
        # If current_weight + ub <= best_score, prune
        if current_weight + ub <= best_score + 1e-15:
            return 0.0
        
        # Try to include the first feasible vertex
        if not feasible_remaining:
            return 0.0
        
        first_v = feasible_remaining[0]
        
        # Option 1: Include first_v
        new_chosen_set = chosen_set | {first_v}
        new_chosen_list = chosen_list + [first_v]
        new_remaining = [v for v in remaining_vertices if v != first_v]
        
        # Filter new_remaining to only those coprime to first_v
        new_remaining_filtered = [v for v in new_remaining if gcd(v, first_v) == 1]
        
        include_val = score_contrib[first_v] + branch_and_bound(
            new_chosen_set, new_chosen_list, current_weight + score_contrib[first_v], new_remaining_filtered
        )
        
        # Option 2: Exclude first_v
        exclude_val = branch_and_bound(
            chosen_set, chosen_list, current_weight, [v for v in remaining_vertices if v != first_v]
        )
        
        return max(include_val, exclude_val)
    
    # The branch and bound above is too slow for 511 vertices.
    # Instead, let's do a more targeted local search with multiple restarts.
    
    # Multiple restarts with different greedy orders
    for restart in range(5):
        if time.time() >= deadline:
            break
        
        # Different greedy strategies
        if restart == 0:
            # Original: highest weight first
            order = list(range(511, 0, -1))
        elif restart == 1:
            # Random shuffle
            import random
            random.seed(42 + restart)
            order = list(range(1, N))
            random.shuffle(order)
        elif restart == 2:
            # Lowest weight first (counterintuitive but may find different local optima)
            order = list(range(1, N))
        elif restart == 3:
            # Primes first
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [a for a in range(1, N) if is_prime(a)]
            non_primes = [a for a in range(1, N) if not is_prime(a)]
            order = primes + non_primes
        else:
            # Weighted random
            import random
            random.seed(123 + restart)
            order = list(range(1, N))
            random.shuffle(order)
        
        # Greedy construction
        chosen_r = []
        chosen_set_r = set()
        for a in order:
            if all(gcd(a, b) == 1 for b in chosen_r):
                chosen_r.append(a)
                chosen_set_r.add(a)
        
        # Local search on this solution
        improved_r = True
        while improved_r and time.time() < deadline:
            improved_r = False
            
            # Try to add
            for a in order:
                if a not in chosen_set_r:
                    if is_coprime_to_all(a, chosen_r):
                        new_score = current_score() + score_contrib[a]
                        if new_score > best_score + 1e-15:
                            chosen_r.append(a)
                            chosen_set_r.add(a)
                            best_score = new_score
                            best_chosen = chosen_r[:]
                            improved_r = True
                        break
            
            # Try swaps
            for i in range(len(chosen_r)):
                if time.time() >= deadline:
                    break
                old_a = chosen_r[i]
                for new_a in order:
                    if new_a in chosen_set_r:
                        continue
                    coprime = True
                    for j in range(len(chosen_r)):
                        if j == i:
                            continue
                        if gcd(new_a, chosen_r[j]) != 1:
                            coprime = False
                            break
                    if not coprime:
                        continue
                    delta = score_contrib[new_a] - score_contrib[old_a]
                    if delta > 1e-15:
                        chosen_r[i] = new_a
                        chosen_set_r.remove(old_a)
                        chosen_set_r.add(new_a)
                        best_score += delta
                        best_chosen = chosen_r[:]
                        improved_r = True
                        break
    
    # Phase 4: Final high-precision polish
    # Try all possible single-element improvements with exact rational arithmetic
    from fractions import Fraction
    
    # Convert best_chosen to a set for fast lookup
    best_set = set(best_chosen)
    
    # Try adding each possible element
    for a in range(511, 0, -1):
        if a not in best_set:
            if is_coprime_to_all(a, best_chosen):
                new_solution = best_chosen + [a]
                new_score = sum(score_contrib[x] for x in new_solution)
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_chosen = new_solution
                    best_set.add(a)
    
    # Try all possible swaps
    for i in range(len(best_chosen)):
        old_a = best_chosen[i]
        for new_a in range(511, 0, -1):
            if new_a in best_set:
                continue
            coprime = True
            for j in range(len(best_chosen)):
                if j == i:
                    continue
                if gcd(new_a, best_chosen[j]) != 1:
                    coprime = False
                    break
            if not coprime:
                continue
            delta = score_contrib[new_a] - score_contrib[old_a]
            if delta > 1e-15:
                best_chosen[i] = new_a
                best_set.remove(old_a)
                best_set.add(new_a)
                best_score += delta
    
    return best_chosen
# EVOLVE-BLOCK-END
