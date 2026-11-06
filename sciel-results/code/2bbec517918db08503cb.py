"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _num_omega(n):
    """Count distinct prime factors."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))

    # Precompute LCM table for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def get_lcm(a, b):
        return lcm_table[a][b]

    # Precompute bad_pairs[x]: list of (a, b) such that (a, b, x) is a bad triple
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_pairs = [[] for _ in range(n + 1)]
    # Also precompute for each pair (a,b), which c makes it bad
    # For fast checking: for each x, store bitmasks of pairs that conflict with x
    
    # For each triple check
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    bad_pairs[a].append((b, c))
                    bad_pairs[b].append((a, c))
                    bad_pairs[c].append((a, b))

    # For each element x, precompute a bitmask representation
    # We'll use a different approach: for each x, store the list of pairs
    # and use a set of selected elements for fast membership

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a, b in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    # Greedy with a given order
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected

    # Local search: remove-then-add-back strategy
    def local_search(initial_set, max_iterations=3000):
        selected_set = set(initial_set)
        best_size = len(selected_set)
        best_set = set(selected_set)
        all_elements = list(universe)
        no_improve_count = 0
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Determine kick size - adapt based on progress
            if no_improve_count < 100:
                kick_size = random.randint(1, 3)
            elif no_improve_count < 300:
                kick_size = random.randint(2, 5)
            else:
                kick_size = random.randint(3, 8)
            
            if kick_size > len(selected_set) - 1:
                kick_size = max(1, len(selected_set) - 1)
                
            if kick_size <= 0:
                continue
                
            # Select elements to remove - prefer those with many conflicts
            candidates_to_remove = list(selected_set)
            
            # Score each element by how many potential additions it blocks
            def conflict_score(x):
                score = 0
                for a, b in bad_pairs[x]:
                    if a in selected_set and b in selected_set:
                        score += 10  # already conflicting pair
                    elif a in selected_set or b in selected_set:
                        score += 3   # partially blocking
                    else:
                        score += 1   # could block future additions
                return score
            
            # Sort by conflict score (high first) and pick from top
            scored = [(conflict_score(x), x) for x in candidates_to_remove]
            scored.sort(key=lambda t: -t[0])
            
            # Pick from top candidates with some randomness
            top_k = min(kick_size * 3, len(scored))
            top_candidates = [x for _, x in scored[:top_k]]
            
            removed = random.sample(top_candidates, min(kick_size, len(top_candidates)))
            remaining_set = selected_set - set(removed)
            
            # Candidates to add back
            candidates = [x for x in all_elements if x not in remaining_set]
            
            # Order candidates by how "easy" they are to add
            def ease_score(x):
                # Fewer bad pairs means easier to add
                return len(bad_pairs[x])
            
            candidates.sort(key=ease_score)
            
            # Greedily add
            new_set = set(remaining_set)
            for x in candidates:
                if can_add(x, new_set):
                    new_set.add(x)
            
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set
                selected_set = new_set
                no_improve_count = 0
            elif len(new_set) == best_size and random.random() < 0.3:
                # Accept same-size solutions sometimes for diversity
                selected_set = new_set
                no_improve_count += 1
            else:
                no_improve_count += 1
                # If stuck for too long, do a bigger kick
                if no_improve_count > 500:
                    big_kick = random.randint(5, 10)
                    if big_kick < len(selected_set):
                        removed = random.sample(list(selected_set), big_kick)
                        remaining_set = selected_set - set(removed)
                        candidates = [x for x in all_elements if x not in remaining_set]
                        random.shuffle(candidates)
                        new_set = set(remaining_set)
                        for x in candidates:
                            if can_add(x, new_set):
                                new_set.add(x)
                        selected_set = new_set
                        no_improve_count = 0
            
        return list(best_set)

    # Generate multiple orderings
    def make_order(seed, strategy):
        if strategy == 0:
            # Natural order
            return list(universe)
        elif strategy == 1:
            # By number of prime factors (ascending)
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            # Random
            random.seed(seed)
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            # Reverse natural
            return list(reversed(universe))
        elif strategy == 4:
            # By value descending, then omega
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
            # Random with bias toward low omega
            random.seed(seed)
            weights = [1.0 / (1 + _num_omega(x)) for x in universe]
            order = []
            remaining = list(universe)
            remaining_weights = weights[:]
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        order.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            return order
        elif strategy == 6:
            # By degree in conflict graph (ascending)
            return sorted(universe, key=lambda x: (len(bad_pairs[x]), x))
        elif strategy == 7:
            # By degree in conflict graph (descending)
            return sorted(universe, key=lambda x: (-len(bad_pairs[x]), x))
        return list(universe)

    # Run multiple greedy passes with different strategies
    best_result = []
    best_size = 0

    strategies = [0, 1, 3, 4, 6, 7]
    for s in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = make_order(0, s)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Randomized greedy passes
    for i in range(80):
        if time.time() - start_time > DEADLINE - 8:
            break
        order = make_order(i * 12345 + 67890, 2)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Biased random greedy passes
    for i in range(30):
        if time.time() - start_time > DEADLINE - 8:
            break
        order = make_order(i * 99999 + 1, 5)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Local search on the best result
    if time.time() - start_time < DEADLINE - 5:
        improved = local_search(best_result, max_iterations=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Additional local search on other good results
    if time.time() - start_time < DEADLINE - 8:
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            order = make_order(100000 + attempt * 7777, 2)
            result = greedy(order)
            if len(result) >= best_size - 3:  # only if close to best
                improved = local_search(result, max_iterations=3000)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_result = improved

    # Final local search polish on best
    if time.time() - start_time < DEADLINE - 2:
        improved = local_search(best_result, max_iterations=3000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # One more round of diverse local searches
    if time.time() - start_time < DEADLINE - 3:
        for attempt in range(10):
            if time.time() - start_time > DEADLINE - 1:
                break
            # Start from slightly perturbed best
            perturbed = set(best_result)
            # Remove a few random elements
            k = random.randint(2, 5)
            if k < len(perturbed):
                to_remove = random.sample(list(perturbed), k)
                perturbed -= set(to_remove)
            # Add back greedily
            candidates = [x for x in universe if x not in perturbed]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, perturbed):
                    perturbed.add(x)
            improved = local_search(list(perturbed), max_iterations=2000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved

    # Ensure nonempty
    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
