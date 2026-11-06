"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

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
    universe_set = set(universe)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Precompute bad_pairs[x]: set of frozenset({a,b}) such that (a,b,x) is a bad triple
    bad_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    bad_pairs[c].add(frozenset((a, b)))
                    bad_pairs[b].add(frozenset((a, c)))
                    bad_pairs[a].add(frozenset((b, c)))

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for pair in bad_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return False
        return True

    # Precompute constraint degree
    constraint_degree = [0] * (n + 1)
    for x in range(1, n + 1):
        constraint_degree[x] = len(bad_pairs[x])

    # Greedy with a given order
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected

    # Improved local search with kick-and-refill
    def local_search(initial_set, max_iterations=10000):
        selected_set = set(initial_set)
        best_set = set(initial_set)
        best_size = len(selected_set)
        no_improve_count = 0

        all_elements = list(universe)

        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break

            # Adaptive kick size
            if no_improve_count < 5:
                kick_size = random.randint(1, 2)
            elif no_improve_count < 15:
                kick_size = random.randint(1, 4)
            elif no_improve_count < 40:
                kick_size = random.randint(2, 6)
            else:
                kick_size = random.randint(3, 10)

            if kick_size > len(selected_set):
                kick_size = max(1, len(selected_set) - 1)

            if kick_size == 0:
                continue

            removed = random.sample(list(selected_set), kick_size)
            remaining_set = selected_set - set(removed)

            # Candidates to add
            candidates = [x for x in all_elements if x not in remaining_set]
            random.shuffle(candidates)

            new_set = set(remaining_set)
            for x in candidates:
                if can_add(x, new_set):
                    new_set.add(x)

            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set
                selected_set = new_set
                no_improve_count = 0
            else:
                # Accept with some probability
                if len(new_set) >= len(selected_set) and random.random() < 0.25:
                    selected_set = new_set
                no_improve_count += 1

            # If stuck for too long, do a bigger restart
            if no_improve_count > 60:
                random.shuffle(all_elements)
                fresh = greedy(all_elements)
                if len(fresh) > best_size:
                    best_size = len(fresh)
                    best_set = set(fresh)
                    selected_set = set(fresh)
                    no_improve_count = 0
                else:
                    selected_set = set(best_set)
                    no_improve_count = 0

        return list(best_set)

    # Generate multiple orderings
    def make_order(seed, strategy):
        if strategy == 0:
            return list(universe)
        elif strategy == 1:
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            random.seed(seed)
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            return list(reversed(universe))
        elif strategy == 4:
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
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
            return sorted(universe, key=lambda x: (constraint_degree[x], x))
        elif strategy == 7:
            return sorted(universe, key=lambda x: (-constraint_degree[x], x))
        elif strategy == 8:
            random.seed(seed)
            return sorted(universe, key=lambda x: (constraint_degree[x], random.random()))
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
    for i in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 12345 + 67890, 2)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Constraint-degree randomized greedy passes
    for i in range(25):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 99999 + 1, 8)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Biased random greedy passes
    for i in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 99999 + 1, 5)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Local search on the best result
    if time.time() - start_time < DEADLINE - 3:
        improved = local_search(best_result, max_iterations=10000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    # Additional local search on other good results
    if time.time() - start_time < DEADLINE - 5:
        for attempt in range(15):
            if time.time() - start_time > DEADLINE - 3:
                break
            strat = 2 if attempt % 2 == 0 else 8
            order = make_order(100000 + attempt * 7777, strat)
            result = greedy(order)
            if len(result) >= best_size - 2:
                improved = local_search(result, max_iterations=5000)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_result = improved

    # Final local search polish
    if time.time() - start_time < DEADLINE - 1:
        improved = local_search(best_result, max_iterations=4000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
