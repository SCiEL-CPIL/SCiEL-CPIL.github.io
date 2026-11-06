"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))

    # Precompute all "bad" pairs: pairs (a, b) that form an equal-lcm triple with some c
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # For greedy construction, we need: for candidate x and existing pair (a,b),
    # check if lcm(a,b) == lcm(a,x) == lcm(b,x)
    #
    # Better approach: precompute for each x, the set of pairs (a,b) that would form
    # a bad triple with x. But that's O(n^3) which is ~2M, manageable.
    #
    # Actually, let's think differently. For a fixed x, which pairs (a,b) satisfy
    # lcm(a,b) == lcm(a,x) == lcm(b,x)?
    # This means lcm(a,b) = L where L = lcm(a,x) = lcm(b,x).
    # So a | L, b | L, x | L, and lcm(a,b)=L, lcm(a,x)=L, lcm(b,x)=L.
    #
    # For the greedy approach with precomputation:
    # Precompute a "conflict" structure. For each element x, store which other elements
    # conflict with it in a pairwise sense that matters for triples.
    #
    # Actually, the cleanest precomputation: for each pair (a,b) with a<b, compute L=lcm(a,b).
    # Then for a new element x, it conflicts with pair (a,b) iff L == lcm(a,x) == lcm(b,x).
    #
    # Let's precompute: for each x, a set of "forbidden pairs" is too large.
    # Instead, for each x, precompute the set of elements y such that there exists z
    # where (x,y,z) is a bad triple. But we need the pair structure.
    #
    # Most efficient for greedy: maintain the selected set. When considering x,
    # check all pairs (a,b) in selected: is lcm(a,b)==lcm(a,x)==lcm(b,x)?
    # With precomputed lcm table, each check is O(1).
    #
    # Precompute lcm table for all pairs in [1,128]
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    def is_bad_triple(a, b, x):
        """Check if (a, b, x) forms an equal-lcm triple."""
        lab = get_lcm(a, b)
        lax = get_lcm(a, x)
        lbx = get_lcm(b, x)
        return lab == lax == lbx

    # Strategy: multiple randomized greedy passes + local search
    def greedy_randomized(order_seed=None):
        if order_seed is not None:
            random.seed(order_seed)
        else:
            random.seed(random.randint(0, 2**31 - 1))
        candidates = list(universe)
        random.shuffle(candidates)
        selected = []
        selected_set = set()
        for x in candidates:
            can_add = True
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    if is_bad_triple(a, b, x):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected.append(x)
                selected_set.add(x)
        return selected

    def local_search(initial_set, max_iterations=5000):
        """Try to improve by removing one element and adding another."""
        selected = list(initial_set)
        selected_set = set(selected)
        best = len(selected)

        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break

            improved = False
            # Try removing each element and adding a better one
            for remove_idx in range(len(selected)):
                rem = selected[remove_idx]
                remaining = selected[:remove_idx] + selected[remove_idx+1:]
                remaining_set = selected_set - {rem}

                # Find best element to add
                best_add = None
                best_score = 0
                for x in universe:
                    if x in remaining_set:
                        continue
                    # Check if x can be added to remaining
                    can_add = True
                    for i in range(len(remaining)):
                        a = remaining[i]
                        for j in range(i + 1, len(remaining)):
                            b = remaining[j]
                            if is_bad_triple(a, b, x):
                                can_add = False
                                break
                        if not can_add:
                            break
                    if can_add:
                        # Score: prefer elements that "open up" more possibilities
                        # Simple heuristic: just count how many elements could then be added
                        # For simplicity, just pick the first valid one or use a heuristic
                        best_add = x
                        break

                if best_add is not None:
                    new_selected = remaining + [best_add]
                    new_selected_set = remaining_set | {best_add}
                    if len(new_selected) >= len(selected):
                        selected = new_selected
                        selected_set = new_selected_set
                        best = len(selected)
                        improved = True
                        break

            if not improved:
                break

        return selected

    # Run multiple greedy passes with different random seeds
    best_result = []
    num_greedy_passes = 20

    for i in range(num_greedy_passes):
        if time.time() - start_time > DEADLINE:
            break
        result = greedy_randomized(order_seed=i * 12345 + 67890)
        if len(result) > len(best_result):
            best_result = result

    # Also try a deterministic greedy (sorted order) as baseline
    selected = []
    for x in universe:
        can_add = True
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if is_bad_triple(a, b, x):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            selected.append(x)
    if len(selected) > len(best_result):
        best_result = selected

    # Try local search on the best result
    if time.time() - start_time < DEADLINE - 5:
        improved = local_search(best_result, max_iterations=2000)
        if len(improved) > len(best_result):
            best_result = improved

    # Additional randomized local search passes
    if time.time() - start_time < DEADLINE - 10:
        for attempt in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            result = greedy_randomized(order_seed=100000 + attempt * 7777)
            improved = local_search(result, max_iterations=1000)
            if len(improved) > len(best_result):
                best_result = improved

    # Ensure nonempty
    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
