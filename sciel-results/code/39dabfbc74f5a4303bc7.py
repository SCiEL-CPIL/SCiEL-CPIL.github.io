import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin before 60s limit

    universe = list(range(1, 23))
    n = len(universe)

    # Precompute LCM of 1..22 and integer weights
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, universe)
    weights = {i: L // i for i in universe}

    # ---------- Build conflict graph for MIS ----------
    # A set is dissociated iff no subset has a nontrivial signed sum = 0.
    # Enumerate all "bad" subsets (size <= 4) and forbid them.
    # Two elements conflict if they appear together in a bad subset.
    from itertools import combinations

    def is_bad(subset):
        """Return True if subset has a nontrivial signed sum = 0."""
        k = len(subset)
        # signed sums: for each element, coeff in {-1, 0, +1}, not all zero
        # Use meet-in-the-middle for k up to 4: split into two halves
        half = k // 2
        left = subset[:half]
        right = subset[half:]

        # All signed sums for a list
        def signed_sums(arr):
            res = [0]
            for w in arr:
                res = [x + w for x in res] + [x - w for x in res] + [x for x in res]
            return res

        ls = set(signed_sums(left))
        for rs in signed_sums(right):
            # total = l + r == 0  =>  l == -r
            # but we need not-all-zero. If l=0 and r=0, that's all-zero (skip).
            if (-rs) in ls:
                # Check if this combination is nontrivial:
                # If rs==0 and the matching l==0, could be all-zero.
                # But signed_sums includes the all-zero combination.
                # We need at least one element with nonzero coefficient.
                # If rs != 0, then right side is nontrivial -> bad.
                # If rs == 0, we need left side to be nontrivial (l=0 with non-all-zero).
                if rs != 0:
                    return True
                else:
                    # rs == 0, need a nontrivial left signed sum = 0
                    # Check if left has a nontrivial signed sum = 0
                    # signed_sums(left) contains 0 from all-zero; need another way.
                    # Simpler: brute force for small k
                    pass
        # Fallback: brute force for k <= 4 (3^4 = 81 combos)
        if k <= 4:
            import itertools as _it
            for coeffs in _it.product([-1, 0, 1], repeat=k):
                if all(c == 0 for c in coeffs):
                    continue
                s = sum(c * weights[x] for c, x in zip(coeffs, subset))
                if s == 0:
                    return True
        return False

    # Enumerate bad subsets of size 2, 3, 4
    bad_pairs = set()
    bad_triples = set()
    bad_quads = set()

    for a, b in combinations(universe, 2):
        if is_bad([a, b]):
            bad_pairs.add((a, b))

    for a, b, c in combinations(universe, 3):
        if is_bad([a, b, c]):
            bad_triples.add((a, b, c))

    for a, b, c, d in combinations(universe, 4):
        if is_bad([a, b, c, d]):
            bad_quads.add((a, b, c, d))

    # Build conflict graph: edge (i,j) if {i,j} is in a bad subset
    # (i.e., they can't both be in a dissociated set)
    # Actually, the constraint is stronger: any bad subset of ANY size is forbidden.
    # For MIS, we need pairwise conflicts. But a bad triple {a,b,c} means not all three
    # can be together, even if no pair is bad. This is a hypergraph, not a graph.
    #
    # Simplification: For the MIS approach to work directly, we need pairwise conflicts.
    # A pair {a,b} is "conflicting" if there exists some c such that {a,b,c} is bad,
    # OR {a,b} itself is bad. But that's not quite right either.
    #
    # Better approach: Use the greedy + verification approach but with the integer
    # arithmetic for speed, and add a local search / branch-and-bound for expansion.
    #
    # Let me reconsider. The current program's greedy approach works well. The MIS
    # idea is to find a maximum set with no bad subsets. Let me use a different
    # strategy: backtracking search with pruning, using the integer weights.

    # ---------- Backtracking search for maximum dissociated set ----------
    # Order elements by weight descending (smaller reciprocals first) for better pruning
    sorted_universe = sorted(universe, key=lambda x: -weights[x])
    # Map to indices for bit operations
    elem_to_idx = {x: i for i, x in enumerate(sorted_universe)}
    sorted_weights = [weights[x] for x in sorted_universe]

    best_set = []
    best_size = 0

    def check_dissociated(subset_indices):
        """Check if the subset (by sorted indices) is dissociated using integer weights."""
        sums = {0}
        for idx in subset_indices:
            w = sorted_weights[idx]
            new_sums = []
            for s in sums:
                ns = s + w
                if ns in sums:
                    return False
                new_sums.append(ns)
            sums.update(new_sums)
        return True

    # Greedy first to get a good initial bound
    def greedy(order):
        current_set = []
        sums = {0}
        for x in order:
            w = weights[x]
            collision = False
            new_sums = []
            for s in sums:
                ns = s + w
                if ns in sums:
                    collision = True
                    break
                new_sums.append(ns)
            if not collision:
                current_set.append(x)
                sums.update(new_sums)
        return current_set

    # Try several greedy orders
    rng = random.Random(42)
    orders = [
        sorted_universe,
        sorted(universe),
        sorted(universe, reverse=True),
    ]
    for _ in range(30):
        p = universe[:]
        rng.shuffle(p)
        orders.append(p)

    for order in orders:
        if time.time() - start_time > timeout:
            break
        gs = greedy(order)
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs[:]
            # Try to expand
            remaining = [x for x in universe if x not in gs]
            sums = {0}
            for x in gs:
                w = weights[x]
                sums.update([s + w for s in list(sums)])
            added = True
            while added:
                added = False
                for x in remaining[:]:
                    w = weights[x]
                    collision = False
                    new_sums = []
                    for s in sums:
                        ns = s + w
                        if ns in sums:
                            collision = True
                            break
                        new_sums.append(ns)
                    if not collision:
                        gs.append(x)
                        remaining.remove(x)
                        sums.update(new_sums)
                        added = True
                        best_size = len(gs)
                        if time.time() - start_time > timeout:
                            return gs

    # ---------- Branch and bound for improvement ----------
    # Use the sorted order. At each level, try including or excluding the next element.
    # Prune if even adding all remaining can't beat best_size.

    def backtrack(pos, current, sums, current_size):
        nonlocal best_set, best_size

        if time.time() - start_time > timeout:
            return

        # Prune: can't beat best
        if current_size + (n - pos) <= best_size:
            return

        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = [sorted_universe[i] for i in current]

        if pos == n:
            return

        # Option 1: Include sorted_universe[pos]
        w = sorted_weights[pos]
        collision = False
        new_sums = []
        for s in sums:
            ns = s + w
            if ns in sums:
                collision = True
                break
            new_sums.append(ns)

        if not collision:
            current.append(pos)
            new_sums_set = sums | set(new_sums)
            backtrack(pos + 1, current, new_sums_set, current_size + 1)
            current.pop()

        # Option 2: Exclude sorted_universe[pos]
        backtrack(pos + 1, current, sums, current_size)

    # Start backtracking from the greedy solution as initial
    # Convert best_set to indices in sorted order
    initial_indices = [elem_to_idx[x] for x in best_set]
    initial_sums = {0}
    for idx in initial_indices:
        w = sorted_weights[idx]
        initial_sums.update([s + w for s in list(initial_sums)])

    # Only backtrack on elements not in the initial set, trying to add more
    # This is complex; instead, do a full backtrack but with a good initial bound
    # The initial bound from greedy should prune a lot.

    # To keep it fast, limit the depth of backtrack or use it selectively
    # Given n=22, full backtrack is too slow. Use it with strong pruning.
    # Start from empty and let the initial best_size prune.

    backtrack(0, [], {0}, 0)

    return best_set
