"""Rewritten program for the suite task; uses hypergraph-aware greedy + local search swaps."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def construct():
    from random import Random

    t0 = time.time()

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve(elements):
        n = len(elements)
        if n == 0:
            return []
        idx = {v: i for i, v in enumerate(elements)}

        # Build sum-triples (a, b, c) with a + b = c, all in set, a <= b.
        # For each element, track the triples it participates in.
        # conflicts[c] = list of (a, b) pairs such that a+b=c
        # conflicts_sum[a] = list of (b, c) such that a+b=c
        from collections import defaultdict
        conf_c = defaultdict(list)   # c -> list of (a,b)
        conf_a = defaultdict(list)   # a -> list of (b,c)
        for i in range(n):
            a = elements[i]
            for j in range(i, n):
                b = elements[j]
                c = a + b
                if c in idx:
                    k = idx[c]
                    conf_c[k].append((i, j))
                    conf_a[i].append((j, k))
                    if j != i:
                        conf_a[j].append((i, k))

        # Greedy: pick elements in order of fewest conflicts (low degree first),
        # only adding if it doesn't create a triple with already-chosen elements.
        # chosen: set of indices
        # To check if adding x is safe: for all (a,b) in conf_c[x], not both a,b in chosen.
        # Also for all (b,c) in conf_a[x], not both b,c in chosen.
        # Actually simpler: x is safe if no two chosen elements sum to x, and x + chosen != chosen.

        # Let's use a different representation: for a candidate x, it's unsafe if:
        #   exists y, z in chosen with y + z == x  (i.e., (y,z) in conf_c[x])
        #   exists y in chosen with x + y == z where z in chosen (i.e., (y,z) in conf_a[x] with y,z in chosen)
        #   exists y in chosen with y + x == z where z in chosen (same as above, covered)

        # Greedy order: sort by degree (total conflicts) ascending
        degree = [len(conf_c[i]) + len(conf_a[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degree[i])

        chosen = set()
        for i in order:
            # Check if adding i is safe
            safe = True
            # Check: any (a,b) in conf_c[i] with both a,b in chosen
            for (a, b) in conf_c[i]:
                if a in chosen and b in chosen:
                    safe = False
                    break
            if safe:
                # Check: any (b,c) in conf_a[i] with both b,c in chosen
                for (b, c) in conf_a[i]:
                    if b in chosen and c in chosen:
                        safe = False
                        break
            if safe:
                chosen.add(i)

        # Local search: 1-for-2 swaps
        # Try to remove one element r from chosen and add two elements p, q not in chosen,
        # such that the new set is still sum-free and |new| = |old| + 1.
        # More generally, try removing k elements and adding k+1 elements.
        # Start with simple 1-for-2.

        def is_sum_free(idx_set):
            """Check if idx_set is sum-free."""
            for i in idx_set:
                for (a, b) in conf_c[i]:
                    if a in idx_set and b in idx_set:
                        return False
            return True

        def can_add_to(chosen_set, candidate):
            """Check if candidate can be added to chosen_set without creating a triple."""
            # candidate is safe if no two elements in chosen_set sum to candidate,
            # and candidate + x != y for any x,y in chosen_set.
            for (a, b) in conf_c[candidate]:
                if a in chosen_set and b in chosen_set:
                    return False
            for (b, c) in conf_a[candidate]:
                if b in chosen_set and c in chosen_set:
                    return False
            return True

        # Local search with 1-for-2 swaps
        improved = True
        while improved:
            improved = False
            # For each element r in chosen, try removing it and adding two new elements
            for r in list(chosen):
                if time.time() - t0 > _DEADLINE - 5:
                    break
                new_chosen = chosen - {r}
                # Find candidates that can be added
                # We want to add 2 elements that are mutually compatible and compatible with new_chosen
                # Try pairs of non-chosen elements
                non_chosen = [i for i in range(n) if i not in new_chosen]
                # Sort non_chosen by degree (prefer low-degree)
                non_chosen.sort(key=lambda i: degree[i])
                found = False
                for pi in range(len(non_chosen)):
                    p = non_chosen[pi]
                    if not can_add_to(new_chosen, p):
                        continue
                    temp = new_chosen | {p}
                    for qj in range(pi + 1, len(non_chosen)):
                        q = non_chosen[qj]
                        if not can_add_to(temp, q):
                            continue
                        # Check q is compatible with p (already checked via can_add_to(temp, q))
                        chosen = temp | {q}
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if time.time() - t0 > _DEADLINE - 5:
                break

        return [elements[i] for i in chosen]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - t0 > _DEADLINE - 5:
            # Fallback: return odd numbers for remaining
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        results.append(solve(vals))

    return results
# EVOLVE-BLOCK-END
