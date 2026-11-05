"""Combined program: structural candidates + hypergraph-aware greedy + local search swaps."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

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

    def build_conflicts(elements):
        """Build conflict structures for fast sum-free checking."""
        n = len(elements)
        idx = {v: i for i, v in enumerate(elements)}
        conf_c = defaultdict(list)   # c -> list of (a_idx, b_idx)
        conf_a = defaultdict(list)   # a_idx -> list of (b_idx, c_idx)
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
        return idx, conf_c, conf_a

    def can_add(conf_c, conf_a, chosen_set, candidate):
        """Check if candidate can be added to chosen_set without creating a triple."""
        for (a, b) in conf_c[candidate]:
            if a in chosen_set and b in chosen_set:
                return False
        for (b, c) in conf_a[candidate]:
            if b in chosen_set and c in chosen_set:
                return False
        return True

    def greedy_from_order(elements, order, conf_c, conf_a):
        """Greedy: add elements in given order if safe."""
        chosen = set()
        for i in order:
            if can_add(conf_c, conf_a, chosen, i):
                chosen.add(i)
        return chosen

    def local_search_1_for_2(elements, chosen, conf_c, conf_a, t0, deadline):
        """Local search: try removing one element and adding two others."""
        n = len(elements)
        improved = True
        while improved:
            improved = False
            for r in list(chosen):
                if time.time() - t0 > deadline:
                    return chosen
                new_chosen = chosen - {r}
                # Find candidates that can be added to new_chosen
                candidates = []
                for i in range(n):
                    if i not in new_chosen and can_add(conf_c, conf_a, new_chosen, i):
                        candidates.append(i)
                # Try pairs from candidates
                for pi in range(len(candidates)):
                    p = candidates[pi]
                    temp = new_chosen | {p}
                    for qj in range(pi + 1, len(candidates)):
                        q = candidates[qj]
                        if can_add(conf_c, conf_a, temp, q):
                            chosen = temp | {q}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return chosen

    def solve(elements):
        n = len(elements)
        if n == 0:
            return []
        idx, conf_c, conf_a = build_conflicts(elements)
        
        # Generate structural candidates
        candidates = []
        
        # Odd numbers
        odd_set = set(i for i in range(n) if elements[i] % 2 != 0)
        candidates.append(odd_set)
        
        # Upper half (x > M/2)
        M = elements[-1]
        m = elements[0]
        upper = set(i for i in range(n) if elements[i] > M / 2.0)
        candidates.append(upper)
        
        # Lower half (x < m/2)
        lower = set(i for i in range(n) if elements[i] < m / 2.0)
        candidates.append(lower)
        
        # Positive numbers
        pos = set(i for i in range(n) if elements[i] > 0)
        candidates.append(pos)
        
        # Negative numbers
        neg = set(i for i in range(n) if elements[i] < 0)
        candidates.append(neg)
        
        # Large positive (x > M/3)
        large_pos = set(i for i in range(n) if elements[i] > 0 and elements[i] > M / 3.0)
        candidates.append(large_pos)
        
        # Large negative (x < m/3)
        large_neg = set(i for i in range(n) if elements[i] < 0 and elements[i] < m / 3.0)
        candidates.append(large_neg)
        
        # Absolute value thresholds
        abs_vals = sorted(set(abs(x) for x in elements))
        for K in abs_vals[:10]:
            S_K = set(i for i in range(n) if abs(elements[i]) > K)
            candidates.append(S_K)
        
        # Greedy from multiple orderings
        degree = [len(conf_c[i]) + len(conf_a[i]) for i in range(n)]
        
        # Low degree first
        order_low_deg = sorted(range(n), key=lambda i: degree[i])
        candidates.append(greedy_from_order(elements, order_low_deg, conf_c, conf_a))
        
        # High abs first
        order_abs_desc = sorted(range(n), key=lambda i: abs(elements[i]), reverse=True)
        candidates.append(greedy_from_order(elements, order_abs_desc, conf_c, conf_a))
        
        # Sorted ascending
        order_asc = list(range(n))
        candidates.append(greedy_from_order(elements, order_asc, conf_c, conf_a))
        
        # Sorted descending
        order_desc = list(range(n - 1, -1, -1))
        candidates.append(greedy_from_order(elements, order_desc, conf_c, conf_a))
        
        # Random shuffle
        rng = Random(42)
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        candidates.append(greedy_from_order(elements, order_rand, conf_c, conf_a))
        
        # Validate and pick best candidate
        best_chosen = set()
        best_len = 0
        for cand in candidates:
            if len(cand) <= best_len:
                continue
            # Verify sum-free
            valid = True
            for i in cand:
                for (a, b) in conf_c[i]:
                    if a in cand and b in cand:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_chosen = cand
                best_len = len(cand)
        
        # If no valid candidate found (shouldn't happen), use odd numbers as fallback
        if not best_chosen:
            best_chosen = set(i for i in range(n) if elements[i] % 2 != 0)
        
        # Local search improvement
        if time.time() - t0 < _DEADLINE - 10:
            best_chosen = local_search_1_for_2(elements, best_chosen, conf_c, conf_a, t0, _DEADLINE - 5)
        
        return [elements[i] for i in best_chosen]

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
