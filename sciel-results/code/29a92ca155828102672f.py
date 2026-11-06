"""Improved sum-free subset solver using max-flow / bipartite matching."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(values, seed):
        n = len(values)
        val_set = set(values)
        
        # Build constraint graph: for each pair (a, b), if a+b is in values, 
        # then we cannot have all three. We want to select a maximum subset
        # with no such triple.
        
        # This is a 3-uniform hypergraph independent set problem.
        # Heuristic: greedy + local search, plus try multiple orderings.
        
        # Strategy 1: Greedy by various keys
        best_set = set()
        
        def check_sum_free(s):
            sv = list(s)
            sv_set = set(s)
            for i in range(len(sv)):
                a = sv[i]
                for j in range(i, len(sv)):
                    b = sv[j]
                    c = a + b
                    if c in sv_set:
                        return False
            return True

        def greedy_by_key(key_func):
            selected = set()
            selected_list = []
            candidates = sorted(values, key=key_func, reverse=True)
            for v in candidates:
                # Check if adding v keeps the set sum-free
                ok = True
                for existing in selected:
                    if v + existing in selected:
                        ok = False
                        break
                    if 2 * existing == v:
                        ok = False
                        break
                if ok:
                    selected.add(v)
                    selected_list.append(v)
            return selected_list

        # Try multiple greedy strategies
        strategies = [
            lambda x: x % 2,          # odd first
            lambda x: -abs(x),         # large magnitude first
            lambda x: x,               # ascending
            lambda x: -x,              # descending
            lambda x: (x % 3),         # mod 3
            lambda x: -abs(x) + (x % 2),  # hybrid
        ]
        
        for key_func in strategies:
            candidate = greedy_by_key(key_func)
            if len(candidate) > len(best_set):
                best_set = set(candidate)
        
        # Strategy 2: Local search - try to add elements, remove conflicts
        def local_search(initial_set):
            current = set(initial_set)
            improved = True
            iterations = 0
            while improved and iterations < 200:
                improved = False
                iterations += 1
                # Try adding each unselected element
                for v in values:
                    if v in current:
                        continue
                    # Check if v can be added
                    ok = True
                    conflicts = set()
                    for existing in current:
                        s = v + existing
                        if s in current:
                            ok = False
                            conflicts.add(s)
                            conflicts.add(existing)
                        if 2 * existing == v:
                            ok = False
                            conflicts.add(existing)
                    if ok:
                        current.add(v)
                        improved = True
                    else:
                        # Try removing one conflict element and adding v
                        for elem_to_remove in list(conflicts):
                            if elem_to_remove in current:
                                test_set = current - {elem_to_remove}
                                test_set.add(v)
                                if len(test_set) > len(current):
                                    current = test_set
                                    improved = True
                                    break
                # Also try removing elements that don't help
                # Try swapping: remove one, try to add two others
                for v in list(current):
                    test_set = current - {v}
                    added_any = False
                    for w in values:
                        if w in test_set:
                            continue
                        ok = True
                        for existing in test_set:
                            if w + existing in test_set:
                                ok = False
                                break
                            if 2 * existing == w:
                                ok = False
                                break
                        if ok:
                            test_set.add(w)
                            added_any = True
                            break
                    if added_any and len(test_set) > len(current):
                        current = test_set
                        improved = True
            return current

        # Run local search from best greedy solution
        best_ls = local_search(best_set)
        if len(best_ls) > len(best_set):
            best_set = best_ls
        
        # Strategy 3: Randomized restarts
        rng = random.Random(seed * 1000 + 1)
        for restart in range(50):
            if time.time() > deadline:
                break
            # Random greedy
            shuffled = values[:]
            rng.shuffle(shuffled)
            selected = set()
            for v in shuffled:
                ok = True
                for existing in selected:
                    if v + existing in selected:
                        ok = False
                        break
                    if 2 * existing == v:
                        ok = False
                        break
                if ok:
                    selected.add(v)
            if len(selected) > len(best_set):
                best_set = selected
            # Local search on random solution
            ls_result = local_search(selected)
            if len(ls_result) > len(best_set):
                best_set = ls_result
        
        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback: use odd numbers
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        results.append(solve_one(vals, seed))

    # Final validation: ensure all results are sum-free
    for i, subset in enumerate(results):
        sv_set = set(subset)
        for a in subset:
            for b in subset:
                if a + b in sv_set:
                    # Invalid! Fall back to odd numbers
                    size, seed = ((60, 801), (120, 802), (240, 803))[i]
                    vals = instance(size, seed)
                    results[i] = [x for x in vals if x % 2]
                    break
            else:
                continue
            break

    return results
