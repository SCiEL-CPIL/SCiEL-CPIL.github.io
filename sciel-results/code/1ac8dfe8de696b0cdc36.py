import time
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_sum_free(A, rng, time_budget=3.0):
        n = len(A)
        if n == 0:
            return []

        val_to_idx = {v: i for i, v in enumerate(A)}

        # Build conflict structure: for each pair (i,j), does A[i]+A[j] exist?
        # conflict[i][j] = k if A[i]+A[j] == A[k], else -1
        # We need: no a+b=c with a,b,c all in set (a=b allowed)
        # So conflict arises when we have i, j, and k=A[i]+A[j] all present.

        # For efficient delta computation, precompute for each element which triples it's in
        # triple: (i, j, k) where A[i]+A[j]=A[k], i<=j
        triples = []
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    triples.append((i, j, k))

        # For each element, list of triple indices it participates in
        elem_triples = [[] for _ in range(n)]
        for t_idx, (i, j, k) in enumerate(triples):
            elem_triples[i].append(t_idx)
            if i != j:
                elem_triples[j].append(t_idx)
            elem_triples[k].append(t_idx)

        # Greedy construction: sort by "conflict degree" (number of triples involving element)
        # and greedily add elements that don't create conflicts
        def greedy_construct():
            # Sort elements by degree (fewer conflicts first)
            degrees = [len(elem_triples[i]) for i in range(n)]
            order = sorted(range(n), key=lambda x: (degrees[x], rng.random()))

            in_set = [False] * n
            # Track count of elements from each triple that are in the set
            triple_count = [0] * len(triples)

            for idx in order:
                # Check if adding idx creates any violation
                can_add = True
                for t_idx in elem_triples[idx]:
                    a, b, c = triples[t_idx]
                    # Count how many of a,b,c are already in set (before adding idx)
                    cnt = (1 if in_set[a] else 0) + (1 if in_set[b] else 0) + (1 if in_set[c] else 0)
                    # idx is one of a,b,c
                    # After adding idx, count becomes cnt + 1
                    if cnt + 1 == 3:
                        can_add = False
                        break
                if can_add:
                    in_set[idx] = True
                    for t_idx in elem_triples[idx]:
                        triple_count[t_idx] += 1

            return in_set

        # Local search with flip + swap moves
        def local_search(in_set, T_init=5.0, cooling=0.9995, min_T=0.005, max_time=None):
            current_size = sum(in_set)
            # Compute penalty
            penalty = 0
            for t_idx, (a, b, c) in enumerate(triples):
                if in_set[a] and in_set[b] and in_set[c]:
                    penalty += 1

            current_energy = -current_size + penalty
            best_in_set = in_set[:]
            best_size = current_size if penalty == 0 else 0
            best_energy = current_energy if penalty == 0 else float('inf')

            T = T_init
            start = time.time()
            step = 0

            while (time.time() - start) < max_time:
                step += 1
                # Choose move type: flip (60%) or swap (40%)
                if rng.random() < 0.6:
                    # Flip move
                    i = rng.randint(0, n - 1)
                    new_state = not in_set[i]

                    delta_penalty = 0
                    for t_idx in elem_triples[i]:
                        a, b, c = triples[t_idx]
                        cnt = (1 if in_set[a] else 0) + (1 if in_set[b] else 0) + (1 if in_set[c] else 0)
                        # i is in this triple, so cnt includes in_set[i]
                        # Actually cnt already includes in_set[i] since i is one of a,b,c
                        # If we flip i:
                        if new_state:
                            # i was not in set, now is
                            new_cnt = cnt + 1
                        else:
                            # i was in set, now not
                            new_cnt = cnt - 1

                        old_viol = 1 if cnt == 3 else 0
                        new_viol = 1 if new_cnt == 3 else 0
                        delta_penalty += (new_viol - old_viol)

                    new_size = current_size + (1 if new_state else -1)
                    new_penalty = penalty + delta_penalty
                    new_energy = -new_size + new_penalty
                    delta_E = new_energy - current_energy

                    if delta_E <= 0 or rng.random() < math.exp(-delta_E / T):
                        in_set[i] = new_state
                        penalty = new_penalty
                        current_size = new_size
                        current_energy = new_energy

                        if penalty == 0 and current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set[:]
                            best_energy = current_energy

                else:
                    # Swap move: remove one from set, add one not in set
                    in_indices = [i for i in range(n) if in_set[i]]
                    out_indices = [i for i in range(n) if not in_set[i]]

                    if len(in_indices) == 0 or len(out_indices) == 0:
                        continue

                    i_out = in_indices[rng.randint(0, len(in_indices) - 1)]
                    i_in = out_indices[rng.randint(0, len(out_indices) - 1)]

                    # Compute delta for removing i_out and adding i_in
                    delta_penalty = 0
                    # First remove i_out
                    for t_idx in elem_triples[i_out]:
                        a, b, c = triples[t_idx]
                        cnt = (1 if in_set[a] else 0) + (1 if in_set[b] else 0) + (1 if in_set[c] else 0)
                        # i_out is in set, so cnt includes 1 for i_out
                        # After removal: cnt - 1
                        old_viol = 1 if cnt == 3 else 0
                        new_viol = 1 if (cnt - 1) == 3 else 0
                        delta_penalty += (new_viol - old_viol)

                    # Now add i_in (conceptually i_out is removed)
                    for t_idx in elem_triples[i_in]:
                        a, b, c = triples[t_idx]
                        cnt = (1 if in_set[a] else 0) + (1 if in_set[b] else 0) + (1 if in_set[c] else 0)
                        # i_in is NOT in set, i_out IS in set
                        # After: i_out removed, i_in added
                        # If i_out and i_in are in the same triple, we need to be careful
                        # cnt currently includes i_out if i_out is in {a,b,c}
                        # Let's recompute: after swap, in_set has i_out=False, i_in=True
                        # For this triple, the new count = cnt - (1 if i_out in {a,b,c}) + (1 if i_in in {a,b,c})
                        # But i_in is in {a,b,c} by definition (it's in elem_triples[i_in])
                        # i_out may or may not be in {a,b,c}
                        has_i_out = 1 if (a == i_out or b == i_out or c == i_out) else 0
                        # i_in is definitely in {a,b,c}
                        new_cnt = cnt - has_i_out + 1
                        old_viol = 1 if cnt == 3 else 0
                        new_viol = 1 if new_cnt == 3 else 0
                        delta_penalty += (new_viol - old_viol)

                    # Size doesn't change in a swap
                    new_size = current_size
                    new_penalty = penalty + delta_penalty
                    new_energy = -new_size + new_penalty
                    delta_E = new_energy - current_energy

                    if delta_E <= 0 or rng.random() < math.exp(-delta_E / T):
                        in_set[i_out] = False
                        in_set[i_in] = True
                        penalty = new_penalty
                        # current_size unchanged
                        current_energy = new_energy

                        if penalty == 0 and current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set[:]
                            best_energy = current_energy

                T *= cooling
                if T < min_T:
                    T = min_T

            return best_in_set, best_size

        # Multi-restart with perturbation
        best_overall = []
        best_overall_size = 0

        # First, try greedy construction
        in_set = greedy_construct()
        result, sz = local_search(in_set, T_init=8.0, cooling=0.9995, min_T=0.005, max_time=min(2.0, time_budget * 0.4))
        if sz > best_overall_size:
            best_overall_size = sz
            best_overall = [A[i] for i in range(n) if result[i]]

        # Try parity-based initialization
        in_set2 = [False] * n
        for i in range(n):
            if A[i] % 2 != 0:
                in_set2[i] = True
        result2, sz2 = local_search(in_set2, T_init=5.0, cooling=0.9995, min_T=0.005, max_time=min(1.5, time_budget * 0.3))
        if sz2 > best_overall_size:
            best_overall_size = sz2
            best_overall = [A[i] for i in range(n) if result2[i]]

        # Perturbation restarts
        remaining_time = time_budget - (time.time() - (time.time() - time_budget))  # rough
        # Let's just do a few more restarts if time permits
        start_all = time.time()
        for restart in range(5):
            if time.time() - start_all > time_budget * 0.5:
                break

            # Perturb the best solution
            in_set3 = best_in_set.copy() if best_in_set else greedy_construct()
            # Randomly flip 5-15% of elements
            num_flips = max(1, int(n * rng.uniform(0.05, 0.15)))
            for _ in range(num_flips):
                idx = rng.randint(0, n - 1)
                in_set3[idx] = not in_set3[idx]

            result3, sz3 = local_search(in_set3, T_init=3.0, cooling=0.9997, min_T=0.003, max_time=min(1.0, time_budget * 0.15))
            if sz3 > best_overall_size:
                best_overall_size = sz3
                best_overall = [A[i] for i in range(n) if result3[i]]

        # Final validation
        if best_overall:
            s_set = set(best_overall)
            valid = True
            for a in best_overall:
                for b in best_overall:
                    if a + b in s_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                # Repair
                current = set(best_overall)
                changed = True
                while changed:
                    changed = False
                    for a in list(current):
                        for b in current:
                            if a + b in current:
                                current.discard(a + b)
                                changed = True
                                break
                        if changed:
                            break
                best_overall = list(current)

        return best_overall

    results = []
    overall_start = time.time()
    total_budget = 110.0  # Leave 10s safety margin for 120s limit

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        A = instance(size, seed)
        # Allocate time proportional to problem size
        if size <= 60:
            budget = 20.0
        elif size <= 120:
            budget = 35.0
        else:
            budget = 55.0

        remaining = total_budget - (time.time() - overall_start)
        budget = min(budget, max(5.0, remaining))

        rng = random.Random(seed + 42)
        subset = solve_sum_free(A, rng, time_budget=budget)
        results.append(subset)

    return results
