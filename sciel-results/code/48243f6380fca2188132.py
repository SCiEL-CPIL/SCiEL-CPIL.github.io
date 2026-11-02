import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix
    M = candidates @ candidates.T

    # Compatibility: M[i,j] <= 2 for i != j
    compat = (M <= 2)
    np.fill_diagonal(compat, True)

    best_set = None
    best_count = 0

    rng = np.random.default_rng(42)

    # Strategy A: Multiple random orderings
    for trial in range(200):
        if time.time() > deadline - 40:
            break

        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            if np.all(compat[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1

        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy B: Least conflicts first
    if time.time() < deadline - 35:
        conflict_counts = np.sum(~compat, axis=1)
        order_b = np.argsort(conflict_counts, kind='stable')

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order_b:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            if np.all(compat[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1

        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy C: Most compatible first
    if time.time() < deadline - 30:
        compat_counts = np.sum(compat, axis=1)
        order_c = np.argsort(-compat_counts, kind='stable')

        selected_mask = np.zeros(N, dtype=bool)
        count = 0

        for idx in order_c:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            if np.all(compat[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1

        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy D: Local search - try swaps
    if best_set is not None and time.time() < deadline - 15:
        selected_list = list(np.where(best_set)[0])
        unselected_list = list(np.where(~best_set)[0])

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False

            # Try to find a pair swap: remove one, add two
            for rem_idx in selected_list:
                if time.time() > deadline - 10:
                    break

                # Find candidates that are compatible with all selected except rem_idx
                other_selected = [s for s in selected_list if s != rem_idx]
                if not other_selected:
                    continue

                # Find unselected vertices compatible with other_selected
                compatible_adds = [u for u in unselected_list if np.all(compat[u, other_selected])]

                if len(compatible_adds) >= 2:
                    # Try adding two
                    add1 = compatible_adds[0]
                    add2 = compatible_adds[1]

                    # Check if add1 and add2 are compatible with each other
                    if compat[add1, add2]:
                        # Do the swap
                        best_set[rem_idx] = False
                        best_set[add1] = True
                        best_set[add2] = True
                        selected_list.remove(rem_idx)
                        selected_list.append(add1)
                        selected_list.append(add2)
                        unselected_list.remove(add1)
                        unselected_list.remove(add2)
                        unselected_list.append(rem_idx)
                        best_count += 1
                        improved = True
                        break

    # Strategy E: Simulated annealing
    if best_set is not None and time.time() < deadline - 5:
        current = best_set.copy()
        current_count = np.sum(current)
        best_sa = current.copy()
        best_sa_count = current_count

        sel = list(np.where(current)[0])
        unsel = list(np.where(~current)[0])

        temp = 2.0
        cool = 0.997

        for it in range(10000):
            if time.time() > deadline - 2:
                break

            r = rng.random()
            if r < 0.3 and len(unsel) > 0:
                # Try add
                idx = unsel[rng.integers(len(unsel))]
                if np.all(compat[idx, sel]):
                    current[idx] = True
                    sel.append(idx)
                    unsel.remove(idx)
                    current_count += 1
                    if current_count > best_sa_count:
                        best_sa = current.copy()
                        best_sa_count = current_count
                else:
                    # Try swap: remove random sel, add idx
                    if sel:
                        rem = sel[rng.integers(len(sel))]
                        others = [s for s in sel if s != rem]
                        if others and np.all(compat[idx, others]):
                            current[rem] = False
                            current[idx] = True
                            sel.remove(rem)
                            sel.append(idx)
                            unsel.remove(idx)
                            unsel.append(rem)
                            # Accept with prob 1 (delta=0)
            elif r < 0.6 and len(sel) > 2:
                # Remove one
                rem = sel[rng.integers(len(sel))]
                current[rem] = False
                sel.remove(rem)
                unsel.append(rem)
                current_count -= 1
                if rng.random() > np.exp(-1.0 / temp):
                    # Revert
                    current[rem] = True
                    sel.append(rem)
                    unsel.remove(rem)
                    current_count += 1

            temp *= cool

        if best_sa_count > best_count:
            best_count = best_sa_count
            best_set = best_sa

    # Final: add any remaining compatible
    if best_set is not None and time.time() < deadline - 1:
        unsel = np.where(~best_set)[0]
        for idx in unsel:
            if time.time() > deadline - 0.5:
                break
            if np.all(compat[idx, best_set]):
                best_set[idx] = True
                best_count += 1

    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set].astype(np.int64)
