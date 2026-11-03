import time
import random
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for each word
    # mask[i] is a bitmask of all words within Hamming distance <= radius of word i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        # radius 1: self + neighbors
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)

        # 1-for-1 swap and removal
        for _ in range(100):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    # Can remove
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Find replacement
                best_gain = -1
                best_word = -1
                # Optimization: Only check words that cover the unique words
                # We can iterate all, but break if we find a good one?
                # Actually, we want the one that minimizes "loss" or maximizes "gain" elsewhere.
                # Gain = (masks[w2] & ~m).bit_count()
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    if (masks[w2] & unique_mask) == unique_mask:
                        gain = (masks[w2] & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Replace
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit

                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                for i_idx in range(min(50, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]

                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1 or idx2 >= len(selected):
                        continue
                    w2 = selected[idx2]

                    m1 = masks[w1]
                    m2 = masks[w2]

                    u_ab = 0
                    inter = m1 & m2
                    mm = inter
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff1 = m1 & ~m2
                    mm = diff1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff2 = m2 & ~m1
                    mm = diff2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
                        # Remove both
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break

                    if u_ab.bit_count() > 10:
                        continue

                    # Find replacement w3
                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        if (masks[w3] & u_ab) == u_ab:
                            best_word = w3
                            break

                    if best_word >= 0:
                        # Replace w1, w2 with w3
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit

                        mm = masks[best_word]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit

                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected.append(best_word)

                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(best_word)

                        improved = True
                        break

                if improved:
                    selected = polish(selected, time_limit)
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        mm = masks[w]
                        while mm:
                            lowbit = mm & -mm
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            mm ^= lowbit
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    def solve_lp():
        """
        Solve LP relaxation of Set Cover.
        Minimize sum(x_i)
        Subject to sum_{j in S_i} x_j >= 1 for all i
        0 <= x_i <= 1
        """
        # A_ub * x <= b_ub is not suitable for >= constraints directly in linprog standard form (min c^T x s.t. A_ub x <= b_ub)
        # We can negate: -sum x_j <= -1
        # A is (total_space, total_space)
        # A[i, j] = 1 if i is in S_j (word i is covered by codeword j)
        # Note: masks[j] contains bits for words covered by j.
        # So A[i, j] = 1 if (masks[j] >> i) & 1.
        
        # This matrix is huge (2187x2187).
        # We can use sparse matrix.
        rows = []
        cols = []
        data = []
        for j in range(total_space):
            mm = masks[j]
            while mm:
                lowbit = mm & -mm
                i = lowbit.bit_length() - 1
                rows.append(i)
                cols.append(j)
                data.append(1)
                mm ^= lowbit
        
        A = csr_matrix((data, (rows, cols)), shape=(total_space, total_space))
        
        # Constraint: A @ x >= 1  =>  -A @ x <= -1
        c = np.ones(total_space)
        b = -np.ones(total_space)
        
        try:
            res = linprog(c, A_ub=-A, b_ub=b, bounds=(0, 1), method='highs')
            if res.status == 0:
                return res.x
        except Exception:
            pass
        return None

    def round_lp(x, time_limit):
        """
        Round LP solution to a valid integer solution.
        """
        if x is None:
            return run_greedy()
        
        # Sort by x_i descending
        indices = np.argsort(x)[::-1]
        selected = []
        covered = 0
        remaining = full_mask
        selected_set = set()
        
        # Add words with high probability
        for idx in indices:
            if not remaining:
                break
            w = idx
            if w in selected_set:
                continue
            # Check if adding w helps
            if (masks[w] & remaining) != 0:
                selected.append(w)
                selected_set.add(w)
                covered |= masks[w]
                remaining = full_mask & ~covered
            if time.time() > time_limit:
                break
        
        # If still not covered, use greedy to finish
        while remaining and time.time() < time_limit:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
        return selected

    def solve_ilp(selected, time_limit):
        if not selected or time.time() > time_limit:
            return selected[:]
        m = len(selected)
        if m < 10:
            return selected[:]

        rows = []
        cols = []
        data = []
        for j, w in enumerate(selected):
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                i = lowbit.bit_length() - 1
                rows.append(i)
                cols.append(j)
                data.append(1)
                mm ^= lowbit
        
        A = csr_matrix((data, (rows, cols)), shape=(total_space, m))
        c = np.ones(m)
        lb = np.ones(total_space)
        ub = np.full(total_space, np.inf)
        lb_vars = np.zeros(m)
        ub_vars = np.ones(m)
        integrality = np.ones(m)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(lb_vars, ub_vars),
                options={"time_limit": max(1.0, time_limit - time.time() - 2.0)}
            )
            if res.status in (0, 1):
                x = res.x
                new_selected = [selected[j] for j in range(m) if x[j] > 0.5]
                if new_selected:
                    covered = 0
                    for w in new_selected:
                        covered |= masks[w]
                    if covered == full_mask:
                        return new_selected
        except Exception:
            pass
        return selected[:]

    best_size = total_space
    best_selected = list(range(total_space))

    # Strategy 1: LP Initialization
    if time.time() < deadline - 20:
        x_lp = solve_lp()
        if x_lp is not None:
            sol_lp = round_lp(x_lp, deadline - 20)
            polished = polish(sol_lp, deadline - 20)
            ls = local_search(polished, deadline - 20)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 2: Greedy Initialization
    if time.time() < deadline - 20:
        sol_greedy = run_greedy(randomized=False)
        polished = polish(sol_greedy, deadline - 20)
        ls = local_search(polished, deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 3: Randomized Greedy Restarts
    for r in range(50):
        if time.time() > deadline - 30:
            break
        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:
            polished = polish(sol, deadline - 30)
            ls = local_search(polished, deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 4: MILP Polish
    if time.time() < deadline - 5:
        ilp_res = solve_ilp(best_selected, deadline - 3)
        if len(ilp_res) < best_size:
            best_size = len(ilp_res)
            best_selected = ilp_res[:]

    # Final Polish
    if time.time() < deadline - 1:
        final_polish = polish(best_selected, deadline - 1)
        if len(final_polish) < best_size:
            best_size = len(final_polish)
            best_selected = final_polish[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
