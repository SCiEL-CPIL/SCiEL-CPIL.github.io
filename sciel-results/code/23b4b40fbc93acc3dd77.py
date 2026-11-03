import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Aggressive local improvement
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass
    if time.time() < deadline - 5.0:
        polished = _polish_scope(best_solution, n_rows, k, deadline)
        polished_scope = max(max(row) for row in polished)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_solution = polished

        if best_scope <= lower_bound:
            return best_solution

    # 4. Spectral-guided greedy construction
    if time.time() < deadline - 15.0:
        spectral_solution = _spectral_greedy(n_rows, k, deadline)
        if spectral_solution:
            spectral_scope = max(max(row) for row in spectral_solution)
            if spectral_scope < best_scope:
                best_scope = spectral_scope
                best_solution = spectral_solution
                best_solution = _improve_scope(best_solution, n_rows, k, deadline)
                best_scope = max(max(row) for row in best_solution)
                if best_scope <= lower_bound:
                    return best_solution

    # 5. Randomized backtracking for target scopes below current best
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(40):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Improve and polish
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if time.time() < deadline - 3.0:
                        polished = _polish_scope(best_solution, n_rows, k, deadline)
                        new_scope = max(max(row) for row in polished)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 5:
            if target_scope < lower_bound:
                break

    # Final polish
    if time.time() < deadline - 2.0:
        best_solution = _polish_scope(best_solution, n_rows, k, deadline)
        best_scope = max(max(row) for row in best_solution)

    return best_solution


def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _spectral_greedy(n_rows, k, deadline):
    """
    Spectral relaxation-guided greedy construction using difference rarity.
    """
    k_plus_1 = k + 1
    max_scope = 250

    rng = random.Random(42)
    diff_rarity = np.zeros(max_scope + 1)

    n_samples = min(150, n_rows * 8)
    for _ in range(n_samples):
        row = [0]
        x = 0
        while len(row) < k_plus_1 and x < max_scope:
            x += 1
            new_diffs = {x - a for a in row}
            if all(d <= max_scope for d in new_diffs):
                row.append(x)

        if len(row) == k_plus_1:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    if d <= max_scope:
                        diff_rarity[d] += 1

    total = diff_rarity.sum()
    if total > 0:
        rarity_score = diff_rarity / total
    else:
        rarity_score = np.ones(max_scope + 1) / (max_scope + 1)

    priority = 1.0 / (rarity_score + 0.01)
    priority[0] = 0

    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k_plus_1:
            if time.time() > deadline:
                return None
            x += 1
            if x > max_scope:
                return None
            new_diffs = {x - a for a in row}
            if any(d in used for d in new_diffs):
                continue
            if max(new_diffs) > max_scope:
                continue
            row.append(x)
            used.update(new_diffs)

        rows.append(row)

    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)

    return rows


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with bitset for fast diff checking and better candidate ordering.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    # Use a Python int as a bitset for used differences
    used_bits = 0
    max_bits = max_scope + 1

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            # Check if any difference conflicts using bit operations
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_bits & (1 << d):
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 50:
                break

        if not valid_candidates:
            return None

        # Try smallest candidates first, with some randomization
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Compute new differences and check conflicts
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)

            # Check all new diffs are available
            bits_to_set = 0
            for d in new_diffs:
                bits_to_set |= (1 << d)

            if used_bits & bits_to_set:
                continue

            used_bits |= bits_to_set
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            used_bits &= ~bits_to_set

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                used_bits |= (1 << d)

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: pull each element to smallest feasible value,
    then use 2-element swaps to escape local minima.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0:
                        ok = False
                        break
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                for j in range(pos + 1, k_plus_1):
                    d = row[j] - new_val
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue

                # Remove old diffs
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(row[pos] - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.add(row[j] - row[pos])

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    def _try_2swap(row_idx, pos_a, pos_b):
        """Try swapping values at pos_a and pos_b (or shifting one) to escape local min."""
        row = best[row_idx]
        if pos_a >= pos_b:
            return False
        old_a, old_b = row[pos_a], row[pos_b]
        # Try shifting pos_b down to pos_a+1, then pos_a to its smallest
        # Simple: try all feasible pairs (new_a, new_b) with new_a < new_b
        # This is expensive, so limit range
        min_a = row[pos_a - 1] + 1 if pos_a > 0 else 1
        max_a = old_a
        min_b = old_a + 1
        max_b = old_b

        for new_a in range(min_a, max_a + 1):
            for new_b in range(max(min_b, new_a + 1), max_b + 1):
                new_diffs = set()
                ok = True
                # Diffs involving pos_a
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_a:
                        d = new_a - row[i]
                    else:
                        d = row[i] - new_a
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                # Diffs involving pos_b
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_b:
                        d = new_b - row[i]
                    else:
                        d = row[i] - new_b
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                # Diff between pos_a and pos_b
                d_ab = new_b - new_a
                if d_ab <= 0 or d_ab in used_diffs:
                    continue
                new_diffs.add(d_ab)

                # Remove old diffs
                old_diffs = set()
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_a:
                        old_diffs.add(old_a - row[i])
                    else:
                        old_diffs.add(row[i] - old_a)
                    if i < pos_b:
                        old_diffs.add(old_b - row[i])
                    else:
                        old_diffs.add(row[i] - old_b)
                old_diffs.add(old_b - old_a)

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos_a] = new_a
                row[pos_b] = new_b
                return True
        return False

    improved = True
    rng = random.Random(12345)
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            if _pull_to_smallest(row_idx):
                improved = True

        # 2-swap perturbations to escape local minima - increase frequency
        n_swaps = 8 if iteration % 3 == 0 else 5
        for _ in range(n_swaps):
            if time.time() > deadline:
                break
            row_idx = rng.randint(0, n_rows - 1)
            pos_a = rng.randint(1, k_plus_1 - 2)
            pos_b = rng.randint(pos_a + 1, k_plus_1 - 1)
            if _try_2swap(row_idx, pos_a, pos_b):
                improved = True
                # Pull to smallest again after swap
                if _pull_to_smallest(row_idx):
                    improved = True
        
        # Additional: try single-element shifts for more aggressive exploration
        if iteration % 2 == 0 and time.time() < deadline:
            for _ in range(3):
                if time.time() > deadline:
                    break
                row_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k_plus_1 - 1)
                row = best[row_idx]
                old_val = row[pos]
                
                # Try shifting to a slightly larger value to enable later reductions
                for delta in [2, 3, 5]:
                    new_val = old_val + delta
                    if new_val > max(max(r) for r in best):
                        continue
                    
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - row[pos])

                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    
                    # Try to pull back down
                    if _pull_to_smallest(row_idx):
                        improved = True
                    break
        
        iteration += 1

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    Final high-precision polish: multiple passes of pull-to-smallest with
    tighter local search.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                for j in range(pos + 1, k_plus_1):
                    d = row[j] - new_val
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue

                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(row[pos] - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.add(row[j] - row[pos])

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    # Multiple passes with increasing aggressiveness
    for pass_num in range(5):
        if time.time() > deadline:
            break
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            _pull_to_smallest(row_idx)
        
        # After each full pass, try more aggressive 2-swaps
        if pass_num >= 2 and time.time() < deadline:
            for _ in range(3):
                if time.time() > deadline:
                    break
                row_idx = random.Random(pass_num * 7 + 13).randint(0, n_rows - 1)
                pos_a = random.Random(pass_num * 11 + 17).randint(1, k_plus_1 - 2)
                pos_b = random.Random(pass_num * 13 + 19).randint(pos_a + 1, k_plus_1 - 1)
                
                # Inline simplified 2-swap for polish
                row = best[row_idx]
                old_a, old_b = row[pos_a], row[pos_b]
                min_a = row[pos_a - 1] + 1 if pos_a > 0 else 1
                max_a = old_a
                min_b = old_a + 1
                max_b = old_b

                for new_a in range(min_a, max_a + 1):
                    for new_b in range(max(min_b, new_a + 1), max_b + 1):
                        new_diffs = set()
                        ok = True
                        for i in range(k_plus_1):
                            if i == pos_a or i == pos_b:
                                continue
                            if i < pos_a:
                                d = new_a - row[i]
                            else:
                                d = row[i] - new_a
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if not ok:
                            continue
                        for i in range(k_plus_1):
                            if i == pos_a or i == pos_b:
                                continue
                            if i < pos_b:
                                d = new_b - row[i]
                            else:
                                d = row[i] - new_b
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if not ok:
                            continue
                        d_ab = new_b - new_a
                        if d_ab <= 0 or d_ab in used_diffs:
                            continue
                        new_diffs.add(d_ab)

                        old_diffs = set()
                        for i in range(k_plus_1):
                            if i == pos_a or i == pos_b:
                                continue
                            if i < pos_a:
                                old_diffs.add(old_a - row[i])
                            else:
                                old_diffs.add(row[i] - old_a)
                            if i < pos_b:
                                old_diffs.add(old_b - row[i])
                            else:
                                old_diffs.add(row[i] - old_b)
                        old_diffs.add(old_b - old_a)

                        if new_diffs & (used_diffs - old_diffs):
                            continue

                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos_a] = new_a
                        row[pos_b] = new_b
                        break
                    else:
                        continue
                    break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
