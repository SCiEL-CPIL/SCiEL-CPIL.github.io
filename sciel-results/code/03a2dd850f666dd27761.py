import time
import math
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    divisors.sort()
    num_divs = len(divisors)

    # 2. Precompute masks for each (m, r) as Python integers (bitsets)
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for x in range(r, base, m):
                mask_val |= (1 << x)
            masks[m][r] = mask_val

    # Precompute bit_count for each mask for faster access
    mask_counts = {}
    for m in divisors:
        mask_counts[m] = [0] * m
        for r in range(m):
            mask_counts[m][r] = masks[m][r].bit_count()

    best_classes = None
    best_uncovered = base  # start with all uncovered

    def evaluate(classes):
        """Compute number of uncovered residues."""
        freq = [0] * base
        for m, r in classes:
            for x in range(r, base, m):
                freq[x] += 1
        uncovered = 0
        for x in range(base):
            if freq[x] == 0:
                uncovered += 1
        return uncovered

    def run_greedy(div_order, deadline):
        """Run greedy set cover with given divisor order. Returns (classes, uncovered_count)."""
        uncovered_mask = (1 << base) - 1
        freq = [0] * base
        used_moduli = set()
        classes = []
        remaining = set(div_order)

        while remaining:
            best_gain = 0
            best_m = None
            best_r = None
            for m in remaining:
                for r in range(m):
                    gain = (uncovered_mask & masks[m][r]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
            if best_gain == 0:
                break
            # Add class
            for x in range(best_r, base, best_m):
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                freq[x] += 1
            used_moduli.add(best_m)
            classes.append([best_m, best_r])
            remaining.discard(best_m)
            if time.time() > deadline:
                break
        uncovered_count = base - uncovered_mask.bit_count()
        return classes, uncovered_count

    def local_search(classes, divisors, deadline):
        """Local search: try swaps and adds. Returns improved (classes, uncovered_count)."""
        # Build state
        freq = [0] * base
        uncovered_mask = (1 << base) - 1
        unique_mask = 0
        for m, r in classes:
            for x in range(r, base, m):
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                    unique_mask |= (1 << x)
                elif freq[x] == 1:
                    unique_mask &= ~(1 << x)
                freq[x] += 1

        used_moduli = set(m for m, r in classes)
        remaining_moduli = set(divisors) - used_moduli
        uncovered_count = base - uncovered_mask.bit_count()

        def add_class(m, r):
            nonlocal uncovered_mask, unique_mask, uncovered_count
            for x in range(r, base, m):
                if freq[x] == 0:
                    uncovered_mask &= ~(1 << x)
                    unique_mask |= (1 << x)
                    uncovered_count -= 1
                elif freq[x] == 1:
                    unique_mask &= ~(1 << x)
                freq[x] += 1
            used_moduli.add(m)
            classes.append([m, r])
            remaining_moduli.discard(m)

        def remove_class(m, r):
            nonlocal uncovered_mask, unique_mask, uncovered_count
            for x in range(r, base, m):
                freq[x] -= 1
                if freq[x] == 1:
                    unique_mask |= (1 << x)
                elif freq[x] == 0:
                    unique_mask &= ~(1 << x)
                    uncovered_mask |= (1 << x)
                    uncovered_count += 1
            used_moduli.discard(m)
            # Remove from classes list
            for i, (cm, cr) in enumerate(classes):
                if cm == m and cr == r:
                    classes.pop(i)
                    break
            remaining_moduli.add(m)

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: Try adding any unused modulus (if it reduces uncovered)
            if remaining_moduli:
                best_add_gain = 0
                best_add_m = None
                best_add_r = None
                for m in remaining_moduli:
                    for r in range(m):
                        gain = (uncovered_mask & masks[m][r]).bit_count()
                        if gain > best_add_gain:
                            best_add_gain = gain
                            best_add_m = m
                            best_add_r = r
                if best_add_gain > 0:
                    add_class(best_add_m, best_add_r)
                    improved = True
                    if time.time() > deadline:
                        break

            # Phase 2: Try swaps (remove one used, add one unused)
            if remaining_moduli:
                best_swap_gain = 0
                best_old_idx = -1
                best_old_m = None
                best_old_r = None
                best_new_m = None
                best_new_r = None

                # Precompute for each used class: loss and u_expanded
                # loss_i = popcount(masks[m_i][r_i] & unique_mask)
                # u_expanded_i = uncovered_mask | (masks[m_i][r_i] & unique_mask)
                used_list = list(classes)
                n_used = len(used_list)

                for i in range(n_used):
                    m_old, r_old = used_list[i]
                    mask_old = masks[m_old][r_old]
                    loss = (mask_old & unique_mask).bit_count()
                    u_expanded = uncovered_mask | (mask_old & unique_mask)

                    for m_new in remaining_moduli:
                        # Find best r_new for m_new against u_expanded
                        best_r_new = 0
                        best_int_count = 0
                        for r_new in range(m_new):
                            cnt = (masks[m_new][r_new] & u_expanded).bit_count()
                            if cnt > best_int_count:
                                best_int_count = cnt
                                best_r_new = r_new

                        net_gain = best_int_count - loss
                        if net_gain > best_swap_gain:
                            best_swap_gain = net_gain
                            best_old_idx = i
                            best_old_m = m_old
                            best_old_r = r_old
                            best_new_m = m_new
                            best_new_r = r_new

                    if time.time() > deadline:
                        break

                if best_swap_gain > 0 and best_old_idx >= 0:
                    remove_class(best_old_m, best_old_r)
                    add_class(best_new_m, best_new_r)
                    improved = True

            # Phase 3: Try re-residue (change residue of a used modulus)
            # For each used class, try all other residues
            if time.time() < deadline:
                best_res_gain = 0
                best_res_idx = -1
                best_res_m = None
                best_res_old_r = None
                best_res_new_r = None

                used_list = list(classes)
                n_used = len(used_list)
                for i in range(n_used):
                    m, r_old = used_list[i]
                    mask_old = masks[m][r_old]
                    loss = (mask_old & unique_mask).bit_count()
                    u_expanded = uncovered_mask | (mask_old & unique_mask)

                    for r_new in range(m):
                        if r_new == r_old:
                            continue
                        cnt = (masks[m][r_new] & u_expanded).bit_count()
                        net_gain = cnt - loss
                        if net_gain > best_res_gain:
                            best_res_gain = net_gain
                            best_res_idx = i
                            best_res_m = m
                            best_res_old_r = r_old
                            best_res_new_r = r_new

                    if time.time() > deadline:
                        break

                if best_res_gain > 0 and best_res_idx >= 0:
                    remove_class(best_res_m, best_res_old_r)
                    add_class(best_res_m, best_res_new_r)
                    improved = True

        return classes, uncovered_count

    # Main: try multiple greedy orderings and local search
    # Deterministic first
    classes, unc = run_greedy(divisors, deadline)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = [list(c) for c in classes]

    # Local search on the greedy result
    if time.time() < deadline:
        classes, unc = local_search([list(c) for c in classes], divisors, deadline)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [list(c) for c in classes]

    # Try a few random orderings if time permits
    rng = random.Random(42)
    max_restarts = 5
    for trial in range(max_restarts):
        if time.time() > deadline - 30:
            break
        # Random shuffle of divisors
        shuffled = list(divisors)
        rng.shuffle(shuffled)
        classes, unc = run_greedy(shuffled, deadline)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [list(c) for c in classes]
        # Local search
        if time.time() < deadline:
            classes, unc = local_search([list(c) for c in classes], divisors, deadline)
            if unc < best_uncovered:
                best_uncovered = unc
                best_classes = [list(c) for c in classes]

    # Final polish: one more local search pass on best solution
    if best_classes and time.time() < deadline:
        classes, unc = local_search([list(c) for c in best_classes], divisors, deadline)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [list(c) for c in classes]

    if best_classes is None:
        # Fallback: return empty (shouldn't happen)
        return []

    return best_classes
