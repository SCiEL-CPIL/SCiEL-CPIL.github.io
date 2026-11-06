"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # Compute degree of each position
    degree = [len(pos_to_aps[i]) for i in range(n)]

    # Order positions by degree descending (most constrained first)
    order = sorted(range(n), key=lambda x: -degree[x])

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v

    def backtracking_search(max_colors, time_limit, seed=0):
        """
        Backtracking with forward checking to find a valid coloring using at most max_colors colors.
        Returns a valid coloring or None if not found within time limit.
        """
        rng = random.Random(seed)
        colors = [-1] * n
        color_counts = [0] * max_colors

        # Track which APs are "active" (have at least one unassigned position)
        # For forward checking, we track for each AP how many distinct colors are used
        # and whether it's already invalid

        def get_ap_state(ap):
            """
            For an AP, determine its state given current partial assignment:
            - Returns (num_unassigned, distinct_colors_so_far, is_forced_invalid)
            - is_forced_invalid: True if even with remaining positions, AP cannot have >= 3 colors
            """
            assigned_colors = set()
            unassigned = 0
            for pos in ap:
                if colors[pos] == -1:
                    unassigned += 1
                else:
                    assigned_colors.add(colors[pos])

            num_distinct = len(assigned_colors)

            # If we have fewer than 3 distinct colors and not enough unassigned to reach 3
            # Actually, we need at least 3 distinct in total. If we have 0 or 1 distinct assigned,
            # we need at least 2 or 3 more distinct from unassigned positions.
            # If we have 2 distinct assigned, we need at least 1 more distinct from unassigned.

            # The AP is forced invalid if:
            # - We already have all 4 positions assigned and < 3 distinct
            # - Or: the number of distinct colors we can still achieve < 3
            #   Max distinct = num_distinct + unassigned (if all unassigned get new colors)
            #   But also limited by max_colors

            if unassigned == 0:
                if num_distinct < 3:
                    return (0, num_distinct, True)
                else:
                    return (0, num_distinct, False)

            # Can we still reach 3 distinct?
            # We have num_distinct already, and unassigned positions can add at most min(unassigned, max_colors - num_distinct) new distinct colors
            # But actually, the unassigned positions can be any colors from 0..max_colors-1
            # So max additional distinct = min(unassigned, max_colors - num_distinct) but capped by what's needed
            # We need total >= 3, so we need at least (3 - num_distinct) new distinct colors
            # This is possible if unassigned >= (3 - num_distinct) and max_colors >= 3

            needed = 3 - num_distinct
            if needed <= 0:
                return (unassigned, num_distinct, False)

            # We need 'needed' new distinct colors from 'unassigned' positions
            # This is possible if unassigned >= needed (since each unassigned can be a new color)
            # AND max_colors >= num_distinct + needed (enough colors available)
            if unassigned < needed:
                return (unassigned, num_distinct, True)
            if num_distinct + needed > max_colors:
                return (unassigned, num_distinct, True)

            return (unassigned, num_distinct, False)

        # Precompute for each position, which APs it's in
        # We'll use forward checking: after assigning a color to a position,
        # check all APs containing that position for forced invalidity

        def is_position_feasible(pos, color):
            """Check if assigning 'color' to 'pos' doesn't immediately force any AP invalid."""
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Simulate assigning this color
                old_color = colors[pos]
                colors[pos] = color
                unassigned, num_distinct, invalid = get_ap_state(ap)
                colors[pos] = old_color
                if invalid:
                    return False
            return True

        def backtrack(idx):
            """Recursively assign colors to positions in order[idx:]."""
            if time.time() > time_limit:
                return False

            if idx == n:
                return True

            pos = order[idx]

            # Try colors in order of least frequent first (to use fewer colors)
            # But we want to find ANY valid coloring with at most max_colors
            # Use MRV-like: try colors that leave most flexibility

            # Shuffle colors for diversity
            color_order = list(range(max_colors))
            # Sort by: prefer colors already used (to minimize distinct colors)
            used_colors = set(colors)
            color_order.sort(key=lambda c: (0 if c in used_colors else 1, c))

            for color in color_order:
                if time.time() > time_limit:
                    return False

                if is_position_feasible(pos, color):
                    colors[pos] = color
                    color_counts[color] += 1

                    if backtrack(idx + 1):
                        return True

                    colors[pos] = -1
                    color_counts[color] -= 1

            return False

        if backtrack(0):
            return colors[:]
        return None

    def local_search_refine(colors, time_limit):
        """Simulated annealing to reduce color count while maintaining validity."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1

        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used

        temp = 3.0
        cooling = 0.997
        min_temp = 0.01

        while time.time() < time_limit and temp > min_temp:
            pos = random.randint(0, n - 1)
            old_color = colors[pos]

            used = set(colors)
            candidates = list(used)
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)

            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0

            affected_aps = pos_to_aps[pos]

            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                if len(c) < 3:
                    curr_invalid += 1

            for new_color in candidates:
                if new_color == old_color:
                    continue

                colors[pos] = new_color

                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    if len(c) < 3:
                        new_invalid += 1

                delta_v = new_invalid - curr_invalid
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1

                score = delta_v * 100 + delta_c
                best_score = best_delta_v * 100 + best_delta_c

                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c

            colors[pos] = old_color

            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob

            if accept:
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                current_violations += best_delta_v
                current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)

                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]

                if current_violations == 0 and current_colors_used <= 3:
                    break

            temp *= cooling

        return best_colors, best_violations, best_colors_used

    # Strategy: Try backtracking with increasing max_colors (3, 4, 5, ...)
    # If backtracking succeeds, we have a valid coloring with that many colors.
    # Then use local search to try to reduce further.

    best_result = None
    best_num_colors = 999

    # Try max_colors from 3 upward
    for max_c in range(3, 8):
        if time.time() > deadline - 5.0:
            break

        # Give backtracking a time budget proportional to remaining time
        bt_time_limit = min(time.time() + 10.0, deadline - 5.0)

        result = backtracking_search(max_c, bt_time_limit, seed=max_c * 1000)

        if result is not None:
            v = count_violations(result)
            nc = len(set(result))
            if v == 0 and nc <= max_c:
                best_result = result
                best_num_colors = nc
                break
            elif v == 0 and nc < best_num_colors:
                best_result = result
                best_num_colors = nc

    # If backtracking didn't find anything, fall back to greedy + local search
    if best_result is None:
        # Greedy construction
        colors = [-1] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] == -1:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] == -1:
                colors[x] = x

        v = count_violations(colors)
        nc = len(set(colors))
        if v == 0:
            best_result = colors
            best_num_colors = nc

    # Local search refinement to try to reduce colors further
    if best_result is not None and time.time() < deadline - 3.0:
        refined, final_v, final_c = local_search_refine(best_result, deadline - 2.0)
        if final_v == 0 and final_c < best_num_colors:
            best_result = refined
            best_num_colors = final_c

    # Final fallback
    if best_result is None:
        return list(range(n))

    return best_result
# EVOLVE-BLOCK-END
