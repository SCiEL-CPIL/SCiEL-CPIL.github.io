import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s margin for safety (evaluator kills at 90s)
    n = 128

    # Precompute all 4-term APs (0-indexed): (a, a+d, a+2d, a+3d)
    aps = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < n:
                aps.append((a, a + d, a + 2 * d, i3))

    # For each position, list of AP indices that include it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def ap_violation(ap, colors):
        """Check if a 4-AP uses fewer than 3 distinct colors."""
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Count distinct colors
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct < 3

    def count_violations(colors):
        """Count total number of violating 4-APs."""
        count = 0
        for ap in aps:
            if ap_violation(ap, colors):
                count += 1
        return count

    def delta_violations(pos, old_color, new_color, colors):
        """Compute change in violation count when changing colors[pos] from old_color to new_color."""
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            ap = aps[ap_idx]
            # Check before
            was_violation = False
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                was_violation = True
            else:
                # Count distinct
                distinct = 1
                if c1 != c0:
                    distinct += 1
                if c2 != c0 and c2 != c1:
                    distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2:
                    distinct += 1
                if distinct < 3:
                    was_violation = True

            # Check after (temporarily set)
            colors[pos] = new_color
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            now_violation = False
            if c0 == c1 == c2 == c3:
                now_violation = True
            else:
                distinct = 1
                if c1 != c0:
                    distinct += 1
                if c2 != c0 and c2 != c1:
                    distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2:
                    distinct += 1
                if distinct < 3:
                    now_violation = True
            colors[pos] = old_color

            if was_violation and not now_violation:
                delta -= 1
            elif not was_violation and now_violation:
                delta += 1
        return delta

    def greedy_valid(max_colors=128):
        """Greedy construction trying to use few colors."""
        colors = [0] * n
        for x in range(n):
            if time.time() > deadline:
                return None
            found = False
            for c in range(max_colors):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    vals = []
                    for p in ap:
                        if p == x:
                            vals.append(c)
                        else:
                            vals.append(colors[p])
                    # Check distinct count
                    distinct = len(set(vals))
                    if distinct < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    found = True
                    break
            if not found:
                return None
        return colors

    best_colors = None
    best_num_colors = 128

    # Strategy: Try to find valid colorings with increasing k colors
    # Start with small k and use simulated annealing

    for k in range(3, 15):
        if time.time() > deadline - 20:
            break

        # Multiple random restarts for this k
        found_valid = False
        for restart in range(5):
            if time.time() > deadline - 15:
                break

            # Random initialization with k colors
            colors = [random.randint(0, k - 1) for _ in range(n)]
            current_violations = count_violations(colors)

            if current_violations == 0:
                best_colors = colors[:]
                best_num_colors = len(set(colors))
                found_valid = True
                break

            best_colors_local = colors[:]
            best_violations = current_violations

            # Simulated annealing
            temp = 2.0
            cooling = 0.99995
            steps = 0
            max_steps = 150000

            while steps < max_steps and time.time() < deadline - 12:
                steps += 1
                i = random.randint(0, n - 1)
                old_color = colors[i]
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue

                delta = delta_violations(i, old_color, new_color, colors)
                new_violations = current_violations + delta

                if new_violations <= current_violations:
                    accept = True
                else:
                    # Accept with probability exp(-delta/temp)
                    if temp > 0.001:
                        accept = random.random() < pow(2.718281828, -delta / temp)
                    else:
                        accept = False

                if accept:
                    colors[i] = new_color
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors_local = colors[:]
                    if current_violations == 0:
                        break

                temp *= cooling

            if best_violations == 0:
                best_colors = best_colors_local[:]
                best_num_colors = len(set(best_colors))
                found_valid = True
                break

            # Keep track of best found for this k
            if best_colors is None or len(set(best_colors_local)) < best_num_colors:
                if best_violations < 10:  # Only keep near-valid solutions
                    best_colors = best_colors_local[:]
                    best_num_colors = len(set(best_colors))

        if found_valid:
            break

    if best_colors is None:
        # Fallback: greedy
        g = greedy_valid()
        if g is not None:
            return g
        # Last resort: use many colors
        colors = list(range(n))
        return colors

    # Polish: try to reduce the number of colors used
    colors = best_colors[:]
    current_num = len(set(colors))
    best_num = current_num
    best_polished = colors[:]

    # Local search to merge colors
    for iteration in range(20000):
        if time.time() > deadline - 3:
            break

        # Pick a random color to try to eliminate
        used_colors = list(set(colors))
        if len(used_colors) <= best_num:
            break

        target_color = random.choice(used_colors)

        # Try to reassign all positions with target_color to other colors
        positions_with_target = [i for i in range(n) if colors[i] == target_color]
        random.shuffle(positions_with_target)

        success = True
        for pos in positions_with_target:
            # Try to find a valid alternative color
            found_alt = False
            for new_color in used_colors:
                if new_color == target_color:
                    continue
                # Check validity
                valid = True
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    vals = []
                    for p in ap:
                        if p == pos:
                            vals.append(new_color)
                        else:
                            vals.append(colors[p])
                    if len(set(vals)) < 3:
                        valid = False
                        break
                if valid:
                    colors[pos] = new_color
                    found_alt = True
                    break

            if not found_alt:
                success = False
                break

        if success:
            new_num = len(set(colors))
            if new_num < best_num:
                best_num = new_num
                best_polished = colors[:]
                # Update used_colors
                used_colors = list(set(colors))

    if best_num < best_num_colors:
        return best_polished
    return best_colors
