"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    n = 128

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d)
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

    # Precompute APs ending at each index (for fast greedy)
    aps_end = [[] for _ in range(n)]
    for i in range(n):
        max_d = i // 3
        for d in range(1, max_d + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            aps_end[i].append((a, b, c))

    def is_valid_at(idx, c, colors):
        """Check if assigning color c to idx satisfies all 4-AP constraints ending at idx."""
        for (a, b, cc) in aps_end[idx]:
            if len({colors[a], colors[b], colors[cc], c}) < 3:
                return False
        return True

    def count_violations(colors):
        return sum(
            1 for ap in aps
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3
        )

    def validate(colors):
        return count_violations(colors) == 0

    # Randomized greedy: try to build a valid K-coloring
    def greedy_random(K, seed):
        rng = random.Random(seed)
        colors = [-1] * n
        for idx in range(n):
            valid_colors = []
            for c in range(K):
                if is_valid_at(idx, c, colors):
                    valid_colors.append(c)
            if not valid_colors:
                return None
            # Random choice among valid colors
            colors[idx] = rng.choice(valid_colors)
        return colors

    # Deterministic greedy fallback
    def greedy_deterministic():
        colors = [0] * n
        for x in range(n):
            found = False
            for c in range(128):
                if is_valid_at(x, c, colors):
                    colors[x] = c
                    found = True
                    break
            if not found:
                return None
        return colors

    # Simulated annealing to fix violations
    def simulated_annealing(colors, K, max_steps=100000):
        current = list(colors)
        current_viol = count_violations(current)
        if current_viol == 0:
            return current

        best = current[:]
        best_viol = current_viol
        temp = 1.0
        cooling = 0.99995
        rng = random.Random(42)

        steps = 0
        while steps < max_steps and time.time() < deadline - 10:
            steps += 1
            i = rng.randint(0, n - 1)
            old_color = current[i]
            new_color = rng.randint(0, K - 1)
            if new_color == old_color:
                continue

            # Compute delta in violations
            delta = 0
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                was = len({current[ap[0]], current[ap[1]], current[ap[2]], current[ap[3]]}) < 3
                current[i] = new_color
                now = len({current[ap[0]], current[ap[1]], current[ap[2]], current[ap[3]]}) < 3
                current[i] = old_color
                if was and not now:
                    delta -= 1
                elif not was and now:
                    delta += 1

            new_viol = current_viol + delta

            if new_viol <= current_viol or rng.random() < (2.718281828 ** (-(new_viol - current_viol) / temp)):
                current[i] = new_color
                current_viol = new_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best = current[:]
                if current_viol == 0:
                    break

            temp *= cooling

        return best if best_viol == 0 else current

    # Polish: reduce number of colors used in a valid coloring
    def polish_reduce_colors(colors, max_iters=5000):
        current = list(colors)
        current_num = len(set(current))
        best_num = current_num
        best_colors = current[:]
        rng = random.Random(123)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            pos = rng.randint(0, n - 1)
            old_color = current[pos]
            existing = list(set(current))
            rng.shuffle(existing)
            improved = False
            for new_color in existing:
                if new_color == old_color:
                    continue
                # Check validity
                valid = True
                for (a, b, cc) in aps_end[pos]:
                    if len({current[a], current[b], current[cc], new_color}) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                # Check if this reduces color count
                current[pos] = new_color
                if old_color not in current:
                    new_num = len(set(current))
                    if new_num < best_num:
                        best_num = new_num
                        best_colors = current[:]
                    improved = True
                current[pos] = old_color
                if improved:
                    break
            if not improved:
                continue

        return best_colors

    # Strategy 1: Try randomized greedy with increasing K
    best_colors = None
    best_num_colors = 128

    for K in range(3, 15):
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(200):
            if time.time() > deadline - 20:
                break
            result = greedy_random(K, seed)
            if result is not None and validate(result):
                num_used = len(set(result))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_colors = result[:]
                found = True
                break
        if found and best_num_colors <= K:
            # Found a valid coloring with K colors, can't do better with this K
            # But check if we used fewer than K colors
            if best_colors is not None and len(set(best_colors)) <= K:
                break

    # Strategy 2: If no valid coloring found, use simulated annealing
    if best_colors is None or best_num_colors > 8:
        if time.time() < deadline - 15:
            for K in range(4, 12):
                if time.time() > deadline - 15:
                    break
                # Random initialization
                colors = [random.randint(0, K - 1) for _ in range(n)]
                result = simulated_annealing(colors, K, max_steps=50000)
                if validate(result):
                    num_used = len(set(result))
                    if num_used < best_num_colors:
                        best_num_colors = num_used
                        best_colors = result[:]
                    break

    # Fallback: deterministic greedy
    if best_colors is None:
        best_colors = greedy_deterministic()
        if best_colors is None:
            # Last resort: all different colors
            best_colors = list(range(n))

    # Polish: reduce color count
    if time.time() < deadline - 5:
        polished = polish_reduce_colors(best_colors, max_iters=10000)
        if validate(polished) and len(set(polished)) < len(set(best_colors)):
            best_colors = polished

    return best_colors
# EVOLVE-BLOCK-END
