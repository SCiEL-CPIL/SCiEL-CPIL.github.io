import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    # Precompute all 4-APs: (a, a+d, a+2d, a+3d)
    # For each position x, store the APs where x is the last element
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            i = x - 3 * d
            if i >= 1:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))

    def solve_with_deadline(k, dl, seed_val):
        """
        Try to find a valid coloring with k colors.
        Uses randomized backtracking.
        """
        rng = random.Random(seed_val)
        colors = [None] * (N + 1)
        colors[1] = 0  # Symmetry breaking

        # Precompute for each position x, the list of APs ending at x
        # and for fast checking, we'll track which colors are used in the last 3 terms

        def backtrack(x, depth):
            if time.time() > dl:
                return None

            if x > N:
                return colors[1:].copy()

            # Get valid colors for position x
            valid_colors = list(range(k))
            rng.shuffle(valid_colors)

            for c in valid_colors:
                # Check all APs ending at x
                valid = True
                for (i, j, k_idx) in aps_ending[x]:
                    ci = colors[i]
                    cj = colors[j]
                    ck = colors[k_idx]
                    # Check if {ci, cj, ck, c} has >= 3 distinct colors
                    # Fast check: count distinct
                    if ci == cj:
                        if cj == ck:
                            # ci=cj=ck, need c != ci
                            if c == ci:
                                valid = False
                                break
                        else:
                            # ci=cj != ck, need c not in {ci, ck}
                            if c == ci or c == ck:
                                valid = False
                                break
                    else:
                        # ci != cj
                        if ci == ck:
                            # ci=ck != cj, need c not in {ci, cj}
                            if c == ci or c == cj:
                                valid = False
                                break
                        elif cj == ck:
                            # cj=ck != ci, need c not in {cj, ci}
                            if c == cj or c == ci:
                                valid = False
                                break
                        else:
                            # all three distinct, any c is fine
                            pass

                if valid:
                    colors[x] = c
                    res = backtrack(x + 1, depth + 1)
                    if res is not None:
                        return res
                    colors[x] = None

            return None

        return backtrack(2, 0)

    # Try k=3 with multiple random seeds
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        sol = solve_with_deadline(3, deadline - 10, attempt * 7 + 1)
        if sol:
            used = sorted(set(sol))
            mapping = {c: i for i, c in enumerate(used)}
            return [mapping[c] for c in sol]

    # Try k=4 with multiple random seeds
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        sol = solve_with_deadline(4, deadline - 5, attempt * 13 + 100)
        if sol:
            used = sorted(set(sol))
            mapping = {c: i for i, c in enumerate(used)}
            return [mapping[c] for c in sol]

    # Fallback: greedy construction with more colors
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                i = x - 3 * d
                if i >= 1:
                    j = i + d
                    k = i + 2 * d
                    # Get colors at i, j, k (0-indexed: i-1, j-1, k-1)
                    ci = colors[i - 1]
                    cj = colors[j - 1]
                    ck = colors[k - 1]
                    if len({ci, cj, ck, color}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen with 128 colors
            colors.append(127)

    used = sorted(set(colors))
    mapping = {c: i for i, c in enumerate(used)}
    return [mapping[c] for c in colors]
