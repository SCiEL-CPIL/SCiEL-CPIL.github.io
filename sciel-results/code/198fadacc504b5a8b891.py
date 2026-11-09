"""Rewritten program: find 3-coloring of [0,127] where every 4-AP uses ≥3 colors."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    n = 128

    # Precompute all nonconstant 4-term progressions
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))

    num_progs = len(progressions)

    # For each position, list of (prog_index, position_in_prog)
    pos_progs = [[] for _ in range(n)]
    for pi, prog in enumerate(progressions):
        for pos in prog:
            pos_progs[pos].append(pi)

    def count_violations(colors):
        """Count number of 4-APs with < 3 distinct colors."""
        v = 0
        for prog in progressions:
            c = {colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}
            if len(c) < 3:
                v += 1
        return v

    def is_valid(colors):
        """Check if all 4-APs use ≥3 distinct colors."""
        for prog in progressions:
            if len({colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}) < 3:
                return False
        return True

    # Strategy: Try structured 3-colorings first, then local search
    # 
    # Key insight: For a 4-AP (a, a+d, a+2d, a+3d), if we color by f(x) = x mod 3,
    # the colors are a%3, (a+d)%3, (a+2d)%3, (a+3d)%3.
    # If d % 3 == 0: all same color -> violation!
    # So mod 3 doesn't work.
    #
    # Try coloring based on 2-bit patterns or other structures.
    # 
    # Alternative: Use the fact that we need to avoid 4-APs with ≤2 colors.
    # This means we cannot have:
    #   - Monochromatic 4-APs
    #   - 4-APs using exactly 2 colors
    #
    # Let's try a local search approach with multiple restarts.

    def local_search_3color(init_colors, max_iters=50000):
        """Local search to find valid 3-coloring."""
        colors = list(init_colors)
        current_v = count_violations(colors)
        best_colors = list(colors)
        best_v = current_v

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if current_v == 0:
                return colors

            # Find a violated progression
            violated_prog = None
            for prog in progressions:
                c = {colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}
                if len(c) < 3:
                    violated_prog = prog
                    break

            if violated_prog is None:
                return colors

            # Try changing one position in the violated progression
            best_delta = current_v + 1
            best_move = None
            for pos in violated_prog:
                old_c = colors[pos]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    colors[pos] = new_c
                    # Count new violations
                    new_v = 0
                    for pi in pos_progs[pos]:
                        p = progressions[pi]
                        cc = {colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]}
                        if len(cc) < 3:
                            new_v += 1
                    delta = new_v - current_v
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (pos, new_c, new_v)
                    colors[pos] = old_c

            if best_move is not None and best_delta < current_v:
                pos, new_c, new_v = best_move
                colors[pos] = new_c
                current_v = new_v
                if current_v < best_v:
                    best_v = current_v
                    best_colors = list(colors)
            else:
                # No improving move: try a random perturbation
                pos = random.choice(violated_prog)
                colors[pos] = random.randint(0, 2)
                current_v = count_violations(colors)
                if current_v < best_v:
                    best_v = current_v
                    best_colors = list(colors)

        return best_colors

    # Try multiple initializations
    random.seed(42)

    # Initialization 1: Modulo-based patterns
    inits = []
    # Try various mod patterns
    for mod in [3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
        init = [i % mod for i in range(n)]
        # Map to 3 colors
        if mod <= 3:
            inits.append(init)
        else:
            # Hash to 3 colors
            init3 = [init[i] % 3 for i in range(n)]
            inits.append(init3)

    # Random initializations
    for _ in range(20):
        inits.append([random.randint(0, 2) for _ in range(n)])

    # Structured: alternating patterns
    inits.append([i % 3 for i in range(n)])
    inits.append([(i // 2) % 3 for i in range(n)])
    inits.append([(i % 4) % 3 for i in range(n)])

    best_solution = None
    for init in inits:
        if time.time() > deadline - 10:
            break
        sol = local_search_3color(init, max_iters=20000)
        if sol and is_valid(sol):
            best_solution = sol
            break

    if best_solution:
        return best_solution

    # If 3 colors failed, try 4 colors
    def local_search_kcolor(init_colors, k, max_iters=50000):
        """Local search to find valid k-coloring."""
        colors = list(init_colors)
        current_v = count_violations(colors)
        best_colors = list(colors)
        best_v = current_v

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if current_v == 0:
                return colors

            # Find a violated progression
            violated_prog = None
            for prog in progressions:
                c = {colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]}
                if len(c) < 3:
                    violated_prog = prog
                    break

            if violated_prog is None:
                return colors

            # Try changing one position in the violated progression
            best_delta = current_v + 1
            best_move = None
            for pos in violated_prog:
                old_c = colors[pos]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    colors[pos] = new_c
                    # Count new violations
                    new_v = 0
                    for pi in pos_progs[pos]:
                        p = progressions[pi]
                        cc = {colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]}
                        if len(cc) < 3:
                            new_v += 1
                    delta = new_v - current_v
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (pos, new_c, new_v)
                    colors[pos] = old_c

            if best_move is not None and best_delta < current_v:
                pos, new_c, new_v = best_move
                colors[pos] = new_c
                current_v = new_v
                if current_v < best_v:
                    best_v = current_v
                    best_colors = list(colors)
            else:
                # Random perturbation
                pos = random.choice(violated_prog)
                colors[pos] = random.randint(0, k - 1)
                current_v = count_violations(colors)
                if current_v < best_v:
                    best_v = current_v
                    best_colors = list(colors)

        return best_colors

    # Try 4 colors
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        init = [random.randint(0, 3) for _ in range(n)]
        sol = local_search_kcolor(init, 4, max_iters=30000)
        if sol and is_valid(sol):
            return sol

    # Fallback: Greedy with minimal colors
    colors = []
    for x in range(1, 129):
        idx = x - 1
        found = False
        for color in range(128):
            valid = True
            # Check all 4-APs where this position is the last term
            for d in range(1, (x - 1) // 3 + 1):
                i = x - 1 - 3 * d
                if i < 0:
                    continue
                c1 = colors[i]
                c2 = colors[i + d]
                c3 = colors[i + 2 * d]
                c4 = color
                if len({c1, c2, c3, c4}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check that we don't create a dead-end 3-term AP
            for d in range(1, (x - 1) // 2 + 1):
                i = x - 1 - 2 * d
                if i < 0:
                    continue
                if i + 3 * d < 128:
                    if colors[i] == colors[i + d] == color:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)

    return colors
