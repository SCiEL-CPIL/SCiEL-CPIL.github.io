"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin within 90s wall-clock

    n = 128

    # Precompute constraints for 4-APs ending at x (0-indexed)
    # For position x (0-indexed, 1-based pos = x+1):
    # AP: a, a+d, a+2d, a+3d = pos => a = pos - 3d
    # Indices: a-1, a+d-1, a+2d-1, x
    constraints = [[] for _ in range(n)]
    for x in range(n):
        pos = x + 1
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                continue
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[x].append((i1, i2, i3))

    def check_validity(colors):
        """Check if coloring satisfies the constraint."""
        for x in range(n):
            for i1, i2, i3 in constraints[x]:
                c1 = colors[i1]
                c2 = colors[i2]
                c3 = colors[i3]
                c4 = colors[x]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def count_colors(colors):
        return len(set(colors))

    def try_greedy(max_c, seed_val=0):
        """
        Greedy: assign colors left to right.
        For each position, find all valid colors, prefer already-used ones.
        """
        rng = random.Random(seed_val)
        colors = [-1] * n
        used = set()

        for x in range(n):
            if time.time() > deadline:
                return None

            valid_colors = []
            for c in range(max_c):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1 = colors[i1]
                    c2 = colors[i2]
                    c3 = colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)

            if not valid_colors:
                return None

            # Prefer used colors to minimize distinct count
            used_valid = [c for c in valid_colors if c in used]
            if used_valid:
                # Pick randomly among used valid colors for diversity
                chosen = rng.choice(used_valid)
            else:
                # Must pick a new color; pick smallest
                chosen = min(valid_colors)

            colors[x] = chosen
            used.add(chosen)

        return colors

    def local_search(colors, max_c, seed_val=0, max_iters=5000):
        """
        Stochastic local search: randomly flip a position's color,
        accept if it reduces the number of distinct colors used while maintaining validity,
        or if it maintains validity and same color count.
        """
        rng = random.Random(seed_val)
        current = list(colors)
        current_colors = set(current)
        current_count = len(current_colors)

        best = list(current)
        best_count = current_count

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random position to flip
            x = rng.randint(0, n - 1)
            old_color = current[x]

            # Try a different color
            new_color = rng.randint(0, max_c - 1)
            if new_color == old_color:
                continue

            current[x] = new_color

            # Check validity
            if check_validity(current):
                new_count = len(set(current))
                if new_count < best_count:
                    best_count = new_count
                    best = list(current)
                # Accept if same or better count
                if new_count <= current_count:
                    current_count = new_count
                else:
                    # Reject, revert
                    current[x] = old_color
            else:
                # Invalid, revert
                current[x] = old_color

        return best, best_count

    def check_periodic_pattern(pattern, p):
        """
        Check if a periodic pattern with period p satisfies the constraint for all x in [0, 127].
        pattern is a list of length p.
        """
        for x in range(n):
            c4 = pattern[x % p]
            for i1, i2, i3 in constraints[x]:
                c1 = pattern[i1 % p]
                c2 = pattern[i2 % p]
                c3 = pattern[i3 % p]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    # Try 3 colors first with multiple random seeds
    best_result = None
    best_count = n  # worst case

    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        result = try_greedy(3, seed_val=attempt)
        if result is not None:
            cnt = count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_result = result
            # If we found 3 colors, that's likely optimal, but let's verify
            if cnt == 3:
                # Do a quick local search to see if we can confirm
                break

    # If 3 colors worked, return it
    if best_result is not None and best_count == 3:
        return best_result

    # Try 4 colors
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        result = try_greedy(4, seed_val=100 + attempt)
        if result is not None:
            cnt = count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_result = result
            if cnt == 3:
                break

    if best_result is not None and best_count == 3:
        return best_result

    # If we have a result with 4 colors, try local search to reduce to 3
    if best_result is not None and best_count == 4:
        improved, improved_count = local_search(best_result, 4, seed_val=0, max_iters=2000)
        if improved_count < best_count:
            best_count = improved_count
            best_result = improved
        if best_count == 3:
            return best_result

    # Try periodic patterns with 3 colors
    # This is a targeted search for small periods
    for p in [2, 3, 4, 6, 8, 12]:
        if time.time() > deadline - 5:
            break
        # Exhaustive search over all 3^p patterns is too big for p>4, but for small p it's fine
        # For p=2: 3^2 = 9
        # For p=3: 3^3 = 27
        # For p=4: 3^4 = 81
        # For p=6: 3^6 = 729
        # For p=8: 3^8 = 6561
        # For p=12: 3^12 = 531441 (might be too slow, but let's try with a time check)
        
        # We can generate patterns iteratively
        for i in range(3**p):
            if time.time() > deadline - 5:
                break
            pattern = []
            num = i
            for j in range(p):
                pattern.append(num % 3)
                num //= 3
            
            if check_periodic_pattern(pattern, p):
                # Found a valid periodic pattern
                colors = [pattern[x % p] for x in range(n)]
                cnt = count_colors(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = colors
                if best_count == 3:
                    return best_result

    # Fallback: use the seed program's approach (known to be valid)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break

    # Verify fallback is valid
    if check_validity(colors):
        cnt = count_colors(colors)
        if cnt < best_count or best_result is None:
            return colors

    if best_result is not None:
        return best_result

    # Should not reach here, but return something
    return colors

# EVOLVE-BLOCK-END
