import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128

    def backtrack(colors, pos, num_colors, rng, deadline):
        if pos > n:
            return True
        if time.time() > deadline:
            return False

        color_order = list(range(num_colors))
        rng.shuffle(color_order)

        for color in color_order:
            colors[pos - 1] = color
            valid = True

            # pos as 4th term
            for d in range(1, (pos + 2) // 3):
                p1 = pos - 3 * d
                if p1 < 1:
                    break
                if len({colors[p1-1], colors[p1+d-1], colors[p1+2*d-1], color}) < 3:
                    valid = False
                    break

            if valid:
                # pos as 3rd term, only if 4th is assigned
                for d in range(1, (pos + 1) // 2):
                    p1 = pos - 2 * d
                    p4 = pos + d
                    if p1 < 1 or p4 > n:
                        break
                    if colors[p4-1] != -1:
                        if len({colors[p1-1], colors[p1+d-1], color, colors[p4-1]}) < 3:
                            valid = False
                            break

            if valid:
                # pos as 2nd term, only if 1st and 4th are assigned
                for d in range(1, pos):
                    p1 = pos - d
                    p4 = pos + 2 * d
                    if p4 > n:
                        break
                    if colors[p1-1] != -1 and colors[p4-1] != -1:
                        if len({colors[p1-1], color, colors[p1+d-1], colors[p4-1]}) < 3:
                            valid = False
                            break

            if valid:
                # pos as 1st term, only if 4th is assigned
                for d in range(1, n - pos + 1):
                    p4 = pos + 3 * d
                    if p4 > n:
                        break
                    if colors[p4-1] != -1:
                        if len({color, colors[pos+d-1], colors[pos+2*d-1], colors[p4-1]}) < 3:
                            valid = False
                            break

            if valid:
                if backtrack(colors, pos + 1, num_colors, rng, deadline):
                    return True

        colors[pos - 1] = -1
        return False

    for attempt in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(attempt * 31 + 7)
        colors = [-1] * n
        if backtrack(colors, 1, 3, rng, deadline):
            return colors

    for attempt in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(100000 + attempt * 17)
        colors = [-1] * n
        if backtrack(colors, 1, 4, rng, deadline):
            return colors

    # Fallback greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                if len({colors[a-1], colors[a+d-1], colors[a+2*d-1], color}) < 3:
                    ok = False
                    break
            if ok:
                colors.append(color)
                break
    return colors
