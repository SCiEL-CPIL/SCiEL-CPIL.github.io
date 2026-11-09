import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128

    # Precompute all nonconstant 4-term APs (0-indexed)
    aps = []
    for d in range(1, 43):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    def violations_count(colors):
        v = 0
        for a, b, c, d in aps:
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # Check if fewer than 3 distinct colors
            if ca == cb == cc == cd:
                v += 1
            elif ca == cb == cc or ca == cb == cd or ca == cc == cd or cb == cc == cd:
                v += 1
            elif (ca == cb and cc == cd) or (ca == cc and cb == cd) or (ca == cd and cb == cc):
                v += 1
        return v

    best_colors = None
    best_v = float('inf')

    for k in [3, 4]:
        for seed in range(50):
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed * 31 + k * 7)
            colors = [rng.randint(0, k - 1) for _ in range(n)]
            v = violations_count(colors)

            if v == 0:
                best_colors = colors
                best_v = 0
                break

            if v < best_v:
                best_v = v
                best_colors = colors[:]

            # Simulated annealing local search
            temp = 1.0
            for step in range(6000):
                if time.time() > deadline - 2:
                    break
                pos = rng.randint(0, n - 1)
                old = colors[pos]
                new = rng.randint(0, k - 1)
                if new == old:
                    continue
                colors[pos] = new
                new_v = violations_count(colors)
                delta = new_v - v
                if delta < 0 or (temp > 0.01 and rng.random() < 2.71828 ** (-delta / temp)):
                    v = new_v
                    if v < best_v:
                        best_v = v
                        best_colors = colors[:]
                        if v == 0:
                            break
                else:
                    colors[pos] = old
                temp *= 0.9995
                if temp < 0.01:
                    break

            if best_v == 0:
                break
        if best_v == 0:
            break

    if best_colors is not None and best_v == 0:
        return best_colors
    return best_colors if best_colors else list(range(n))
