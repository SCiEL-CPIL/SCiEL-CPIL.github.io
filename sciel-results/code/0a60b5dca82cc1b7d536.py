import random
import time

def construct():
    deadline = time.time() + 85
    n = 128

    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)

    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for i, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a - 1].append(i)
        pos_to_aps[b - 1].append(i)
        pos_to_aps[c - 1].append(i)
        pos_to_aps[d - 1].append(i)

    def count_violations(colors):
        v = 0
        for (a, b, c, d) in aps:
            if len({colors[a - 1], colors[b - 1], colors[c - 1], colors[d - 1]}) < 3:
                v += 1
        return v

    best_colors = None
    best_num_colors = 999

    for num_colors in range(3, 7):
        if time.time() > deadline:
            break

        for trial in range(50):
            if time.time() > deadline:
                break

            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            cur_viol = count_violations(colors)

            temp = 3.0
            for iteration in range(2000):
                if time.time() > deadline:
                    break

                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)

                if new_color == old_color:
                    continue

                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    a, b, c, d = aps[ap_idx]
                    old_distinct = len({colors[a - 1], colors[b - 1], colors[c - 1], colors[d - 1]})
                    old_val = colors[pos]
                    colors[pos] = new_color
                    new_distinct = len({colors[a - 1], colors[b - 1], colors[c - 1], colors[d - 1]})
                    colors[pos] = old_val
                    delta += (1 if new_distinct < 3 else 0) - (1 if old_distinct < 3 else 0)

                if delta <= 0 or random.random() < pow(2.718, -delta / max(temp, 0.01)):
                    colors[pos] = new_color
                    cur_viol += delta

                temp *= 0.9998

            if cur_viol == 0:
                used = len(set(colors))
                if used < best_num_colors:
                    best_num_colors = used
                    best_colors = colors[:]
                if best_num_colors == 3:
                    return best_colors

    if best_colors is not None:
        return best_colors
    else:
        return list(range(n))
