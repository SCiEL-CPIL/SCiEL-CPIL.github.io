import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles and edge-to-cycles mapping
CYCLES = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1, e2, e3, e4 = edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)
                ci = len(CYCLES)
                CYCLES.append((e1, e2, e3, e4))
                for e in (e1, e2, e3, e4):
                    edge_to_cycles[e].append(ci)
                # Cycle 2: a-b-d-c-a
                e1, e2, e3, e4 = edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)
                ci = len(CYCLES)
                CYCLES.append((e1, e2, e3, e4))
                for e in (e1, e2, e3, e4):
                    edge_to_cycles[e].append(ci)
                # Cycle 3: a-c-b-d-a
                e1, e2, e3, e4 = edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)
                ci = len(CYCLES)
                CYCLES.append((e1, e2, e3, e4))
                for e in (e1, e2, e3, e4):
                    edge_to_cycles[e].append(ci)

def is_valid_full(colors):
    for c in CYCLES:
        c0 = colors[c[0]]
        c1 = colors[c[1]]
        c2 = colors[c[2]]
        c3 = colors[c[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def check_local(colors, edge):
    for ci in edge_to_cycles[edge]:
        c = CYCLES[ci]
        c0 = colors[c[0]]
        c1 = colors[c[1]]
        c2 = colors[c[2]]
        c3 = colors[c[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Initial: color(i,j) = i, uses 15 colors
    current = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            current[idx] = i
            idx += 1

    cur_score = count_colors(current)  # 15
    best = current[:]
    best_score = cur_score

    T = 5.0
    T_min = 0.05
    alpha = 0.9995

    # Color usage tracking
    color_counts = [0] * 120
    for c in current:
        color_counts[c] += 1

    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current[e]

        # Propose new color
        r = random.random()
        if r < 0.6:
            # Copy from a random other edge
            e2 = random.randint(0, NUM_EDGES - 1)
            if e2 == e:
                e2 = (e + 1) % NUM_EDGES
            new_color = current[e2]
        elif r < 0.85:
            # Pick from colors 0..14 (the "base" palette)
            new_color = random.randint(0, 14)
        else:
            # Try a higher color to potentially increase diversity
            new_color = random.randint(15, 119)

        if new_color == old_color:
            continue

        # Apply
        current[e] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1

        # Local validity check
        if check_local(current, e):
            new_score = count_colors(current)
            delta = new_score - cur_score

            if new_score > best_score:
                best_score = new_score
                best = current[:]

            # Accept: always if improving, else with SA probability
            if delta >= 0:
                cur_score = new_score
            else:
                if random.random() < math.exp(delta / T):
                    cur_score = new_score
                else:
                    # Revert
                    current[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
        else:
            # Revert
            current[e] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1

        T *= alpha
        if T < T_min:
            T = T_min

    # Final check
    if is_valid_full(best):
        return best
    else:
        safe = [0] * NUM_EDGES
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                safe[idx] = i
                idx += 1
        return safe
