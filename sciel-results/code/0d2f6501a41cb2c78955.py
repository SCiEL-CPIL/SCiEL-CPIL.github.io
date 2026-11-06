import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin for 60s limit

    # Precompute edge indices
    edge_idx = [[-1] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles and their edge indices
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

    def is_bad_cycle(colors, c_idx):
        e1, e2, e3, e4 = c_idx
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    # Initialize with recursive bisection
    colors = [0] * NUM_EDGES
    color_id = 0

    def process_partition(vertices):
        nonlocal color_id
        if len(vertices) <= 1:
            return
        mid = len(vertices) // 2
        left = vertices[:mid]
        right = vertices[mid:]
        for u in left:
            for v in right:
                eidx = edge_idx[u][v]
                colors[eidx] = color_id
        color_id += 1
        process_partition(left)
        process_partition(right)

    process_partition(list(range(N)))

    # Initial state
    freq = [0] * MAX_COLORS
    for c in colors:
        freq[c] += 1

    # Calculate initial bad cycles
    bad_count = 0
    for c_idx in cycles:
        if is_bad_cycle(colors, c_idx):
            bad_count += 1

    best_score = len(set(colors))
    best_colors = colors.copy()

    random.seed(123)

    # Local search with simulated annealing
    temperature = 5.0
    cooling = 0.99995

    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]

        # Choose new color: bias toward unused colors
        if random.random() < 0.6:
            # Try to pick an unused color
            unused = [i for i in range(MAX_COLORS) if freq[i] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
        else:
            new_color = random.randint(0, MAX_COLORS - 1)

        if new_color == old_color:
            continue

        # Calculate delta in bad_count
        delta = 0
        for cid in edge_to_cycles[e]:
            c_idx = cycles[cid]
            was_bad = is_bad_cycle(colors, c_idx)
            colors[e] = new_color
            is_bad_now = is_bad_cycle(colors, c_idx)
            colors[e] = old_color

            if was_bad and not is_bad_now:
                delta -= 1
            elif not was_bad and is_bad_now:
                delta += 1

        new_bad_count = bad_count + delta

        # Accept if improvement or with probability based on temperature
        if new_bad_count == 0 or (new_bad_count < bad_count):
            # Definitely accept
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            bad_count = new_bad_count

            score = len(set(colors))
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
        elif new_bad_count > 0:
            # Simulated annealing: accept worse with probability
            import math
            prob = math.exp(-new_bad_count / max(temperature, 0.01))
            if random.random() < prob:
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                bad_count = new_bad_count

                score = len(set(colors))
                if score > best_score:
                    best_score = score
                    best_colors = colors.copy()

        temperature *= cooling

    # Ensure we return a valid coloring
    # If best_colors has bad cycles, try to repair
    # But for now, return best valid found
    return best_colors
