import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices for all pairs (i, j) with 0 <= i < j < N
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    num_cycles = len(cycles)

    # Map edge index to list of cycle indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # Convert to numpy for vectorized operations
    cycles_np = np.array(cycles, dtype=np.int32)

    def get_seed_colors():
        """Recursive partition seed coloring."""
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
        return colors

    def check_validity_np(colors_np):
        """Check if coloring is valid using numpy."""
        cycle_colors = colors_np[cycles_np]  # shape (num_cycles, 4)
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        all_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return not np.any(all_distinct)

    def is_valid_change(colors_list, e, new_color):
        """Check if assigning new_color to edge e creates a rainbow 4-cycle."""
        for c_idx in edge_to_cycles[e]:
            c = cycles[c_idx]
            cols = []
            for ce in c:
                if ce == e:
                    cols.append(new_color)
                else:
                    cols.append(colors_list[ce])
            if len(set(cols)) == 4:
                return False
        return True

    def count_distinct(colors):
        return len(set(colors))

    def mrv_backtracking(colors, freq_arr, max_time=20.0):
        """
        MRV-guided backtracking to maximize color diversity.
        For each edge, try to assign a new unused color that doesn't violate constraints.
        """
        colors = list(colors)
        freq_arr = list(freq_arr)
        score = sum(1 for f in freq_arr if f > 0)
        best_local = score
        best_local_colors = colors.copy()

        # Edge ordering: sort by number of cycles (fewer cycles = easier to assign new color)
        edge_order = sorted(range(NUM_EDGES), key=lambda e: len(edge_to_cycles[e]))
        random.shuffle(edge_order)

        for e in edge_order:
            if time.time() - start_time > max_time:
                break

            old_color = colors[e]

            # Get unused colors
            unused = [c for c in range(MAX_COLORS) if freq_arr[c] == 0]
            if not unused:
                break

            # Try each unused color (randomized)
            random.shuffle(unused)
            found = False
            for new_c in unused:
                if is_valid_change(colors, e, new_c):
                    # Accept this assignment
                    freq_arr[old_color] -= 1
                    freq_arr[new_c] += 1
                    colors[e] = new_c
                    score += 1

                    if score > best_local:
                        best_local = score
                        best_local_colors = colors.copy()
                    found = True
                    break

        return best_local_colors, best_local, freq_arr

    def local_search_polish(colors, freq_arr, time_budget=15.0):
        """Local search with simulated annealing to refine the coloring."""
        colors = list(colors)
        freq_arr = list(freq_arr)
        current_score = sum(1 for f in freq_arr if f > 0)
        best_score = current_score
        best_colors = colors.copy()

        temperature = 2.0
        cooling_rate = 0.9999
        min_temperature = 0.05
        no_improve_steps = 0
        max_no_improve = 2000

        end_time = time.time() + time_budget

        while time.time() < end_time:
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]

            # Smart color selection: bias toward unused or rare colors
            if random.random() < 0.65 and current_score < MAX_COLORS:
                unused = [c for c in range(MAX_COLORS) if freq_arr[c] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    # Pick from rare colors
                    weights = [1.0 / (freq_arr[c] + 1) for c in range(MAX_COLORS)]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    new_color = MAX_COLORS - 1
                    for c in range(MAX_COLORS):
                        cumulative += weights[c]
                        if r <= cumulative:
                            new_color = c
                            break
            else:
                new_color = random.randint(0, MAX_COLORS - 1)

            if new_color == old_color:
                continue

            if is_valid_change(colors, e, new_color):
                old_freq = freq_arr[old_color]
                new_freq = freq_arr[new_color]

                score_change = 0
                if old_freq == 1:
                    score_change -= 1
                if new_freq == 0:
                    score_change += 1

                new_score = current_score + score_change

                if new_score >= current_score:
                    accept = True
                else:
                    delta = current_score - new_score
                    accept = random.random() < np.exp(-delta / temperature)

                if accept:
                    freq_arr[old_color] -= 1
                    freq_arr[new_color] += 1
                    colors[e] = new_color
                    current_score = new_score

                    if current_score > best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                        no_improve_steps = 0
                        temperature = 2.0
                    else:
                        no_improve_steps += 1

                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature

                    if no_improve_steps > max_no_improve:
                        # Reset to best and perturb
                        colors = best_colors.copy()
                        freq_arr = [0] * MAX_COLORS
                        for c in colors:
                            freq_arr[c] += 1
                        current_score = best_score
                        temperature = 2.0
                        no_improve_steps = 0

                        # Perturb a few edges
                        for _ in range(10):
                            pe = random.randint(0, NUM_EDGES - 1)
                            pc = random.randint(0, MAX_COLORS - 1)
                            if pc != colors[pe] and is_valid_change(colors, pe, pc):
                                old_pc = colors[pe]
                                freq_arr[old_pc] -= 1
                                freq_arr[pc] += 1
                                colors[pe] = pc
                                if freq_arr[old_pc] == 0:
                                    current_score -= 1
                                if freq_arr[pc] == 1:
                                    current_score += 1
                                break

        return best_colors, best_score, freq_arr

    # Initialize with seed coloring
    random.seed(42)
    np.random.seed(42)

    current_colors = get_seed_colors()
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1

    best_score = count_distinct(current_colors)
    best_colors = current_colors.copy()

    # Phase 1: MRV backtracking (multiple restarts)
    bt_budget = 25.0
    bt_start = time.time()
    while time.time() - bt_start < bt_budget:
        bt_colors, bt_score, bt_freq = mrv_backtracking(
            current_colors, freq, max_time=min(15.0, bt_budget - (time.time() - bt_start))
        )

        if bt_score > best_score:
            best_score = bt_score
            best_colors = bt_colors
            current_colors = bt_colors
            freq = bt_freq

        if best_score >= MAX_COLORS:
            break

        # If stuck, perturb and retry
        if time.time() - bt_start > bt_budget * 0.7:
            # Add some random perturbation to escape local optimum
            for _ in range(random.randint(3, 8)):
                e = random.randint(0, NUM_EDGES - 1)
                new_c = random.randint(0, MAX_COLORS - 1)
                if new_c != current_colors[e] and is_valid_change(current_colors, e, new_c):
                    old_c = current_colors[e]
                    freq[old_c] -= 1
                    freq[new_c] += 1
                    current_colors[e] = new_c

    # Phase 2: Local search polish with remaining time
    remaining_time = deadline - time.time() - 2.0  # 2s safety margin
    if remaining_time > 5.0:
        # Run local search from best solution
        ls_colors, ls_score, ls_freq = local_search_polish(
            best_colors, freq, time_budget=remaining_time
        )

        if ls_score > best_score:
            best_score = ls_score
            best_colors = ls_colors

    # Final verification
    best_colors_np = np.array(best_colors, dtype=np.int32)
    if not check_validity_np(best_colors_np):
        # Fallback to seed if somehow invalid
        best_colors = get_seed_colors()

    return best_colors
