import time
import random
import itertools
import math

def construct():
    deadline = time.time() + 148  # Safety margin for 150s limit

    n = 16
    vertices = list(range(n))

    # Precompute edges
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list)  # 120

    # Precompute K4s
    k4s = []
    k4_edges = []
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        k4s.append(combo)
        k4_edges.append([edge_index[e] for e in edges])
    num_k4s = len(k4_edges)  # 1820

    # Precompute edge -> k4s mapping
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)

    MAX_COLORS = 20

    def init_state(colors):
        """Initialize k4_counts, k4_distinct, total_violations, global_counts, colors_used_count."""
        k4_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        global_counts = [0] * MAX_COLORS
        colors_used_count = 0

        for e in range(num_edges):
            c = colors[e]
            if c < 0 or c >= MAX_COLORS:
                c = c % MAX_COLORS
            global_counts[c] += 1

        for c in range(MAX_COLORS):
            if global_counts[c] > 0:
                colors_used_count += 1

        for k4_idx in range(num_k4s):
            for e in k4_edges[k4_idx]:
                c = colors[e]
                if c < 0 or c >= MAX_COLORS:
                    c = c % MAX_COLORS
                k4_counts[k4_idx][c] += 1

            distinct = 0
            for c in range(MAX_COLORS):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct

        total_violations = 0
        for k4_idx in range(num_k4s):
            if k4_distinct[k4_idx] < 5:
                total_violations += 1

        return k4_counts, k4_distinct, total_violations, global_counts, colors_used_count

    def get_cost(total_violations, colors_used_count):
        return total_violations * 1000 + colors_used_count

    def random_coloring():
        return [random.randint(0, MAX_COLORS - 1) for _ in range(num_edges)]

    def seed_coloring():
        # (i+j) % 17
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 17)
        return [c % MAX_COLORS for c in colors]

    def verify(colors):
        """Verify a coloring is valid and return (valid, num_colors_used)."""
        used = set()
        for c in colors:
            if 0 <= c < MAX_COLORS:
                used.add(c)
        for combo in k4s:
            a, b, c, d = combo
            edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            col_set = set()
            for e in edges:
                ci = edge_index[e]
                col_set.add(colors[ci])
            if len(col_set) < 5:
                return False, len(used)
        return True, len(used)

    best_score = float('inf')
    best_colors = None

    # Try seed coloring first
    seed = seed_coloring()
    valid, nc = verify(seed)
    if valid:
        best_score = nc
        best_colors = seed[:]

    # Simulated annealing with multiple restarts
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            colors = seed[:]
        else:
            colors = random_coloring()

        k4_counts, k4_distinct, total_violations, global_counts, colors_used_count = init_state(colors)
        current_cost = get_cost(total_violations, colors_used_count)

        if total_violations == 0 and colors_used_count < best_score:
            best_score = colors_used_count
            best_colors = colors[:]

        T = 10.0
        T_min = 0.05
        cooling_rate = 0.99995
        max_iters = 300000

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Pick random edge
            e = random.randint(0, num_edges - 1)
            c_old = colors[e]

            # Pick new color - bias towards used colors
            if random.random() < 0.85:
                used = [c for c in range(MAX_COLORS) if global_counts[c] > 0]
                if used:
                    c_new = random.choice(used)
                else:
                    c_new = random.randint(0, MAX_COLORS - 1)
            else:
                c_new = random.randint(0, MAX_COLORS - 1)

            if c_new == c_old:
                continue

            # Calculate delta
            delta_viol = 0
            affected_k4s = edge_k4s[e]

            for k4_idx in affected_k4s:
                old_d = k4_distinct[k4_idx]

                cnt = k4_counts[k4_idx]
                d = old_d

                # Remove c_old
                cnt[c_old] -= 1
                if cnt[c_old] == 0:
                    d -= 1

                # Add c_new
                cnt[c_new] += 1
                if cnt[c_new] == 1:
                    d += 1

                k4_distinct[k4_idx] = d

                old_viol = 1 if old_d < 5 else 0
                new_viol = 1 if d < 5 else 0
                delta_viol += new_viol - old_viol

            # Update global counts
            gc_old = global_counts[c_old]
            gc_new = global_counts[c_new]

            global_counts[c_old] -= 1
            global_counts[c_new] += 1

            delta_color_count = 0
            if gc_old == 1:
                delta_color_count -= 1
            if gc_new == 0:
                delta_color_count += 1

            total_violations += delta_viol
            colors_used_count += delta_color_count

            colors[e] = c_new

            new_cost = get_cost(total_violations, colors_used_count)
            delta_cost = new_cost - current_cost

            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                if total_violations == 0 and colors_used_count < best_score:
                    best_score = colors_used_count
                    best_colors = colors[:]
            else:
                # Rollback
                colors[e] = c_old

                for k4_idx in affected_k4s:
                    cnt = k4_counts[k4_idx]
                    d = k4_distinct[k4_idx]

                    # Undo add c_new
                    cnt[c_new] -= 1
                    if cnt[c_new] == 0:
                        d -= 1

                    # Undo remove c_old
                    cnt[c_old] += 1
                    if cnt[c_old] == 1:
                        d += 1

                    k4_distinct[k4_idx] = d

                global_counts[c_old] += 1
                global_counts[c_new] -= 1
                total_violations -= delta_viol
                colors_used_count -= delta_color_count

            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Final verification
    if best_colors is not None:
        valid, nc = verify(best_colors)
        if valid:
            return best_colors
        else:
            # Fallback
            return seed

    return seed
