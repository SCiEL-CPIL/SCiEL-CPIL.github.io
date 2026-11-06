import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 82  # Safety margin before 90s kill


def construct():
    start_time = time.time()

    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1

    # Enumerate all 4-sets and precompute the three 4-cycles for each
    all_4sets = list(itertools.combinations(range(N), 4))

    def get_cycles(a, b, c, d):
        e1 = (edge_index[(a, b)], edge_index[(b, c)], edge_index[(c, d)], edge_index[(d, a)])
        e2 = (edge_index[(a, b)], edge_index[(b, d)], edge_index[(d, c)], edge_index[(c, a)])
        e3 = (edge_index[(a, c)], edge_index[(c, b)], edge_index[(b, d)], edge_index[(d, a)])
        return (e1, e2, e3)

    # For each edge, store the list of (4set_index, cycle_index) pairs
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))

    total_cycles = len(all_4sets) * 3

    def is_valid_cycle(colors, cycle):
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True

    def full_validation(colors):
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True

    def count_violations(colors):
        count = 0
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    count += 1
        return count

    def count_distinct(colors):
        return len(set(colors))

    # Base valid coloring: color(i,j) = i
    base_colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        base_colors[idx] = i

    if not full_validation(base_colors):
        return [0] * NUM_EDGES

    best_colors = base_colors[:]
    best_num_colors = len(set(best_colors))

    # Precompute sorted edges by constraint count (fewer constraints = easier to recolor)
    edge_constraint_counts = [(len(edge_to_4sets[e]), e) for e in range(NUM_EDGES)]
    edge_constraint_counts.sort()
    sorted_edges_by_constraint = [e for _, e in edge_constraint_counts]

    def greedy_add_colors(colors, max_new_color=119):
        """Try to add new colors one at a time via greedy edge recoloring."""
        used_colors = set(colors)
        new_colors = [c for c in range(15, max_new_color + 1) if c not in used_colors]

        for new_color in new_colors:
            if time.time() - start_time > DEADLINE:
                break

            found = False
            # Try edges in order of fewest constraints first
            for e_idx in sorted_edges_by_constraint:
                if time.time() - start_time > DEADLINE:
                    break

                old_color = colors[e_idx]
                colors[e_idx] = new_color

                valid = True
                for (s_idx, c_idx) in edge_to_4sets[e_idx]:
                    cycle = cycles_by_4set[s_idx][c_idx]
                    if not is_valid_cycle(colors, cycle):
                        valid = False
                        break

                if valid:
                    found = True
                    used_colors.add(new_color)
                    break

                colors[e_idx] = old_color

            if not found:
                # Try reverse order
                for e_idx in reversed(sorted_edges_by_constraint):
                    if time.time() - start_time > DEADLINE:
                        break

                    old_color = colors[e_idx]
                    colors[e_idx] = new_color

                    valid = True
                    for (s_idx, c_idx) in edge_to_4sets[e_idx]:
                        cycle = cycles_by_4set[s_idx][c_idx]
                        if not is_valid_cycle(colors, cycle):
                            valid = False
                            break

                    if valid:
                        found = True
                        used_colors.add(new_color)
                        break

                    colors[e_idx] = old_color

        return colors, len(used_colors)

    def simulated_annealing(initial_colors, iters, T_start, T_end):
        """Run SA to maximize distinct colors while keeping zero violations."""
        current_colors = initial_colors[:]
        current_violations = count_violations(current_colors)
        current_distinct = count_distinct(current_colors)

        best_sa_colors = current_colors[:]
        best_sa_distinct = current_distinct
        best_sa_violations = current_violations

        for it in range(iters):
            if time.time() - start_time > DEADLINE:
                break

            progress = it / iters
            T = T_start * (T_end / T_start) ** progress

            e_idx = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e_idx]

            if random.random() < 0.25:
                other_e = random.randint(0, NUM_EDGES - 1)
                new_color = current_colors[other_e]
            else:
                new_color = random.randint(0, 119)

            if new_color == old_color:
                continue

            # Compute delta violations
            delta_viol = 0
            for (s_idx, c_idx) in edge_to_4sets[e_idx]:
                cycle = cycles_by_4set[s_idx][c_idx]
                old_valid = is_valid_cycle(current_colors, cycle)
                current_colors[e_idx] = new_color
                new_valid = is_valid_cycle(current_colors, cycle)
                current_colors[e_idx] = old_color
                if old_valid and not new_valid:
                    delta_viol += 1
                elif not old_valid and new_valid:
                    delta_viol -= 1

            # Compute delta distinct
            old_distinct = current_distinct
            current_colors[e_idx] = new_color
            new_distinct = count_distinct(current_colors)
            current_colors[e_idx] = old_color
            delta_distinct = new_distinct - old_distinct

            delta_obj = delta_viol * 100 - delta_distinct

            if delta_obj < 0:
                accept = True
            else:
                accept = random.random() < (2.718281828 ** (-delta_obj / T))

            if accept:
                current_colors[e_idx] = new_color
                current_violations += delta_viol
                current_distinct = new_distinct

                if current_violations == 0 and current_distinct > best_sa_distinct:
                    best_sa_distinct = current_distinct
                    best_sa_colors = current_colors[:]
                    best_sa_violations = 0
                elif current_violations < best_sa_violations:
                    best_sa_violations = current_violations
                    best_sa_colors = current_colors[:]
                    best_sa_distinct = current_distinct

        return best_sa_colors, best_sa_distinct, best_sa_violations

    def repair_violations(colors, max_iters=5000):
        """Try to fix violations by matching colors within violating cycles."""
        repaired = colors[:]
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            viol = count_violations(repaired)
            if viol == 0:
                break
            found_fix = False
            for s_idx in range(len(all_4sets)):
                for c_idx in range(3):
                    cycle = cycles_by_4set[s_idx][c_idx]
                    if not is_valid_cycle(repaired, cycle):
                        e_to_change = random.choice(cycle)
                        old_c = repaired[e_to_change]
                        other_colors = [repaired[e] for e in cycle if e != e_to_change]
                        new_c = random.choice(other_colors)
                        repaired[e_to_change] = new_c
                        found_fix = True
                        break
                if found_fix:
                    break
            if not found_fix:
                break
        return repaired

    # Multi-start SA with different perturbations
    num_starts = 8
    iters_per_start = 80000
    T_start = 10.0
    T_end = 0.01

    for start_idx in range(num_starts):
        if time.time() - start_time > DEADLINE:
            break

        # Create a perturbed starting point
        if start_idx == 0:
            start_colors = base_colors[:]
        else:
            start_colors = base_colors[:]
            # Randomly perturb some edges
            num_perturb = random.randint(3, 10)
            for _ in range(num_perturb):
                e_idx = random.randint(0, NUM_EDGES - 1)
                start_colors[e_idx] = random.randint(0, 119)

        sa_colors, sa_distinct, sa_viol = simulated_annealing(
            start_colors, iters_per_start, T_start, T_end
        )

        # If SA found a valid coloring, try to improve with greedy
        if sa_viol == 0:
            greedy_colors, greedy_distinct = greedy_add_colors(sa_colors[:])
            if greedy_distinct > best_num_colors:
                best_num_colors = greedy_distinct
                best_colors = greedy_colors[:]
        else:
            # Try to repair
            repaired = repair_violations(sa_colors[:])
            if count_violations(repaired) == 0:
                greedy_colors, greedy_distinct = greedy_add_colors(repaired[:])
                if greedy_distinct > best_num_colors:
                    best_num_colors = greedy_distinct
                    best_colors = greedy_colors[:]

    # Final greedy pass on the best solution
    if time.time() - start_time < DEADLINE:
        final_colors, final_distinct = greedy_add_colors(best_colors[:])
        if final_distinct > best_num_colors:
            best_num_colors = final_distinct
            best_colors = final_colors[:]

    # Final validation
    if full_validation(best_colors):
        return best_colors
    else:
        # Fallback
        return base_colors
