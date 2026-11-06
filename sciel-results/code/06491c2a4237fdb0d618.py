import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 128

    # Precompute all 4-term arithmetic progressions (nonconstant)
    # A 4-term progression: a, a+d, a+2d, a+3d where d >= 1
    # Positions are 0-indexed in our arrays, 1-indexed in the math
    # a >= 1, d >= 1, a+3d <= 128
    tps = []
    pos_to_tps = [[] for _ in range(n)]
    for a in range(1, n + 1):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 <= n:
                tp = (a - 1, a + d - 1, a + 2 * d - 1, a3 - 1)
                tp_idx = len(tps)
                tps.append(tp)
                for pos in tp:
                    pos_to_tps[pos].append(tp_idx)

    num_tps = len(tps)

    # Build constraint structure: for each position, which TPs does it appear in
    # For each TP, the other 3 positions

    # Precompute for each position: list of (tp_idx, other_positions)
    pos_constraints = [[] for _ in range(n)]
    for tp_idx, tp in enumerate(tps):
        for i in range(4):
            pos = tp[i]
            others = tuple(tp[j] for j in range(4) if j != i)
            pos_constraints[pos].append((tp_idx, others))

    # For quick distinct count computation
    def get_distinct_count(coloring, tp):
        return len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]})

    def count_bad_tps(coloring):
        bad = 0
        for tp in tps:
            if len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]}) < 3:
                bad += 1
        return bad

    def verify_valid(coloring):
        for tp in tps:
            if len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]}) < 3:
                return False
        return True

    def try_simulated_annealing(max_colors, seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = random

        coloring = [local_rng.randint(0, max_colors - 1) for _ in range(n)]

        # Initial bad count
        tp_distinct = np.array([get_distinct_count(coloring, tp) for tp in tps], dtype=np.int32)
        bad_count = int(np.sum(tp_distinct < 3))

        if bad_count == 0:
            return coloring

        temperature = 10.0
        cooling = 0.995
        best_coloring = coloring[:]
        best_bad = bad_count

        # Precompute: for each position, the set of TP indices it's in
        pos_tps_list = [list(pos_to_tps[p]) for p in range(n)]

        for iteration in range(30000):
            if time.time() > deadline:
                break

            if bad_count == 0:
                return coloring

            if bad_count < best_bad:
                best_bad = bad_count
                best_coloring = coloring[:]

            # Select position to change
            # Strategy: with high probability pick a position in a bad TP, otherwise random
            if bad_count < 100 and local_rng.random() < 0.7:
                # Pick a random bad TP, then a random position in it
                bad_indices = np.where(tp_distinct < 3)[0]
                if len(bad_indices) > 0:
                    tp_idx = bad_indices[local_rng.randint(0, len(bad_indices) - 1)]
                    tp = tps[tp_idx]
                    pos = tp[local_rng.randint(0, 3)]
                else:
                    pos = local_rng.randint(0, n - 1)
            else:
                pos = local_rng.randint(0, n - 1)

            old_color = coloring[pos]
            new_color = local_rng.randint(0, max_colors - 1)
            if new_color == old_color:
                continue

            # Calculate delta in bad_count
            delta_bad = 0
            affected = pos_tps_list[pos]

            for tp_idx in affected:
                t = tps[tp_idx]
                # Get colors of other 3 positions
                other_colors = set()
                for j in range(4):
                    if t[j] != pos:
                        other_colors.add(coloring[t[j]])

                old_set = other_colors | {old_color}
                new_set = other_colors | {new_color}

                old_distinct = len(old_set)
                new_distinct = len(new_set)

                old_bad = 1 if old_distinct < 3 else 0
                new_bad = 1 if new_distinct < 3 else 0

                delta_bad += new_bad - old_bad

            # Simulated annealing acceptance
            if delta_bad <= 0:
                accept = True
            else:
                if temperature < 0.01:
                    accept = False
                else:
                    accept = local_rng.random() < np.exp(-delta_bad / temperature)

            if accept:
                coloring[pos] = new_color
                bad_count += delta_bad

                # Update tp_distinct for affected TPs
                for tp_idx in affected:
                    t = tps[tp_idx]
                    tp_distinct[tp_idx] = get_distinct_count(coloring, t)

            temperature *= cooling

        return best_coloring if best_bad == 0 else None

    def try_csp_backtracking(max_colors, seed=None):
        """CSP with forward checking and MRV ordering."""
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = random

        # Assignments: -1 means unassigned
        assignments = [-1] * n
        # Domain for each variable: set of possible colors
        domains = [set(range(max_colors)) for _ in range(n)]

        # Track bad TPs: for each TP, track how many distinct colors are "forced"
        # Actually, let's use a different approach:
        # For each TP, track the set of assigned colors in it, and the count of unassigned positions
        # A TP is "satisfied" if it has >= 3 distinct assigned colors
        # A TP is "violated" if it has <= 2 distinct colors and all positions are assigned
        # A TP is "potentially violable" if it has <= 2 distinct assigned colors and unassigned positions remain

        # For forward checking, when we assign position p color c:
        # For each TP containing p:
        #   - Add c to the set of colors in that TP
        #   - If all 4 positions are now assigned: check if distinct count >= 3
        #   - If not all assigned: check if it's still possible to get >= 3 distinct colors
        #     (i.e., the assigned colors + remaining unassigned positions can cover >= 3 colors)

        tp_state = []  # For each TP: set of assigned colors, count of unassigned
        for tp in tps:
            tp_state.append({
                'colors': set(),
                'unassigned': 4,
                'assigned_positions': set()
            })

        def is_feasible():
            """Check if current partial assignment can lead to a valid solution."""
            for i, state in enumerate(tp_state):
                # Number of distinct colors we already have
                num_distinct = len(state['colors'])
                # Maximum additional distinct colors we can add = min(unassigned, max_colors - num_distinct)
                # But more precisely, each unassigned position can add at most 1 new color
                max_additional = min(state['unassigned'], max_colors - num_distinct)
                if num_distinct + max_additional < 3:
                    return False
            return True

        def forward_check(pos, color):
            """After assigning pos=color, check if any constraint becomes infeasible."""
            for tp_idx, others in pos_constraints[pos]:
                state = tp_state[tp_idx]
                state['colors'].add(color)
                state['unassigned'] -= 1
                state['assigned_positions'].add(pos)

                # If all positions in this TP are now assigned
                if state['unassigned'] == 0:
                    if len(state['colors']) < 3:
                        return False
                else:
                    # Check feasibility: can we still reach 3 distinct colors?
                    num_distinct = len(state['colors'])
                    max_additional = min(state['unassigned'], max_colors - num_distinct)
                    if num_distinct + max_additional < 3:
                        return False

            return True

        def undo_forward_check(pos, color):
            for tp_idx, others in pos_constraints[pos]:
                state = tp_state[tp_idx]
                state['unassigned'] += 1
                state['assigned_positions'].discard(pos)
                # We can't easily remove color from the set since other positions might have it
                # This is a simplification - we'll rebuild state if needed
                # Actually, let's just not undo and instead rebuild from scratch on backtracking
                pass

        # Better approach: use a simpler backtracking without complex undo
        def backtrack(depth):
            if time.time() > deadline:
                return False

            # Find unassigned variable with MRV (fewest remaining values)
            best_var = -1
            best_domain_size = max_colors + 1

            for i in range(n):
                if assignments[i] == -1:
                    ds = len(domains[i])
                    if ds < best_domain_size:
                        best_domain_size = ds
                        best_var = i
                        if ds <= 1:
                            break

            if best_var == -1:
                # All assigned, verify
                return verify_valid(assignments)

            if not is_feasible():
                return False

            var = best_var
            values = list(domains[var])
            local_rng.shuffle(values)

            for color in values:
                assignments[var] = color
                domains[var] = {color}

                if forward_check(var, color):
                    if backtrack(depth + 1):
                        return True

                # Undo
                assignments[var] = -1
                domains[var] = set(range(max_colors))
                # Undo forward check - this is tricky, so we rebuild tp_state
                for tp_idx, others in pos_constraints[var]:
                    state = tp_state[tp_idx]
                    state['unassigned'] += 1
                    state['assigned_positions'].discard(var)
                    # Rebuild colors from assignments
                    state['colors'] = set()
                    for p in tps[tp_idx]:
                        if assignments[p] != -1:
                            state['colors'].add(assignments[p])

            return False

        result = backtrack(0)
        if result:
            return assignments
        return None

    def try_greedy_then_local(max_colors):
        """Greedy initial assignment, then local search polish."""
        coloring = [0] * n

        # Greedy: assign positions in order, picking a color that doesn't create new violations
        for pos in range(n):
            best_color = 0
            best_score = float('inf')
            for c in range(max_colors):
                coloring[pos] = c
                # Count violations involving pos
                score = 0
                for tp_idx in pos_to_tps[pos]:
                    tp = tps[tp_idx]
                    distinct = len({coloring[tp[0]], coloring[tp[1]], coloring[tp[2]], coloring[tp[3]]})
                    if distinct < 3:
                        score += 1
                if score < best_score:
                    best_score = score
                    best_color = c
                    if score == 0:
                        break
            coloring[pos] = best_color

        # Local search polish
        tp_distinct = np.array([get_distinct_count(coloring, tp) for tp in tps], dtype=np.int32)
        bad_count = int(np.sum(tp_distinct < 3))

        if bad_count == 0:
            return coloring

        temperature = 5.0
        cooling = 0.99
        pos_tps_list = [list(pos_to_tps[p]) for p in range(n)]

        for iteration in range(50000):
            if time.time() > deadline:
                break
            if bad_count == 0:
                return coloring

            if bad_count < 50:
                bad_indices = np.where(tp_distinct < 3)[0]
                if len(bad_indices) > 0:
                    tp_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                    tp = tps[tp_idx]
                    pos = tp[random.randint(0, 3)]
                else:
                    pos = random.randint(0, n - 1)
            else:
                pos = random.randint(0, n - 1)

            old_color = coloring[pos]
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue

            delta_bad = 0
            for tp_idx in pos_tps_list[pos]:
                t = tps[tp_idx]
                other_colors = set()
                for j in range(4):
                    if t[j] != pos:
                        other_colors.add(coloring[t[j]])
                old_distinct = len(other_colors | {old_color})
                new_distinct = len(other_colors | {new_color})
                delta_bad += (1 if new_distinct < 3 else 0) - (1 if old_distinct < 3 else 0)

            if delta_bad <= 0 or random.random() < np.exp(-delta_bad / temperature):
                coloring[pos] = new_color
                bad_count += delta_bad
                for tp_idx in pos_tps_list[pos]:
                    t = tps[tp_idx]
                    tp_distinct[tp_idx] = get_distinct_count(coloring, t)

            temperature *= cooling

        return coloring if bad_count == 0 else None

    # Strategy: try different approaches for increasing k
    best_result = None

    for k in range(3, 128):
        if time.time() > deadline:
            break

        found = False

        # Approach 1: Simulated annealing with multiple restarts
        num_sa_restarts = 30 if k <= 5 else 15
        for attempt in range(num_sa_restarts):
            if time.time() > deadline:
                break
            result = try_simulated_annealing(k, seed=attempt * 1000 + k)
            if result is not None and verify_valid(result):
                best_result = result
                found = True
                break

        if found:
            break

        # Approach 2: CSP backtracking (for small k)
        if k <= 5:
            for attempt in range(5):
                if time.time() > deadline:
                    break
                result = try_csp_backtracking(k, seed=attempt * 2000 + k)
                if result is not None and verify_valid(result):
                    best_result = result
                    found = True
                    break

        if found:
            break

        # Approach 3: Greedy + local search
        if k <= 10:
            for attempt in range(10):
                if time.time() > deadline:
                    break
                result = try_greedy_then_local(k)
                if result is not None and verify_valid(result):
                    best_result = result
                    found = True
                    break

        if found:
            break

    if best_result is not None:
        return best_result

    # Fallback: trivially valid coloring using all 128 colors
    return list(range(n))
