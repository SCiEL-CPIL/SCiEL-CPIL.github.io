"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]

    # Precompute suffix sums for pruning in DFS
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + unit_vals[i]

    best_sum = Fraction(0)
    best_indices = []
    best_deficit = Fraction(1)

    # Greedy initial solution
    def greedy(order):
        selected = set()
        total = Fraction(0)
        for idx in order:
            v = unit_vals[idx]
            if total + v < 1:
                selected.add(idx)
                total += v
        return selected, total

    # Try multiple orderings for greedy
    random.seed(123)
    orderings = [
        list(range(n)),
        list(reversed(range(n))),
    ]
    for _ in range(20):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_sum = total
                best_indices = sorted(idxs)

    # DFS with branch and bound
    state = {
        'best_sum': best_sum,
        'best_indices': best_indices,
        'best_deficit': best_deficit,
        'time_out': False
    }

    def dfs(i, current_sum, current_indices):
        # Time check
        if time.time() > deadline:
            state['time_out'] = True
            return

        # Pruning: If current sum + max possible remaining sum <= best sum, prune
        if current_sum + suffix_sum[i] <= state['best_sum']:
            return

        # Base case
        if i == n:
            if current_sum > state['best_sum']:
                state['best_sum'] = current_sum
                state['best_indices'] = list(current_indices)
                state['best_deficit'] = Fraction(1) - current_sum
            return

        # Branch 1: Take item i
        next_sum_take = current_sum + unit_vals[i]
        if next_sum_take < 1:
            current_indices.append(i)
            dfs(i + 1, next_sum_take, current_indices)
            current_indices.pop()
            
            if state['time_out']:
                return

        # Branch 2: Skip item i
        dfs(i + 1, current_sum, current_indices)

    # Run DFS from the start
    dfs(0, Fraction(0), [])

    if state['time_out']:
        # If DFS timed out, fall back to local search refinement
        pass

    # Local search with simulated annealing for further refinement
    def local_search(start_set, max_iterations=20000):
        nonlocal state
        
        if not start_set:
            return
        
        cur = set(start_set)
        cur_total = sum(unit_vals[i] for i in cur)
        cur_deficit = Fraction(1) - cur_total
        if cur_deficit <= 0:
            return

        not_in_set = set(range(n)) - cur
        temperature = 0.01
        cooling = 0.999

        iters = 0
        while iters < max_iterations and time.time() < deadline:
            iters += 1

            # Choose a random move
            move_type = random.random()
            best_new_def = cur_deficit
            best_new_set = None
            best_new_total = None

            if move_type < 0.3 and not_in_set:
                # Add a random element
                add = random.choice(tuple(not_in_set))
                nt = cur_total + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = cur | {add}
                        best_new_total = nt
            elif move_type < 0.6 and len(cur) > 1:
                # Remove a random element
                rm = random.choice(tuple(cur))
                nt = cur_total - unit_vals[rm]
                nd = Fraction(1) - nt
                if nd < best_new_def:
                    best_new_def = nd
                    best_new_set = cur - {rm}
                    best_new_total = nt
            elif move_type < 0.9 and len(cur) > 0 and not_in_set:
                # 1-swap
                rm = random.choice(tuple(cur))
                add = random.choice(tuple(not_in_set))
                nt = cur_total - unit_vals[rm] + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = (cur - {rm}) | {add}
                        best_new_total = nt
            else:
                # 2-swap
                if len(cur) >= 2 and len(not_in_set) >= 2:
                    rm1, rm2 = random.sample(tuple(cur), 2)
                    add1, add2 = random.sample(tuple(not_in_set), 2)
                    nt = cur_total - unit_vals[rm1] - unit_vals[rm2] + unit_vals[add1] + unit_vals[add2]
                    if nt < 1:
                        nd = Fraction(1) - nt
                        if nd < best_new_def:
                            best_new_def = nd
                            best_new_set = (cur - {rm1, rm2}) | {add1, add2}
                            best_new_total = nt

            if best_new_set is not None:
                # Accept if better or with probability exp(-delta/T)
                delta = float(best_new_def - cur_deficit)
                if delta < 0:
                    # Better, always accept
                    cur = best_new_set
                    cur_total = best_new_total
                    cur_deficit = best_new_def
                    not_in_set = set(range(n)) - cur
                    if cur_deficit < state['best_deficit']:
                        state['best_deficit'] = cur_deficit
                        state['best_sum'] = cur_total
                        state['best_indices'] = sorted(cur)
                else:
                    # Worse, accept with probability
                    if random.random() < math.exp(-delta / temperature):
                        cur = best_new_set
                        cur_total = best_new_total
                        cur_deficit = best_new_def
                        not_in_set = set(range(n)) - cur
                    temperature *= cooling

        return cur_deficit

    # Run local search from the best solution found by DFS
    if state['best_indices'] and time.time() < deadline:
        local_search(set(state['best_indices']), max_iterations=50000)

    # Try a few more random restarts with local search if time permits
    for restart in range(50):
        if time.time() >= deadline:
            break
        o = list(range(n))
        random.shuffle(o)
        idxs, total = greedy(o)
        if idxs:
            deficit = Fraction(1) - total
            if deficit > 0 and deficit < state['best_deficit']:
                state['best_deficit'] = deficit
                state['best_sum'] = total
                state['best_indices'] = sorted(idxs)
            if time.time() < deadline:
                local_search(idxs, max_iterations=5000)

    # Return the best indices found
    if not state['best_indices']:
        # Fallback if no solution found (should not happen as 1/2 is valid)
        return [0]
        
    return state['best_indices']

# EVOLVE-BLOCK-END
