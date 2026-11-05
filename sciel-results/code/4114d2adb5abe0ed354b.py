import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict graph using bitsets for speed
    # conflict_mask[v] is a bitset where bit u is set if v conflicts with u
    conflict_mask = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = 0
            a, b = i, j
            while b:
                a, b = b, a % b
            g = a
            u_prime = i // g
            v_prime = j // g
            if g % (u_prime + v_prime) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Bitset for all vertices
    all_mask = 0
    for v in vertices:
        all_mask |= (1 << v)

    # Build list of conflict pairs (clauses) for WalkSAT
    clauses = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (conflict_mask[i] >> j) & 1:
                clauses.append((i, j))

    def can_add(v, s_mask):
        """Check if vertex v can be added to set represented by s_mask."""
        return (conflict_mask[v] & s_mask) == 0

    def mask_to_set(mask):
        return {v for v in vertices if (mask >> v) & 1}

    def greedy_by_order(order, s_mask=0):
        result_mask = s_mask
        for v in order:
            if (result_mask >> v) & 1:
                continue
            if can_add(v, result_mask):
                result_mask |= (1 << v)
        return result_mask

    # Min-degree greedy: repeatedly pick the vertex with fewest conflicts
    # among remaining vertices, then remove it and its neighbors.
    def greedy_min_degree():
        rem = all_mask
        result_mask = 0
        while rem:
            # Find vertex with minimum degree in the induced subgraph on rem
            best_v = -1
            best_deg = N + 1
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length()
                deg = (conflict_mask[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v < 0:
                break
            result_mask |= (1 << best_v)
            rem &= ~(1 << best_v)
            rem &= ~conflict_mask[best_v]
        return result_mask

    def greedy_max_degree():
        rem = all_mask
        result_mask = 0
        while rem:
            best_v = -1
            best_deg = -1
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length()
                deg = (conflict_mask[v] & rem).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v < 0:
                break
            result_mask |= (1 << best_v)
            rem &= ~(1 << best_v)
            rem &= ~conflict_mask[best_v]
        return result_mask

    def greedy_randomized(top_k=3):
        """Randomized min-degree greedy: pick from top-k lowest-degree vertices."""
        rem = all_mask
        result_mask = 0
        while rem:
            temp = rem
            deg_list = []
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length()
                deg = (conflict_mask[v] & rem).bit_count()
                deg_list.append((deg, v))
                temp ^= lsb
            deg_list.sort()
            k = min(top_k, len(deg_list))
            if k == 0:
                break
            chosen = deg_list[random.randrange(k)][1]
            result_mask |= (1 << chosen)
            rem &= ~(1 << chosen)
            rem &= ~conflict_mask[chosen]
        return result_mask

    # WalkSAT-style stochastic local search for Max-SAT.
    # Variables: x_v (v in 1..N), True means v is selected.
    # Hard clauses: (¬x_i ∨ ¬x_j) for each conflict pair (i,j).
    # Soft objective: maximize number of True variables (unit clauses x_v).
    #
    # We maintain a bitset s_mask of selected vertices. A hard clause
    # (i,j) is violated if both i and j are selected.
    #
    # At each step, if there is a violated clause, pick one at random
    # (weighted by clause weight), then flip one of its two variables.
    # Flipping a selected variable to False reduces size by 1;
    # flipping an unselected variable to True increases size by 1
    # but may violate new clauses.
    #
    # To escape local optima, we also allow "restarts" with random
    # perturbations, and we track clause weights that increase for
    # frequently violated clauses.

    def walksat_search(s_mask, time_limit, p_flip_selected=0.5):
        """
        WalkSAT-style local search.
        s_mask: initial bitset of selected vertices.
        Returns improved s_mask.
        """
        clause_weights = [1.0] * len(clauses)
        best_mask = s_mask
        best_size = bin(s_mask).count('1')
        no_improve_steps = 0
        max_no_improve = 200

        while time.time() < time_limit:
            # Find violated clauses (both endpoints selected)
            violated = []
            for idx, (i, j) in enumerate(clauses):
                if (s_mask >> i) & 1 and (s_mask >> j) & 1:
                    violated.append(idx)

            if not violated:
                # No violated clauses: try to add more vertices
                improved = False
                for v in vertices:
                    if (s_mask >> v) & 1:
                        continue
                    if can_add(v, s_mask):
                        s_mask |= (1 << v)
                        improved = True
                if improved:
                    sz = bin(s_mask).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_mask = s_mask
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    if no_improve_steps > max_no_improve:
                        break
                continue

            # Pick a violated clause weighted by clause_weights
            total_w = sum(clause_weights[idx] for idx in violated)
            r = random.random() * total_w
            cumulative = 0.0
            chosen_idx = violated[0]
            for idx in violated:
                cumulative += clause_weights[idx]
                if r <= cumulative:
                    chosen_idx = idx
                    break

            i, j = clauses[chosen_idx]

            # Increase weight of this clause
            clause_weights[chosen_idx] *= 1.05
            # Decay other weights slightly to prevent overflow
            for k in range(len(clause_weights)):
                if k != chosen_idx:
                    clause_weights[k] *= 0.999

            # Decide which variable to flip
            # If we flip a selected one to False, size decreases by 1
            # If we flip an unselected one to True, size increases by 1
            # but may create new violations.
            i_selected = (s_mask >> i) & 1
            j_selected = (s_mask >> j) & 1

            if i_selected and j_selected:
                # Both selected: flip one to False
                # Choose the one whose removal creates fewer new "lost"
                # opportunities, or randomly.
                if random.random() < p_flip_selected:
                    v_flip = i
                else:
                    v_flip = j
                s_mask &= ~(1 << v_flip)
            elif i_selected:
                # i selected, j not: flip j to True (may help)
                # or flip i to False
                if random.random() < 0.3:
                    s_mask &= ~(1 << i)
                else:
                    if can_add(j, s_mask & ~(1 << j)):
                        # Adding j doesn't conflict with others
                        s_mask |= (1 << j)
                    else:
                        s_mask &= ~(1 << i)
            elif j_selected:
                if random.random() < 0.3:
                    s_mask &= ~(1 << j)
                else:
                    if can_add(i, s_mask & ~(1 << i)):
                        s_mask |= (1 << i)
                    else:
                        s_mask &= ~(1 << j)
            else:
                # Neither selected: flip one to True
                if random.random() < 0.5:
                    if can_add(i, s_mask):
                        s_mask |= (1 << i)
                    elif can_add(j, s_mask):
                        s_mask |= (1 << j)
                else:
                    if can_add(j, s_mask):
                        s_mask |= (1 << j)
                    elif can_add(i, s_mask):
                        s_mask |= (1 << i)

            sz = bin(s_mask).count('1')
            if sz > best_size:
                best_size = sz
                best_mask = s_mask
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps > max_no_improve:
                    break

        return best_mask

    # 2-for-3 swap: remove two vertices, add three non-conflicting ones.
    def swap_2_for_3(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s_list = [v for v in vertices if (s_mask >> v) & 1]
            if len(s_list) < 2:
                break
            random.shuffle(s_list)
            for u1 in s_list[:min(len(s_list), 30)]:
                if time.time() >= time_limit:
                    break
                for u2 in s_list[:min(len(s_list), 30)]:
                    if u2 <= u1:
                        continue
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u1) & ~(1 << u2)
                    # Find all addable vertices
                    addable = []
                    for v in vertices:
                        if (s_mask >> v) & 1:
                            continue
                        if can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) < 3:
                        continue
                    random.shuffle(addable)
                    limit = min(len(addable), 25)
                    found = False
                    for i in range(limit):
                        for j in range(i + 1, limit):
                            for k in range(j + 1, limit):
                                v1, v2, v3 = addable[i], addable[j], addable[k]
                                if ((conflict_mask[v1] >> v2) & 1) == 0 and \
                                   ((conflict_mask[v1] >> v3) & 1) == 0 and \
                                   ((conflict_mask[v2] >> v3) & 1) == 0:
                                    s_mask = temp_mask | (1 << v1) | (1 << v2) | (1 << v3)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
        return s_mask

    # 1-for-2 swap: remove one vertex, add two non-conflicting ones.
    def swap_1_for_2(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s_list = [v for v in vertices if (s_mask >> v) & 1]
            random.shuffle(s_list)
            for u in s_list[:min(len(s_list), 40)]:
                if time.time() >= time_limit:
                    break
                temp_mask = s_mask & ~(1 << u)
                addable = []
                for v in vertices:
                    if (s_mask >> v) & 1:
                        continue
                    if v == u:
                        continue
                    if can_add(v, temp_mask):
                        addable.append(v)
                if len(addable) < 2:
                    continue
                random.shuffle(addable)
                limit = min(len(addable), 30)
                found = False
                for i in range(limit):
                    for j in range(i + 1, limit):
                        v1, v2 = addable[i], addable[j]
                        if (conflict_mask[v1] >> v2) & 1 == 0:
                            s_mask = temp_mask | (1 << v1) | (1 << v2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return s_mask

    # Greedy construction
    best_mask = 0

    # Try multiple orderings
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    order_asc = sorted(vertices, key=lambda x: degrees[x])
    order_desc = sorted(vertices, key=lambda x: -degrees[x])
    order_odd = [v for v in vertices if v % 2 == 1]
    order_even = [v for v in vertices if v % 2 == 0]
    order_odd_first = sorted(vertices, key=lambda x: (x % 2 == 0, degrees[x]))
    order_value = list(range(1, N + 1))
    order_value_desc = list(range(N, 0, -1))

    candidates = [
        greedy_by_order(order_asc),
        greedy_by_order(order_desc),
        greedy_by_order(order_odd),
        greedy_by_order(order_even),
        greedy_by_order(order_odd_first),
        greedy_by_order(order_value),
        greedy_by_order(order_value_desc),
        greedy_min_degree(),
        greedy_max_degree(),
    ]

    # Multiple random shuffles
    for _ in range(30):
        r = vertices[:]
        random.shuffle(r)
        candidates.append(greedy_by_order(r))

    # Randomized min-degree greedy
    for _ in range(20):
        candidates.append(greedy_randomized(top_k=3))
        candidates.append(greedy_randomized(top_k=5))

    for c in candidates:
        if bin(c).count('1') > bin(best_mask).count('1'):
            best_mask = c

    # Local search: add vertices, then try 1-for-2 swaps
    def local_search(s_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding each excluded vertex
            excluded = [v for v in vertices if not ((s_mask >> v) & 1)]
            random.shuffle(excluded)
            for v in excluded:
                if time.time() >= time_limit:
                    break
                if can_add(v, s_mask):
                    s_mask |= (1 << v)
                    improved = True

            # Phase 2: Try 1-for-2 swaps
            if time.time() < time_limit:
                s_list = [v for v in vertices if (s_mask >> v) & 1]
                random.shuffle(s_list)
                for u in s_list[:min(len(s_list), 40)]:
                    if time.time() >= time_limit:
                        break
                    temp_mask = s_mask & ~(1 << u)
                    addable = []
                    for v in vertices:
                        if (s_mask >> v) & 1:
                            continue
                        if v == u:
                            continue
                        if can_add(v, temp_mask):
                            addable.append(v)
                    if len(addable) >= 2:
                        random.shuffle(addable)
                        found = False
                        limit = min(len(addable), 20)
                        for i in range(limit):
                            for j in range(i + 1, limit):
                                v1, v2 = addable[i], addable[j]
                                if (conflict_mask[v1] >> v2) & 1 == 0:
                                    s_mask = temp_mask | (1 << v1) | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
        return s_mask

    # Run local search from best candidate
    time_limit = deadline - 15.0
    best_mask = local_search(best_mask, time_limit)

    # WalkSAT phase: run from best and from perturbations
    walksat_time_limit = deadline - 10.0
    ws_result = walksat_search(best_mask, walksat_time_limit)
    if bin(ws_result).count('1') > bin(best_mask).count('1'):
        best_mask = ws_result

    # Multiple restarts with random perturbations + WalkSAT
    while time.time() < walksat_time_limit:
        # Create perturbation: remove 1-3 random vertices
        perturbed_mask = best_mask
        s_list = [v for v in vertices if (perturbed_mask >> v) & 1]
        if s_list:
            n_remove = min(random.randint(1, 3), len(s_list))
            to_remove = random.sample(s_list, n_remove)
            for v in to_remove:
                perturbed_mask &= ~(1 << v)

        # Run WalkSAT from perturbed state with a short time budget
        sub_limit = min(walksat_time_limit, time.time() + 1.5)
        result_mask = walksat_search(perturbed_mask, sub_limit)

        # Also try local search on the result
        sub_limit2 = min(walksat_time_limit, time.time() + 1.0)
        result_mask = local_search(result_mask, sub_limit2)

        if bin(result_mask).count('1') > bin(best_mask).count('1'):
            best_mask = result_mask

    # Final polish: 1-for-2 and 2-for-3 swaps
    polish_limit = deadline - 5.0
    best_mask = swap_1_for_2(best_mask, polish_limit)
    best_mask = swap_2_for_3(best_mask, polish_limit)
    # One more add pass
    for v in vertices:
        if time.time() >= polish_limit:
            break
        if not ((best_mask >> v) & 1) and can_add(v, best_mask):
            best_mask |= (1 << v)

    # Final validation
    result_list = sorted(mask_to_set(best_mask))
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            if (a * b) % (a + b) == 0:
                return list(range(1, 257, 2))

    return result_list
# EVOLVE-BLOCK-END
