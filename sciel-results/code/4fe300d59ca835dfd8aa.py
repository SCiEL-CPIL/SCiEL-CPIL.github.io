import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    N = 256

    # Build squares up to 2*N
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Individually invalid: 2*a is a perfect square
    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j_val = s - node
            if 1 <= j_val <= N and j_val != node and j_val in valid_set:
                j_idx = node_to_idx[j_val]
                conflict_bits[i] |= (1 << j_idx)

    # Precompute degrees for heuristics
    def popcount(x):
        return bin(x).count('1')

    degrees = [popcount(conflict_bits[i]) for i in range(n)]

    def greedy_min_degree():
        """Greedy min-degree independent set."""
        remaining = (1 << n) - 1
        selected_bits = 0
        while remaining:
            best_node = -1
            best_degree = n + 1
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = popcount(conflict_bits[i] & remaining)
                    if deg < best_degree:
                        best_degree = deg
                        best_node = i
            if best_node == -1:
                break
            selected_bits |= (1 << best_node)
            remaining &= ~(1 << best_node)
            remaining &= ~conflict_bits[best_node]
        return selected_bits

    def greedy_add_pass(bits, size, time_limit):
        """Repeatedly add any non-conflicting node until no more can be added."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if (bits >> i) & 1:
                    continue
                if conflict_bits[i] & bits:
                    continue
                bits |= (1 << i)
                size += 1
                improved = True
        return bits, size

    def greedy_coloring_bound(candidates_bits):
        """
        Compute an upper bound on the independence number using greedy coloring
        of the conflict subgraph restricted to candidates.
        """
        if candidates_bits == 0:
            return 0

        uncolored = candidates_bits
        num_colors = 0

        while uncolored:
            num_colors += 1
            current_color_set = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                if conflict_bits[i] & current_color_set:
                    temp ^= lsb
                    continue
                current_color_set |= lsb
                temp ^= lsb
            uncolored &= ~current_color_set

        return num_colors

    def branch_and_bound(current_bits, candidates_bits, current_size, best_size_holder):
        """
        Recursive B&B to find MIS using greedy coloring bound.
        """
        if time.time() > deadline:
            return

        if candidates_bits == 0:
            if current_size > best_size_holder[0]:
                best_size_holder[0] = current_size
                best_size_holder[1] = current_bits
            return

        # Upper bound check using greedy coloring
        bound = greedy_coloring_bound(candidates_bits)
        if current_size + bound <= best_size_holder[0]:
            return

        # Select a branching vertex: high degree in candidate subgraph
        best_node = -1
        best_deg = -1
        temp = candidates_bits
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            deg = popcount(conflict_bits[i] & candidates_bits)
            if deg > best_deg:
                best_deg = deg
                best_node = i
            temp ^= lsb

        if best_node == -1:
            return

        # Branch 1: Include best_node
        new_current = current_bits | (1 << best_node)
        new_candidates = candidates_bits & ~(1 << best_node) & ~conflict_bits[best_node]
        branch_and_bound(new_current, new_candidates, current_size + 1, best_size_holder)

        if time.time() > deadline:
            return

        # Branch 2: Exclude best_node
        new_candidates_excl = candidates_bits & ~(1 << best_node)
        branch_and_bound(current_bits, new_candidates_excl, current_size, best_size_holder)

    def simulated_annealing(initial_mask, time_limit):
        """Simulated annealing to improve the independent set."""
        current_mask = initial_mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size

        T = 10.0
        cooling = 0.9998

        while time.time() < time_limit and T > 0.01:
            # Pick a random node not in current set
            unselected = [i for i in range(n) if not ((current_mask >> i) & 1)]
            if not unselected:
                break
            a = random.choice(unselected)

            # Find conflicting nodes in current set
            conflicting = []
            temp = conflict_bits[a] & current_mask
            while temp:
                lsb = temp & -temp
                b = lsb.bit_length() - 1
                conflicting.append(b)
                temp ^= lsb

            if not conflicting:
                new_mask = current_mask | (1 << a)
                new_size = current_size + 1
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                T *= cooling
                continue

            delta = 1 - len(conflicting)

            if delta > 0 or random.random() < math.exp(delta / T):
                removed_bits = 0
                for b in conflicting:
                    removed_bits |= (1 << b)
                current_mask = (current_mask & ~removed_bits) | (1 << a)
                current_size = current_size - len(conflicting) + 1

                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size

            T *= cooling

        return best_mask, best_size

    # --- Main Execution Flow ---

    random.seed(42)

    # Phase 1: Initial Solution via Greedy Min-Degree
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)

    # Phase 2: Polish with Greedy Add
    best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 50.0)

    # Phase 3: Local Search (Swap) to improve initial solution
    improved = True
    while improved and time.time() < deadline - 35.0:
        improved = False
        unselected = [i for i in range(n) if not ((best_mask >> i) & 1)]
        for a in unselected:
            if (best_mask >> a) & 1:
                continue
            conflicting = [b for b in range(n) if (best_mask >> b) & 1 and (conflict_bits[a] >> b) & 1]
            if len(conflicting) <= 1:
                continue
            removed_bits = 0
            for b in conflicting:
                removed_bits |= (1 << b)
            candidate_bits = (best_mask & ~removed_bits) | (1 << a)
            temp_bits = candidate_bits
            temp_size = popcount(candidate_bits)
            for c in unselected:
                if c == a or (removed_bits >> c) & 1:
                    continue
                if (temp_bits >> c) & 1:
                    continue
                if conflict_bits[c] & temp_bits:
                    continue
                temp_bits |= (1 << c)
                temp_size += 1
            if temp_size > best_size:
                best_mask = temp_bits
                best_size = temp_size
                improved = True
                break

    # Phase 4: Simulated Annealing to escape local optima
    if time.time() < deadline - 25.0:
        sa_mask, sa_size = simulated_annealing(best_mask, deadline - 20.0)
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 5: Multiple random restarts with SA (from inspiration 1)
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        # Greedy init with random order
        init_mask = 0
        for i in random_order:
            if (conflict_bits[i] & init_mask) == 0:
                init_mask |= (1 << i)
        init_size = popcount(init_mask)

        sa_mask, sa_size = simulated_annealing(init_mask, deadline - 8.0)
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 6: Branch and Bound with coloring bound
    best_holder = [best_size, best_mask]

    if time.time() < deadline - 3.0:
        initial_candidates = (1 << n) - 1
        branch_and_bound(0, initial_candidates, 0, best_holder)

    # Final Polish
    if time.time() < deadline - 1.0:
        final_mask, final_size = greedy_add_pass(best_holder[1], best_holder[0], deadline - 0.5)
        if final_size > best_holder[0]:
            best_holder[0] = final_size
            best_holder[1] = final_mask

    # Extract result
    result = []
    for i in range(n):
        if (best_holder[1] >> i) & 1:
            result.append(valid_nodes[i])

    result.sort()

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        result = []
        for i in range(n):
            if (greedy_min_degree() >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
