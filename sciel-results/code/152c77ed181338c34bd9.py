"""Branch-and-bound with greedy coloring upper bound for maximum independent set."""
# EVOLVE-BLOCK-START
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
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    def popcount(x):
        return bin(x).count('1')

    def greedy_color_bound(available_mask):
        """
        Greedy coloring of the subgraph induced by available_mask.
        Returns number of colors used (upper bound on max independent set).
        """
        color_classes = []
        remaining = available_mask
        while remaining:
            # Find a maximal independent set in remaining
            color_class = 0
            temp_remaining = remaining
            while temp_remaining:
                # Pick vertex with minimum degree in temp_remaining
                best_v = -1
                best_deg = n + 1
                for v in range(n):
                    if (temp_remaining >> v) & 1:
                        deg = popcount(conflict_bits[v] & temp_remaining)
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
                color_class |= (1 << best_v)
                temp_remaining &= ~(1 << best_v)
                temp_remaining &= ~conflict_bits[best_v]
            color_classes.append(color_class)
            remaining &= ~color_class
        return len(color_classes)

    def branch_and_bound(available_mask, current_size, current_mask, best_size_ref):
        """
        Branch and bound with greedy coloring upper bound.
        Mutates best_size_ref[0] and best_mask_ref[0].
        """
        if available_mask == 0:
            return

        # Upper bound via greedy coloring
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size_ref[0]:
            return

        if time.time() > deadline - 5.0:
            return

        # Pick a vertex to branch on (highest degree in available for tighter bound)
        best_v = -1
        best_deg = -1
        for v in range(n):
            if (available_mask >> v) & 1:
                deg = popcount(conflict_bits[v] & available_mask)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_mask = current_mask | (1 << best_v)
        new_available = available_mask & ~(1 << best_v) & ~conflict_bits[best_v]
        branch_and_bound(new_available, current_size + 1, new_mask, best_size_ref)

        # Branch 2: Exclude best_v
        new_available2 = available_mask & ~(1 << best_v)
        branch_and_bound(new_available2, current_size, current_mask, best_size_ref)

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

    def simulated_annealing(mask, size, steps, t_start, t_end, time_limit):
        """Simulated annealing with single-vertex moves."""
        current_mask = mask
        current_size = size
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(steps, 1))

        for _ in range(steps):
            if time.time() > time_limit:
                break

            unselected = [i for i in range(n) if not (current_mask >> i) & 1]
            if not unselected:
                break
            v = random.choice(unselected)

            neighbors_in_set = conflict_bits[v] & current_mask
            n_neighbors = popcount(neighbors_in_set)

            if n_neighbors == 0:
                trial_mask = current_mask | (1 << v)
                trial_size = current_size + 1
                delta = -1
            else:
                trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                trial_size = current_size - n_neighbors + 1
                delta = n_neighbors - 1

            if delta <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                if t > 0 and random.random() < math.exp(-delta / t):
                    current_mask = trial_mask
                    current_size = trial_size

            t *= cooling

        return current_mask, current_size

    random.seed(42)

    # Phase 1: Greedy min-degree initial solution
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)

    # Phase 2: Greedy add pass
    best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 50.0)

    # Phase 3: Branch and bound with greedy coloring
    # Use the current best as the lower bound
    best_size_ref = [best_size]
    best_mask_ref = [best_mask]
    
    # Modify branch_and_bound to update best_mask_ref
    def branch_and_bound_v2(available_mask, current_size, current_mask):
        if available_mask == 0:
            return

        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size_ref[0]:
            return

        if time.time() > deadline - 10.0:
            return

        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        for v in range(n):
            if (available_mask >> v) & 1:
                deg = popcount(conflict_bits[v] & available_mask)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_mask = current_mask | (1 << best_v)
        new_available = available_mask & ~(1 << best_v) & ~conflict_bits[best_v]
        branch_and_bound_v2(new_available, current_size + 1, new_mask)
        
        # Update best if we found a better solution
        if current_size + 1 > best_size_ref[0]:
            best_size_ref[0] = current_size + 1
            best_mask_ref[0] = new_mask

        # Branch 2: Exclude best_v
        new_available2 = available_mask & ~(1 << best_v)
        branch_and_bound_v2(new_available2, current_size, current_mask)

    # Run B&B on the full graph
    full_available = (1 << n) - 1
    branch_and_bound_v2(full_available, 0, 0)

    if best_size_ref[0] > best_size:
        best_size = best_size_ref[0]
        best_mask = best_mask_ref[0]

    # Phase 4: SA polish
    if time.time() < deadline - 15.0:
        sa_mask, sa_size = simulated_annealing(
            best_mask, best_size,
            steps=50000,
            t_start=30.0,
            t_end=0.01,
            time_limit=deadline - 10.0
        )
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 5: Multiple random restarts with SA
    restart_limit = deadline - 8.0
    for restart in range(20):
        if time.time() > restart_limit:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        init_mask = 0
        for i in random_order:
            if (conflict_bits[i] & init_mask) == 0:
                init_mask |= (1 << i)
        init_size = popcount(init_mask)

        sa_mask, sa_size = simulated_annealing(
            init_mask, init_size,
            steps=30000,
            t_start=20.0,
            t_end=0.01,
            time_limit=deadline - 6.0
        )
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size

    # Phase 6: Final greedy polish
    if time.time() < deadline - 3.0:
        best_mask, best_size = greedy_add_pass(best_mask, best_size, deadline - 2.0)

    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
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
        # Fallback: use greedy
        result = []
        for i in range(n):
            if (greedy_min_degree() >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
