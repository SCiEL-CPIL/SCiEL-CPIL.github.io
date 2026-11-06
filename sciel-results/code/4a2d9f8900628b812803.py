"""Combined approach: Branch-and-bound with greedy coloring + SA polish for max independent set."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

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

    # Greedy min-degree independent set
    def greedy_min_degree():
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

    # Greedy add pass
    def greedy_add_pass(bits, size):
        improved = True
        while improved:
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

    # Initial solution
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)
    best_mask, best_size = greedy_add_pass(best_mask, best_size)

    # Branch and bound with greedy coloring upper bound
    def greedy_coloring(subgraph_mask):
        """Greedy coloring to get an upper bound on independent set size."""
        # Get list of vertices in subgraph
        verts = [i for i in range(n) if (subgraph_mask >> i) & 1]
        if not verts:
            return 0

        # Order by degree in subgraph (highest first)
        verts.sort(key=lambda v: popcount(conflict_bits[v] & subgraph_mask), reverse=True)

        colors = {}
        color_count = 0
        for v in verts:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in verts:
                if u == v:
                    continue
                if (conflict_bits[v] >> u) & 1 and u in colors:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)

        return color_count

    def branch_and_bound(remaining_mask, current_size, current_mask):
        """Branch and bound for maximum independent set."""
        nonlocal best_mask, best_size

        if time.time() > deadline - 5.0:
            return

        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Upper bound: current_size + greedy coloring bound
        upper_bound = current_size + greedy_coloring(remaining_mask)
        if upper_bound <= best_size:
            return

        # Pick vertex with highest degree in remaining
        verts = [i for i in range(n) if (remaining_mask >> i) & 1]
        verts.sort(key=lambda v: popcount(conflict_bits[v] & remaining_mask), reverse=True)
        v = verts[0]

        # Branch 1: include v
        new_remaining = remaining_mask & ~(1 << v) & ~conflict_bits[v]
        branch_and_bound(new_remaining, current_size + 1, current_mask | (1 << v))

        # Branch 2: exclude v
        new_remaining = remaining_mask & ~(1 << v)
        branch_and_bound(new_remaining, current_size, current_mask)

    # Run branch and bound with time limit
    if time.time() < deadline - 20.0:
        # Try B&B from the greedy solution
        remaining = (1 << n) - 1 & ~best_mask
        # Actually, let's restart B&B from scratch but use best_size as initial bound
        bbb_start = time.time()
        branch_and_bound((1 << n) - 1, 0, 0)
        bbb_time = time.time() - bbb_start

    # SA polish
    if time.time() < deadline - 10.0:
        random.seed(42)

        for restart in range(20):
            if time.time() > deadline - 12.0:
                break

            # Perturb for restarts > 0
            if restart == 0:
                current_bits = best_mask
                current_size = best_size
            else:
                current_bits = best_mask
                current_size = best_size
                sel_list = [i for i in range(n) if (current_bits >> i) & 1]
                num_remove = max(1, current_size // 6)
                to_remove = random.sample(sel_list, min(num_remove, len(sel_list)))
                for b in to_remove:
                    current_bits &= ~(1 << b)
                    current_size -= 1
                # Greedy add back
                unsel = [i for i in range(n) if not ((current_bits >> i) & 1)]
                for a in unsel:
                    if not (conflict_bits[a] & current_bits):
                        current_bits |= (1 << a)
                        current_size += 1

            T = 1.0
            T_min = 0.01
            cooling = 0.9998
            max_steps = 50000

            unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
            selected_list = [i for i in range(n) if (current_bits >> i) & 1]

            for step in range(max_steps):
                if time.time() > deadline - 12.0:
                    break

                move_type = random.random()

                if move_type < 0.4 and unselected_list:
                    # Try to add a random unselected node
                    a = random.choice(unselected_list)
                    if conflict_bits[a] & current_bits:
                        continue
                    new_bits = current_bits | (1 << a)
                    new_size = current_size + 1
                    current_bits = new_bits
                    current_size = new_size
                    if current_size > best_size:
                        best_mask = current_bits
                        best_size = current_size
                    unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                    selected_list = [i for i in range(n) if (current_bits >> i) & 1]
                elif move_type < 0.65 and selected_list:
                    # Try to remove a random selected node
                    b = random.choice(selected_list)
                    new_bits = current_bits & ~(1 << b)
                    new_size = current_size - 1
                    delta = -1
                    if random.random() < math.exp(delta / T):
                        current_bits = new_bits
                        current_size = new_size
                        unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (current_bits >> i) & 1]
                else:
                    # Try a swap
                    if not selected_list or not unselected_list:
                        continue
                    b = random.choice(selected_list)
                    a = random.choice(unselected_list)
                    temp_bits = current_bits & ~(1 << b)
                    if conflict_bits[a] & temp_bits:
                        continue
                    new_bits = temp_bits | (1 << a)
                    # Accept with probability based on T
                    if random.random() < math.exp(0 / T):
                        current_bits = new_bits
                        if current_size > best_size:
                            best_mask = current_bits
                            best_size = current_size
                        unselected_list = [i for i in range(n) if not ((current_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (current_bits >> i) & 1]

                T *= cooling

            # Greedy add after SA
            current_bits, current_size = greedy_add_pass(current_bits, current_size)
            if current_size > best_size:
                best_mask = current_bits
                best_size = current_size

    # Final greedy add
    best_mask, best_size = greedy_add_pass(best_mask, best_size)

    # Extract result
    result = [valid_nodes[i] for i in range(n) if (best_mask >> i) & 1]
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
        # Fallback: simple greedy
        result = []
        for i in range(n):
            node = valid_nodes[i]
            conflict = False
            for r in result:
                s = node + r
                if math.isqrt(s) ** 2 == s:
                    conflict = True
                    break
            if not conflict:
                result.append(node)
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
