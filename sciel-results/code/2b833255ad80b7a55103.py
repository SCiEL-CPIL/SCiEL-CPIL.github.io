"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

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

    # Greedy min-degree independent set as initial lower bound
    remaining = (1 << n) - 1
    selected_bits = 0

    while remaining:
        best_node = -1
        best_degree = n + 1
        for i in range(n):
            if (remaining >> i) & 1:
                deg = bin(conflict_bits[i] & remaining).count('1')
                if deg < best_degree:
                    best_degree = deg
                    best_node = i
        selected_bits |= (1 << best_node)
        remaining &= ~(1 << best_node)
        remaining &= ~conflict_bits[best_node]

    best_size = bin(selected_bits).count('1')
    best_solution_bits = selected_bits

    # Local search: greedy add
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(n):
            if (best_solution_bits >> i) & 1:
                continue
            if conflict_bits[i] & best_solution_bits:
                continue
            best_solution_bits |= (1 << i)
            best_size += 1
            improved = True

    # Swap-based local search
    unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]

    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for a in unselected:
            if (best_solution_bits >> a) & 1:
                continue
            conflicting = [b for b in range(n) if (best_solution_bits >> b) & 1 and (conflict_bits[a] >> b) & 1]
            if len(conflicting) <= 1:
                continue
            removed_bits = 0
            for b in conflicting:
                removed_bits |= (1 << b)
            candidate_bits = (best_solution_bits & ~removed_bits) | (1 << a)
            temp_bits = candidate_bits
            temp_size = bin(candidate_bits).count('1')
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
                best_solution_bits = temp_bits
                best_size = temp_size
                unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                improved = True
                break
        if not improved:
            break

    # Branch and bound for remaining time
    if time.time() < deadline - 2.0:
        # Use a simpler B&B: try to improve by 1
        # For each unselected node, check if we can swap it in
        # with a net gain
        remaining_time = deadline - 2.0 - time.time()
        
        # Try all pairs of swaps
        unselected_list = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
        selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
        
        for a in unselected_list:
            if time.time() > deadline - 2.0:
                break
            conflicting = [b for b in selected_list if (conflict_bits[a] >> b) & 1]
            if len(conflicting) <= 1:
                continue
            # Try removing subsets of conflicting (up to size 2 for speed)
            if len(conflicting) == 2:
                b1, b2 = conflicting
                removed = (1 << b1) | (1 << b2)
                candidate = (best_solution_bits & ~removed) | (1 << a)
                # Try to add back other nodes
                temp = candidate
                temp_size = bin(candidate).count('1')
                for c in unselected_list:
                    if c == a or (removed >> c) & 1:
                        continue
                    if (temp >> c) & 1:
                        continue
                    if conflict_bits[c] & temp:
                        continue
                    temp |= (1 << c)
                    temp_size += 1
                if temp_size > best_size:
                    best_solution_bits = temp
                    best_size = temp_size
                    unselected_list = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                    selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
            elif len(conflicting) == 3:
                # Try removing pairs from conflicting
                from itertools import combinations
                for pair in combinations(conflicting, 2):
                    if time.time() > deadline - 2.0:
                        break
                    removed = (1 << pair[0]) | (1 << pair[1])
                    candidate = (best_solution_bits & ~removed) | (1 << a)
                    temp = candidate
                    temp_size = bin(candidate).count('1')
                    for c in unselected_list:
                        if c == a or (removed >> c) & 1:
                            continue
                        if (temp >> c) & 1:
                            continue
                        if conflict_bits[c] & temp:
                            continue
                        temp |= (1 << c)
                        temp_size += 1
                    if temp_size > best_size:
                        best_solution_bits = temp
                        best_size = temp_size
                        unselected_list = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
                        selected_list = [i for i in range(n) if (best_solution_bits >> i) & 1]
                        break

    # Extract result
    result = []
    for i in range(n):
        if (best_solution_bits >> i) & 1:
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
        result = []
        for i in range(n):
            if (selected_bits >> i) & 1:
                result.append(valid_nodes[i])
        result.sort()

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
