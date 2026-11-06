import time
import math
import random

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
    idx_to_node = {i: node for i, node in enumerate(valid_nodes)}

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
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(n):
            if (best_solution_bits >> i) & 1:
                continue
            if conflict_bits[i] & best_solution_bits:
                continue
            best_solution_bits |= (1 << i)
            best_size += 1
            improved = True

    # Swap-based local search with bitsets
    for _ in range(200):
        if time.time() > deadline - 15.0:
            break
        improved = False
        unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
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
                improved = True
                break
        if not improved:
            break

    # Try multiple random restarts of greedy min-degree to find better starting points
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Randomize the order for tie-breaking in min-degree
        remaining = (1 << n) - 1
        sel_bits = 0
        
        while remaining:
            best_node = -1
            best_degree = n + 1
            # Collect all nodes with minimum degree, pick randomly among them
            min_deg_nodes = []
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = bin(conflict_bits[i] & remaining).count('1')
                    if deg < best_degree:
                        best_degree = deg
                        min_deg_nodes = [i]
                    elif deg == best_degree:
                        min_deg_nodes.append(i)
            if min_deg_nodes:
                best_node = random.choice(min_deg_nodes)
            sel_bits |= (1 << best_node)
            remaining &= ~(1 << best_node)
            remaining &= ~conflict_bits[best_node]
        
        cur_size = bin(sel_bits).count('1')
        
        # Greedy add
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(n):
                if (sel_bits >> i) & 1:
                    continue
                if conflict_bits[i] & sel_bits:
                    continue
                sel_bits |= (1 << i)
                cur_size += 1
                improved = True
        
        if cur_size > best_size:
            best_size = cur_size
            best_solution_bits = sel_bits

    # Final verification
    result = [idx_to_node[i] for i in range(n) if (best_solution_bits >> i) & 1]
    result.sort()
    
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                # Fallback to a known valid set
                return list(range(1, 257, 3))
    
    if not result:
        return [1]
    
    return result
