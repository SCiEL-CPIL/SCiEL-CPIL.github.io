"""Combined approach: High-performance Simulated Annealing with Add-and-Remove moves for Max Independent Set."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    N = 256

    # Build squares up to 2*N
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Individually invalid: 2*a is a perfect square (diagonal constraint)
    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    # conflict_bits[i] is a bitmask of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    def popcount(x):
        return bin(x).count('1')

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

    def greedy_add_pass(bits, size):
        """Repeatedly add any non-conflicting node until no more can be added."""
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

    # Initial solution: Greedy min-degree
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)
    best_mask, best_size = greedy_add_pass(best_mask, best_size)

    # Simulated Annealing
    def simulated_annealing(init_mask, init_size, t_start, t_end, steps):
        current_mask = init_mask
        current_size = init_size
        T = t_start
        if steps > 0:
            cooling_rate = (t_end / t_start) ** (1.0 / steps)
        else:
            cooling_rate = 1.0
        
        # Precompute list of all nodes for random selection
        all_nodes = list(range(n))
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random node to try to add
            v = random.choice(all_nodes)
            
            # If already selected, skip (or we could try to remove, but add-and-remove is the main move)
            if (current_mask >> v) & 1:
                continue
            
            # Find neighbors of v in the current set
            neighbors_in_set = conflict_bits[v] & current_mask
            n_neighbors = popcount(neighbors_in_set)
            
            if n_neighbors == 0:
                # Can add v directly, strictly improves size
                current_mask |= (1 << v)
                current_size += 1
            else:
                # Remove neighbors, add v
                # New size = current_size - n_neighbors + 1
                # Delta = (current_size - n_neighbors + 1) - current_size = 1 - n_neighbors
                delta = 1 - n_neighbors
                
                if delta >= 0:
                    # Strictly improves or neutral (if delta=0, size is same, but structure changes)
                    # We accept all non-negative deltas to explore
                    current_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                    current_size += delta
                else:
                    # Worse, accept with probability exp(delta/T)
                    if random.random() < math.exp(delta / T):
                        current_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                        current_size += delta
            
            T *= cooling_rate
        
        return current_mask, current_size

    # Main SA Loop with Restarts
    random.seed(42)
    
    # Long run from greedy init
    if time.time() < deadline - 10.0:
        mask, size = simulated_annealing(best_mask, best_size, 50.0, 0.01, 200000)
        if size > best_size:
            best_size = size
            best_mask = mask
        # Greedy polish
        mask, size = greedy_add_pass(mask, size)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Multiple restarts with random initialization or perturbed best
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Create a perturbed version of the best solution or a new random one
        if r == 0:
            # Slight perturbation of best
            current_mask = best_mask
            current_size = best_size
            # Remove a few random nodes
            sel_list = [i for i in range(n) if (current_mask >> i) & 1]
            if sel_list:
                num_remove = max(1, current_size // 5)
                to_remove = random.sample(sel_list, min(num_remove, len(sel_list)))
                for b in to_remove:
                    current_mask &= ~(1 << b)
                    current_size -= 1
            # Greedy add back
            current_mask, current_size = greedy_add_pass(current_mask, current_size)
        else:
            # Random greedy init with random order
            order = list(range(n))
            random.shuffle(order)
            current_mask = 0
            current_size = 0
            for i in order:
                if not (conflict_bits[i] & current_mask):
                    current_mask |= (1 << i)
                    current_size += 1
            current_mask, current_size = greedy_add_pass(current_mask, current_size)
        
        # Run SA
        # Shorter runs for restarts
        steps = 50000
        t_start = 20.0
        t_end = 0.01
        mask, size = simulated_annealing(current_mask, current_size, t_start, t_end, steps)
        
        # Greedy polish
        mask, size = greedy_add_pass(mask, size)
        
        if size > best_size:
            best_size = size
            best_mask = mask

    # Final Polish
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
