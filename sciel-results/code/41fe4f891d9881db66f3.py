"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    # Precompute clique and independent set properties for all masks
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
    
    is_valid = [is_clique[m] or is_independent[m] for m in range(size)]
    
    # Precompute popcount
    popcount = [0] * size
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Greedy initialization with multiple orderings
    def greedy_order(order):
        parts = []
        unassigned = (1 << n) - 1
        for v in order:
            v_bit = 1 << v
            if not (unassigned & v_bit):
                continue
            placed = False
            for i, part_mask in enumerate(parts):
                new_mask = part_mask | v_bit
                if is_valid[new_mask]:
                    parts[i] = new_mask
                    placed = True
                    break
            if not placed:
                parts.append(v_bit)
            unassigned ^= v_bit
        return parts
    
    # Try multiple orderings
    degrees = [popcount[adj[v]] for v in range(n)]
    orders = [
        list(range(n)),
        sorted(range(n), key=lambda v: -degrees[v]),
        sorted(range(n), key=lambda v: degrees[v]),
        sorted(range(n), key=lambda v: (-degrees[v], v)),
        sorted(range(n), key=lambda v: (degrees[v], v)),
    ]
    
    # Add some randomized orderings
    rng = random.Random(42)
    for _ in range(20):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(order)
    
    best_greedy = None
    for order in orders:
        parts = greedy_order(order)
        if best_greedy is None or len(parts) < len(best_greedy):
            best_greedy = parts
    
    best_count = len(best_greedy)
    best_parts = best_greedy[:]
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def mask_to_list(mask):
        result = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            result.append(v)
            m ^= lsb
        return result
    
    # Bounded search: find a partition using at most max_parts parts
    def bounded_search(unassigned, current_parts, max_parts):
        if time.time() > deadline:
            return None
        
        if not unassigned:
            if len(current_parts) <= max_parts:
                return current_parts[:]
            return None
        
        # Lower bound: need at least 1 more part if unassigned nonempty
        # Also, if we already have max_parts parts, every remaining vertex must fit
        remaining_parts = max_parts - len(current_parts)
        
        if remaining_parts < 0:
            return None
        
        # If remaining_parts == 0, all unassigned must fit into existing parts
        if remaining_parts == 0:
            # Check if all unassigned can be placed
            for v_bit in [1 << v for v in range(n) if unassigned & (1 << v)]:
                v = v_bit.bit_length() - 1
                can_place = False
                for p in current_parts:
                    if is_valid[p | v_bit]:
                        can_place = True
                        break
                if not can_place:
                    return None
            # All must fit, so we need to assign each to a valid part
            # This is a constraint satisfaction problem - use backtracking
            def assign_all(rem, parts_copy):
                if not rem:
                    return parts_copy
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                v_bit = lsb
                new_rem = rem ^ lsb
                for i, p in enumerate(parts_copy):
                    new_p = p | v_bit
                    if is_valid[new_p]:
                        parts_copy[i] = new_p
                        result = assign_all(new_rem, parts_copy)
                        if result is not None:
                            return result
                        parts_copy[i] = p
                return None
            return assign_all(unassigned, current_parts[:])
        
        # Select vertex with fewest valid placements
        best_v = -1
        min_placements = n + 1
        max_deg = -1
        
        u = unassigned
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            placements = 0
            for p in current_parts:
                if is_valid[p | lsb]:
                    placements += 1
            deg_in_unassigned = popcount[adj[v] & unassigned]
            if placements < min_placements or (placements == min_placements and deg_in_unassigned > max_deg):
                min_placements = placements
                max_deg = deg_in_unassigned
                best_v = v
                if placements == 0:
                    break
            u ^= lsb
        
        v = best_v
        v_bit = 1 << v
        new_unassigned = unassigned ^ v_bit
        
        # If vertex has no valid placements, it MUST start a new part
        if min_placements == 0:
            if len(current_parts) < max_parts:
                current_parts.append(v_bit)
                result = bounded_search(new_unassigned, current_parts, max_parts)
                current_parts.pop()
                return result
            else:
                return None
        
        # Try adding to existing parts (sort by part size, smaller first for more flexibility)
        part_indices = list(range(len(current_parts)))
        part_indices.sort(key=lambda i: popcount[current_parts[i]])
        
        for i in part_indices:
            p = current_parts[i]
            new_p = p | v_bit
            if is_valid[new_p]:
                current_parts[i] = new_p
                result = bounded_search(new_unassigned, current_parts, max_parts)
                current_parts[i] = p
                if result is not None:
                    return result
        
        # Try creating a new part
        if len(current_parts) < max_parts:
            current_parts.append(v_bit)
            result = bounded_search(new_unassigned, current_parts, max_parts)
            current_parts.pop()
            if result is not None:
                return result
        
        return None
    
    # Iterative deepening: try to find solutions with fewer and fewer parts
    for target_parts in range(best_count - 1, 0, -1):
        if time.time() > deadline:
            break
        
        result = bounded_search((1 << n) - 1, [], target_parts)
        if result is not None:
            best_count = len(result)
            best_parts = result[:]
            break
    
    # Convert best_parts (bitmasks) to list of lists of ints
    final_result = []
    for mask in best_parts:
        part_vertices = mask_to_list(mask)
        final_result.append(part_vertices)
    
    return final_result
# EVOLVE-BLOCK-END
