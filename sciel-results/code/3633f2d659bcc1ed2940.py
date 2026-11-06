import time
import random
import numpy as np

def _has_k222_count_bitset(adj_bits, n):
    """
    Counts the number of K2,2,2 subgraphs using bitset representation.
    adj_bits[v] is an integer whose k-th bit is 1 iff edge (v,k) exists.
    """
    count = 0
    for c in range(n - 1):
        for d in range(c + 1, n):
            if not (adj_bits[c] >> d) & 1:
                continue
            common = adj_bits[c] & adj_bits[d]
            common &= ~(1 << c)
            common &= ~(1 << d)
            m_size = common.bit_count()
            if m_size < 4:
                continue
            common_idx = [i for i in range(n) if (common >> i) & 1]
            for i in range(m_size - 1):
                a = common_idx[i]
                valid_b = common_idx[i+1:]
                if not valid_b:
                    continue
                # For each b in valid_b, check A[a,b]==1 and |N(a) ∩ N(b) ∩ common| >= 2
                na_common = adj_bits[a] & common
                for b in valid_b:
                    if (adj_bits[a] >> b) & 1:
                        shared = na_common & adj_bits[b]
                        if shared.bit_count() >= 2:
                            count += 1
    return count

def _is_valid_fast_bitset(adj_bits, n):
    return _has_k222_count_bitset(adj_bits, n) == 0

def _build_adj_bitset(colors, n):
    adj_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
            idx += 1
    return adj_bits

def _get_color_list_bitset(adj_bits, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(1 if (adj_bits[i] >> j) & 1 else 0)
    return colors

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _find_violation_edges_bitset(adj_bits, n):
    """
    Find edges that are part of monochromatic K2,2,2 subgraphs.
    Returns a list of (i, j) pairs.
    """
    violating = set()
    for c in range(n - 1):
        for d in range(c + 1, n):
            if not (adj_bits[c] >> d) & 1:
                continue
            common = adj_bits[c] & adj_bits[d]
            common &= ~(1 << c)
            common &= ~(1 << d)
            m_size = common.bit_count()
            if m_size < 4:
                continue
            common_idx = [i for i in range(n) if (common >> i) & 1]
            for i in range(m_size - 1):
                a = common_idx[i]
                valid_b = common_idx[i+1:]
                if not valid_b:
                    continue
                na_common = adj_bits[a] & common
                for b in valid_b:
                    if (adj_bits[a] >> b) & 1:
                        shared = na_common & adj_bits[b]
                        if shared.bit_count() >= 2:
                            violating.add((c, d))
                            violating.add((a, c))
                            violating.add((a, d))
                            violating.add((a, b))
                            violating.add((b, c))
                            violating.add((b, d))
    return list(violating)

def _simulated_annealing_bitset(colors, n, time_limit=30, initial_temp=1.0, cooling_rate=0.9999):
    """
    Simulated annealing with targeted edge flipping using bitset adjacency.
    """
    start_time = time.time()
    adj_bits = _build_adj_bitset(colors, n)
    current_cost = _has_k222_count_bitset(adj_bits, n)
    
    if current_cost == 0:
        return colors, True
    
    best_cost = current_cost
    best_colors = colors[:]
    temp = initial_temp
    
    # Precompute edge-to-vertex mapping
    edge_to_ij = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_ij.append((i, j))
            idx += 1
    
    # Build reverse mapping
    ij_to_idx = {}
    for idx, (i, j) in enumerate(edge_to_ij):
        ij_to_idx[(i, j)] = idx
        ij_to_idx[(j, i)] = idx
    
    # Get initial violation edges for targeted search
    violation_edges = _find_violation_edges_bitset(adj_bits, n)
    if not violation_edges:
        return colors, True
    
    step_count = 0
    while time.time() - start_time < time_limit:
        step_count += 1
        
        # 70% chance: pick an edge from violations, 30% random
        if violation_edges and random.random() < 0.7:
            vi, vj = random.choice(violation_edges)
            idx = ij_to_idx.get((vi, vj))
            if idx is None:
                idx = ij_to_idx.get((vj, vi))
            if idx is None:
                idx = random.randint(0, len(colors) - 1)
        else:
            idx = random.randint(0, len(colors) - 1)
        
        i, j = edge_to_ij[idx]
        
        # Flip edge (i, j)
        old_val = (adj_bits[i] >> j) & 1
        if old_val:
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
        else:
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
        colors[idx] = 1 - old_val
        
        new_cost = _has_k222_count_bitset(adj_bits, n)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors, True
            # Update violation edges periodically
            if step_count % 30 == 0 or new_cost == 0:
                violation_edges = _find_violation_edges_bitset(adj_bits, n)
        else:
            # Revert
            if old_val:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
            else:
                adj_bits[i] &= ~(1 << j)
                adj_bits[j] &= ~(1 << i)
            colors[idx] = old_val
        
        temp *= cooling_rate
        if temp < 1e-10:
            temp = 1e-10
    
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 140
    best_n = 0
    best_colors = None
    
    # Target n=29
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29)
    colors = _paley(target_n, p)
    if colors:
        adj_bits = _build_adj_bitset(colors, target_n)
        if _is_valid_fast_bitset(adj_bits, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Try Simulated Annealing from Paley
    colors_copy = colors[:]
    _, success = _simulated_annealing_bitset(colors_copy, target_n, time_limit=50, initial_temp=1.5, cooling_rate=0.9999)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Random Restarts with Simulated Annealing
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        remaining = deadline - 10 - time.time()
        tl = max(3, min(8, remaining / max(1, 30 - attempt)))
        _, success = _simulated_annealing_bitset(colors, target_n, time_limit=tl, initial_temp=0.8, cooling_rate=0.9999)
        if success:
            return {'n': target_n, 'colors': colors}
    
    # Fallback: Try smaller n with Paley constructions
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        p_candidate = None
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                p_candidate = pp
                break
        if not p_candidate:
            continue
        colors = _paley(n, p_candidate)
        if colors:
            adj_bits = _build_adj_bitset(colors, n)
            if _is_valid_fast_bitset(adj_bits, n):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
