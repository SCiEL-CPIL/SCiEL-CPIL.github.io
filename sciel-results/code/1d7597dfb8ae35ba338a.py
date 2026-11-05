import time
import random
import numpy as np
from itertools import combinations

def _build_in_neigh(n, orientation):
    """Build in-neighbor bitsets from orientation list."""
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
    return in_neigh

def _count_bad_sets(n, in_neigh):
    """
    Count the number of 'bad' 4-sets (isomorphic to 3-cycle + sink).
    A bad set exists for vertex d if in_neigh[d] contains a 3-cycle.
    We sum the number of 3-cycles in in_neigh[d] for all d.
    Note: Each bad 4-set has exactly one sink (the vertex with in-degree 3 within the set).
    So this count is exactly the number of invalid 4-sets.
    """
    count = 0
    all_mask = (1 << n) - 1
    for d in range(n):
        M_d = in_neigh[d]
        if M_d.bit_count() < 3:
            continue
        
        # Count 3-cycles in subgraph induced by M_d
        # Iterate a in M_d
        temp = M_d
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            temp ^= lsb
            
            in_a = in_neigh[a]
            # Candidates for b: b in M_d, b != a, a->b
            # a->b => b not in in_neigh[a]
            C = M_d & ~(1 << a) & ~in_neigh[a]
            
            if not C:
                continue
                
            # Iterate b in C
            temp_b = C
            while temp_b:
                lsb_b = temp_b & -temp_b
                b = lsb_b.bit_length() - 1
                temp_b ^= lsb_b
                
                # Candidates for c: c in M_d, b->c, c->a
                # b->c => c not in in_neigh[b]
                # c->a => c in in_neigh[a]
                # c != b
                target = M_d & ~in_neigh[b] & in_neigh[a] & ~(1 << b)
                count += target.bit_count()
                
    return count

def _check_property_fast(n, in_neigh):
    return _count_bad_sets(n, in_neigh) == 0

def _paley_orientation(p):
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': n, 'orientation': orientation}

def _try_paley_and_check(p, deadline):
    """Try Paley tournament on p vertices and check if valid."""
    if time.time() > deadline:
        return None
    result = _paley_orientation(p)
    n = result['n']
    in_neigh = _build_in_neigh(n, result['orientation'])
    if _check_property_fast(n, in_neigh):
        return result
    return None

def _local_search(n, time_limit, start_orientation=None):
    """Local search for a valid tournament on n vertices."""
    deadline = time.time() + time_limit
    
    if start_orientation is not None:
        orientation = start_orientation[:]
    else:
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    
    in_neigh = _build_in_neigh(n, orientation)
    violations = _count_bad_sets(n, in_neigh)
    
    if violations == 0:
        return {'n': n, 'orientation': orientation}
    
    best_violations = violations
    best_orientation = orientation[:]
    
    # Precompute edge mapping for fast flip updates
    # edge_idx -> (i, j)
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[idx] = (i, j)
            idx += 1
            
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random edge to flip
        edge_idx = random.randint(0, len(orientation) - 1)
        i, j = edge_map[edge_idx]
        
        # Flip the edge
        orientation[edge_idx] = 1 - orientation[edge_idx]
        
        # Update in_neigh
        if orientation[edge_idx] == 0:
            # i -> j
            in_neigh[i] &= ~(1 << j)
            in_neigh[j] |= (1 << i)
        else:
            # j -> i
            in_neigh[j] &= ~(1 << i)
            in_neigh[i] |= (1 << j)
        
        # Count violations
        # To save time, we can count less frequently?
        # But the count is fast now.
        violations = _count_bad_sets(n, in_neigh)
        
        if violations < best_violations:
            best_violations = violations
            best_orientation = orientation[:]
            if violations == 0:
                return {'n': n, 'orientation': orientation}
        elif violations > best_violations:
            # Revert if it gets worse?
            # Simple hill climbing: revert
            orientation[edge_idx] = 1 - orientation[edge_idx]
            if orientation[edge_idx] == 0:
                in_neigh[i] &= ~(1 << j)
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[j] &= ~(1 << i)
                in_neigh[i] |= (1 << j)
            # violations = _count_bad_sets(n, in_neigh) # Revert to previous state
        # If equal, keep (random walk)
        
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5 seconds safety margin
    
    # Primes p = 3 mod 4
    primes_to_try = [31, 43, 47, 59, 67]
    
    best_result = None
    best_n = 91
    
    for p in primes_to_try:
        if time.time() > deadline:
            break
        
        result = _try_paley_and_check(p, deadline)
        if result is not None:
            best_result = result
            best_n = p
            break  # Found the smallest valid Paley
    
    # If we found a valid Paley tournament, try to beat it with local search
    if best_result is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            # Try to find a valid tournament with n < best_n
            # Start from best_n - 1 down to max(5, best_n - 10)
            # Use time proportional to remaining time
            # Allocate time based on how many we want to try
            # Let's try up to 10 steps down
            target_start = best_n - 1
            target_end = max(5, best_n - 10)
            
            # Calculate time per target
            num_targets = target_start - target_end + 1
            time_per_target = (remaining_time - 5) / num_targets if num_targets > 0 else 0
            
            for target_n in range(target_start, target_end - 1, -1):
                if time.time() > deadline - 5:
                    break
                if time_per_target < 0.5:
                    break
                    
                result = _local_search(target_n, time_per_target)
                if result is not None:
                    best_result = result
                    best_n = target_n
                    # Continue trying smaller n
                    continue
                else:
                    # If local search failed, don't try even smaller n (they're harder)
                    # But maybe with more time?
                    # For now, break.
                    break
    
    # If no Paley worked, fall back to 67 (known valid)
    if best_result is None:
        if time.time() < deadline:
            result = _try_paley_and_check(67, deadline)
            if result is not None:
                return result
        # Absolute fallback
        return _paley_orientation(67)
    
    return best_result
