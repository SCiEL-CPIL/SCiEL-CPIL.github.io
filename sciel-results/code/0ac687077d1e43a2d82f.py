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

def _build_edge_map(n):
    """Build a mapping from edge index to (i, j) pair."""
    edge_map = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
            idx += 1
    return edge_map

def _check_property_fast(n, in_neigh):
    """Check if every 4 vertices have a common dominator using bitsets."""
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ib = in_neigh[b]
            for c in range(b + 1, n - 1):
                ic = in_neigh[c]
                for d in range(c + 1, n):
                    id_ = in_neigh[d]
                    
                    # Check a
                    if (ib & (1 << a)) and (ic & (1 << a)) and (id_ & (1 << a)):
                        continue
                    # Check b
                    if (ia & (1 << b)) and (ic & (1 << b)) and (id_ & (1 << b)):
                        continue
                    # Check c
                    if (ia & (1 << c)) and (ib & (1 << c)) and (ic & (1 << c)):
                        continue
                    # Check d
                    if (ia & (1 << d)) and (ib & (1 << d)) and (ic & (1 << d)):
                        continue
                    
                    return False
    return True

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

def _count_violations(n, in_neigh):
    """Count the number of 4-sets that don't have a common dominator."""
    count = 0
    for a in range(n - 3):
        ia = in_neigh[a]
        for b in range(a + 1, n - 2):
            ib = in_neigh[b]
            for c in range(b + 1, n - 1):
                ic = in_neigh[c]
                for d in range(c + 1, n):
                    id_ = in_neigh[d]
                    
                    if (ib & (1 << a)) and (ic & (1 << a)) and (id_ & (1 << a)):
                        continue
                    if (ia & (1 << b)) and (ic & (1 << b)) and (id_ & (1 << b)):
                        continue
                    if (ia & (1 << c)) and (ib & (1 << c)) and (ic & (1 << c)):
                        continue
                    if (ia & (1 << d)) and (ib & (1 << d)) and (ic & (1 << d)):
                        continue
                    count += 1
    return count

def _get_affected_4sets(n, i, j):
    """Get all 4-sets that contain both i and j."""
    affected = []
    others = [v for v in range(n) if v != i and v != j]
    for k in range(len(others) - 1):
        for l in range(k + 1, len(others)):
            a, b = others[k], others[l]
            s = sorted([i, j, a, b])
            affected.append(s)
    return affected

def _check_4set_valid(in_neigh, quad):
    """Check if a specific 4-set has a common dominator."""
    a, b, c, d = quad
    ia, ib, ic, id_ = in_neigh[a], in_neigh[b], in_neigh[c], in_neigh[d]
    
    if (ib & (1 << a)) and (ic & (1 << a)) and (id_ & (1 << a)):
        return True
    if (ia & (1 << b)) and (ic & (1 << b)) and (id_ & (1 << b)):
        return True
    if (ia & (1 << c)) and (ib & (1 << c)) and (ic & (1 << c)):
        return True
    if (ia & (1 << d)) and (ib & (1 << d)) and (ic & (1 << d)):
        return True
    return False

def _local_search(n, time_limit, start_orientation=None, seed=None):
    """Simulated annealing local search for a valid tournament on n vertices."""
    deadline = time.time() + time_limit
    
    if seed is not None:
        random.seed(seed)
    
    if start_orientation is not None:
        orientation = start_orientation[:]
    else:
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    
    in_neigh = _build_in_neigh(n, orientation)
    edge_map = _build_edge_map(n)
    
    current_violations = _count_violations(n, in_neigh)
    if current_violations == 0:
        return {'n': n, 'orientation': orientation}
    
    best_violations = current_violations
    best_orientation = orientation[:]
    best_in_neigh = in_neigh[:]
    
    # Simulated annealing parameters
    initial_temp = 1.0
    final_temp = 0.01
    cooling_rate = 0.999
    
    temp = initial_temp
    step = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
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
        
        # Compute delta in violations for affected 4-sets
        # A 4-set is affected if it contains both i and j
        # Number of such sets: C(n-2, 2)
        
        # We need to count how many of these 4-sets change from valid to invalid or vice versa
        # Before flip: count valid affected sets
        # After flip: count valid affected sets
        # delta = after - before
        
        # For efficiency, only check affected sets
        others = [v for v in range(n) if v != i and v != j]
        
        before_valid = 0
        after_valid = 0
        
        for k in range(len(others) - 1):
            for l in range(k + 1, len(others)):
                a, b = others[k], others[l]
                quad = sorted([i, j, a, b])
                # Check validity with current (after flip) state
                if _check_4set_valid(in_neigh, quad):
                    after_valid += 1
        
        # To get before_valid, we'd need to revert, which is expensive.
        # Alternative: compute delta directly by checking which sets changed status.
        # A set changes status only if the flipped edge is critical for its validity.
        
        # Simpler approach: just recompute violations periodically
        if step % 50 == 0 or step <= 100:
            new_violations = _count_violations(n, in_neigh)
            delta_v = new_violations - current_violations
            
            # Accept or reject based on simulated annealing
            if delta_v <= 0:
                # Accept improvement
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_orientation = orientation[:]
                    best_in_neigh = in_neigh[:]
                    no_improve_steps = 0
                    if new_violations == 0:
                        return {'n': n, 'orientation': orientation}
                else:
                    no_improve_steps += 1
            else:
                # Deterioration - accept with probability exp(-delta/temp)
                if random.random() < np.exp(-delta_v / max(temp, 1e-10)):
                    current_violations = new_violations
                    no_improve_steps += 1
                else:
                    # Reject: revert the flip
                    orientation[edge_idx] = 1 - orientation[edge_idx]
                    if orientation[edge_idx] == 0:
                        in_neigh[i] &= ~(1 << j)
                        in_neigh[j] |= (1 << i)
                    else:
                        in_neigh[j] &= ~(1 << i)
                        in_neigh[i] |= (1 << j)
        
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
        
        # If stuck for too long, restart from best
        if no_improve_steps > max_no_improve:
            orientation = best_orientation[:]
            in_neigh = best_in_neigh[:]
            current_violations = best_violations
            no_improve_steps = 0
            temp = initial_temp * 0.5  # Restart with lower temperature
    
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5 seconds safety margin
    
    # Primes p = 3 mod 4
    # Smallest known valid is 67.
    # Let's try 31, 43, 47, 59, 67.
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
        if remaining_time > 10:
            # Try to find a valid tournament with n < best_n
            # Focus on best_n - 1 first with most time, then smaller
            target_n = best_n - 1
            time_for_this = min(remaining_time - 5, 80.0)
            if time_for_this > 5:
                result = _local_search(target_n, time_for_this, seed=42)
                if result is not None:
                    best_result = result
                    best_n = target_n
            
            # Try even smaller if we have time
            if best_n > 32 and time.time() < deadline - 10:
                target_n = best_n - 1
                time_for_this = min(deadline - time.time() - 5, 30.0)
                if time_for_this > 5:
                    result = _local_search(target_n, time_for_this, seed=123)
                    if result is not None:
                        best_result = result
                        best_n = target_n
    
    # If no Paley worked, fall back to 67 (known valid)
    if best_result is None:
        if time.time() < deadline:
            result = _try_paley_and_check(67, deadline)
            if result is not None:
                return result
        # Absolute fallback
        return _paley_orientation(67)
    
    return best_result
