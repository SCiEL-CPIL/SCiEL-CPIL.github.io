import time
import numpy as np
import itertools

def _paley_tournament(p):
    """
    Returns orientation list for Paley tournament on p vertices.
    p must be a prime congruent to 3 mod 4.
    """
    squares = set((x * x) % p for x in range(1, p))
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            if d in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}

def _fast_check_property(n, orientation):
    """
    Check if every 4-set has a common dominator.
    Returns True if valid.
    """
    out_mask = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
            else:
                out_mask[j] |= (1 << i)
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        mask_bcd = (1 << b) | (1 << c) | (1 << d)
        if (out_mask[a] & mask_bcd) == mask_bcd:
            continue
        mask_acd = (1 << a) | (1 << c) | (1 << d)
        if (out_mask[b] & mask_acd) == mask_acd:
            continue
        mask_abd = (1 << a) | (1 << b) | (1 << d)
        if (out_mask[c] & mask_abd) == mask_abd:
            continue
        mask_abc = (1 << a) | (1 << b) | (1 << c)
        if (out_mask[d] & mask_abc) == mask_abc:
            continue
        return False
    return True

def _build_index_map(n):
    """Precompute index for each (i,j) pair."""
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
    return idx_map

def _count_violations(n, out_mask):
    """
    Count number of 4-sets with no common dominator.
    Returns the count.
    """
    violations = 0
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Check if any of a,b,c,d dominates the other three
        mask_bcd = (1 << b) | (1 << c) | (1 << d)
        mask_acd = (1 << a) | (1 << c) | (1 << d)
        mask_abd = (1 << a) | (1 << b) | (1 << d)
        mask_abc = (1 << a) | (1 << b) | (1 << c)
        
        if ((out_mask[a] & mask_bcd) != mask_bcd and
            (out_mask[b] & mask_acd) != mask_acd and
            (out_mask[c] & mask_abd) != mask_abd and
            (out_mask[d] & mask_abc) != mask_abc):
            violations += 1
    return violations

def _local_search(n, seed=0, deadline=None):
    """
    Local search to find a tournament with the 4-dominator property.
    Uses simulated annealing to escape local minima.
    """
    if deadline is None:
        deadline = time.time() + 10.0
    
    rng = np.random.default_rng(seed)
    idx_map = _build_index_map(n)
    
    # Start with random orientation
    orientation = np.zeros(n * (n - 1) // 2, dtype=np.int8)
    orientation[:] = rng.integers(0, 2, size=orientation.shape)
    
    # Build out_mask
    out_mask = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
            else:
                out_mask[j] |= (1 << i)
    
    current_violations = _count_violations(n, out_mask)
    best_violations = current_violations
    best_orientation = orientation.copy()
    
    if current_violations == 0:
        return {'n': n, 'orientation': orientation.tolist()}
    
    # Simulated annealing
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        # Pick a random edge to flip
        idx_flip = rng.integers(0, n * (n - 1) // 2)
        
        # Find (i, j) from idx_flip
        i = 0
        remaining = idx_flip
        while True:
            count_in_row = n - 1 - i
            if remaining < count_in_row:
                j = i + 1 + remaining
                break
            else:
                remaining -= count_in_row
                i += 1
        
        # Flip the edge
        old_bit = orientation[idx_flip]
        orientation[idx_flip] = 1 - old_bit
        
        # Update out_mask
        if old_bit == 0:
            # i -> j, flip to j -> i
            out_mask[i] &= ~(1 << j)
            out_mask[j] |= (1 << i)
        else:
            # j -> i, flip to i -> j
            out_mask[j] &= ~(1 << i)
            out_mask[i] |= (1 << j)
        
        new_violations = _count_violations(n, out_mask)
        delta = new_violations - current_violations
        
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_orientation = orientation.copy()
                if best_violations == 0:
                    return {'n': n, 'orientation': orientation.tolist()}
        else:
            # Revert
            orientation[idx_flip] = old_bit
            if old_bit == 0:
                out_mask[i] |= (1 << j)
                out_mask[j] &= ~(1 << i)
            else:
                out_mask[j] |= (1 << i)
                out_mask[i] &= ~(1 << j)
        
        temperature *= cooling
    
    # Check best
    if best_violations == 0:
        return {'n': n, 'orientation': best_orientation.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 100s for safety
    
    # Try local search for n = 31 to 66
    # Start from smaller n since we want to minimize n
    for n in range(31, 67):
        if time.time() > deadline - 10:  # Leave 10s for final check
            break
        
        # Try multiple seeds for local search
        found = None
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = _local_search(n, seed=seed, deadline=deadline - 10)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return found
    
    # Fallback: return Paley 67
    return _paley_tournament(67)
