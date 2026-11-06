"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _make_paley(p):
    """Construct Paley tournament orientation for prime p."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _build_out_masks(orientation, n):
    """Build out-neighborhood bitmask for each vertex."""
    out_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                out_masks[i] |= (1 << j)
            else:
                out_masks[j] |= (1 << i)
            idx += 1
    return out_masks

def _count_violations(out_masks, n):
    """Count number of 4-sets without a common dominator."""
    violations = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    S_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    found = False
                    for v in range(n):
                        if (out_masks[v] & S_mask) == S_mask:
                            found = True
                            break
                    if not found:
                        violations += 1
    return violations

def _remove_vertex(orientation, n, v):
    """Remove vertex v from tournament, return new orientation for n-1 vertices."""
    new_n = n - 1
    new_orientation = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i == v or j == v:
                idx += 1
                continue
            new_i = i if i < v else i - 1
            new_j = j if j < v else j - 1
            new_orientation.append(orientation[idx])
            idx += 1
    return new_orientation

def _local_search(orientation, n, deadline, rng):
    """Local search: flip edges to reduce violations."""
    out_masks = _build_out_masks(orientation, n)
    current_violations = _count_violations(out_masks, n)
    
    if current_violations == 0:
        return orientation, out_masks, True
    
    best_orientation = orientation[:]
    best_out_masks = out_masks[:]
    best_violations = current_violations
    
    while time.time() < deadline - 5.0:
        # Pick a random edge to flip
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Find index in orientation
        idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
        old_bit = orientation[idx]
        new_bit = 1 - old_bit
        orientation[idx] = new_bit
        
        # Update out_masks
        if old_bit == 0:
            # was i -> j, now j -> i
            out_masks[i] &= ~(1 << j)
            out_masks[j] |= (1 << i)
        else:
            # was j -> i, now i -> j
            out_masks[j] &= ~(1 << i)
            out_masks[i] |= (1 << j)
        
        new_violations = _count_violations(out_masks, n)
        
        if new_violations < best_violations:
            best_orientation = orientation[:]
            best_out_masks = out_masks[:]
            best_violations = new_violations
            if best_violations == 0:
                break
        else:
            # Revert
            orientation[idx] = old_bit
            if old_bit == 0:
                out_masks[i] |= (1 << j)
                out_masks[j] &= ~(1 << i)
            else:
                out_masks[j] |= (1 << i)
                out_masks[i] &= ~(1 << j)
    
    return best_orientation, best_out_masks, best_violations == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Start with Paley tournament on 67 vertices
    p = 67
    orientation_67 = _make_paley(p)
    
    # Verify 67-vertex tournament is valid (should be)
    out_masks_67 = _build_out_masks(orientation_67, p)
    violations_67 = _count_violations(out_masks_67, p)
    
    if violations_67 == 0:
        # Try to reduce to 66 vertices by removing one vertex
        for v in range(p):
            if time.time() > deadline - 30.0:
                break
            orientation_66 = _remove_vertex(orientation_67, p, v)
            out_masks_66 = _build_out_masks(orientation_66, 66)
            violations_66 = _count_violations(out_masks_66, 66)
            
            if violations_66 == 0:
                return {'n': 66, 'orientation': orientation_66}
            
            # Try local search to repair
            repaired_orientation, _, success = _local_search(
                orientation_66[:], 66, deadline, rng
            )
            if success:
                return {'n': 66, 'orientation': repaired_orientation}
    
    # Fallback: return the valid 67-vertex tournament
    return {'n': 67, 'orientation': orientation_67}

# EVOLVE-BLOCK-END
