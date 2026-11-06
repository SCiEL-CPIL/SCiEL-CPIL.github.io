"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _get_paleyn_orientation(n):
    """
    Constructs a Paley tournament orientation for prime n.
    Returns list of bits or None if n is not a valid prime for Paley (n % 4 == 3).
    """
    if n < 2:
        return None
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None
    if n % 4 != 3:
        return None
        
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _orientation_to_in_bits(n, orientation):
    """Convert orientation list to in_bits array (in_bits[v] has bit u set if u->v)."""
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _in_bits_to_orientation(n, in_bits):
    """Convert in_bits array to orientation list."""
    orientation = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (in_bits[j] >> i) & 1:
                orientation[idx] = 0
            else:
                orientation[idx] = 1
            idx += 1
    return orientation

def _validate_tournament(n, in_bits):
    """Check if every 4-set has a common dominator."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        return False
    return True

def _count_bad_sets(n, in_bits):
    """Count the number of 4-sets that do NOT have a common dominator."""
    bad = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        bad += 1
    return bad

def _local_search(n, in_bits, deadline):
    """
    Local search: try flipping individual edges to reduce the number of bad 4-sets.
    Uses a greedy hill-climbing approach with random restarts.
    """
    start_time = time.time()
    best_n = n
    best_in_bits = in_bits[:]
    best_bad = _count_bad_sets(n, in_bits)
    
    if best_bad == 0:
        return n, in_bits
    
    # Try to reduce bad sets by flipping edges
    # For each edge (i, j), compute the delta in bad sets if we flip it
    # This is expensive, so we do a random greedy approach
    
    max_iter = 5000
    iter_count = 0
    
    while time.time() < deadline - 5 and iter_count < max_iter:
        iter_count += 1
        current_bad = best_bad
        
        # Try random edge flips
        improved = False
        for _ in range(50):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Flip edge (i, j)
            # Currently: if in_bits[j] has bit i, then i->j, else j->i
            had_i_to_j = (in_bits[j] >> i) & 1
            
            if had_i_to_j:
                # i->j, flip to j->i
                in_bits[j] &= ~(1 << i)
                in_bits[i] |= (1 << j)
            else:
                # j->i, flip to i->j
                in_bits[i] &= ~(1 << j)
                in_bits[j] |= (1 << i)
            
            new_bad = _count_bad_sets(n, in_bits)
            
            if new_bad < current_bad:
                current_bad = new_bad
                improved = True
                if current_bad == 0:
                    break
            else:
                # Revert
                if had_i_to_j:
                    in_bits[j] |= (1 << i)
                    in_bits[i] &= ~(1 << j)
                else:
                    in_bits[i] |= (1 << j)
                    in_bits[j] &= ~(1 << i)
        
        if improved and current_bad < best_bad:
            best_bad = current_bad
            best_in_bits = in_bits[:]
            if best_bad == 0:
                break
        elif not improved:
            # No improvement in this round, try a few more random flips
            for _ in range(10):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                had_i_to_j = (in_bits[j] >> i) & 1
                if had_i_to_j:
                    in_bits[j] &= ~(1 << i)
                    in_bits[i] |= (1 << j)
                else:
                    in_bits[i] &= ~(1 << j)
                    in_bits[j] |= (1 << i)
                
                new_bad = _count_bad_sets(n, in_bits)
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_in_bits = in_bits[:]
                    if best_bad == 0:
                        return best_n, best_in_bits
                else:
                    # Revert
                    if had_i_to_j:
                        in_bits[j] |= (1 << i)
                        in_bits[i] &= ~(1 << j)
                    else:
                        in_bits[i] |= (1 << j)
                        in_bits[j] &= ~(1 << i)
    
    return best_n, best_in_bits

def _greedy_prune(n, in_bits, deadline):
    """
    Greedy set-cover pruning using monotonicity:
    If a tournament satisfies the property, any induced sub-tournament also does.
    So vertex v can be removed iff v is not the unique dominator of any 4-set.
    Returns (pruned_n, pruned_in_bits).
    """
    start_time = time.time()
    current_n = n
    current_in_bits = in_bits[:]
    
    while time.time() < deadline - 3:
        current_n = len(current_in_bits)
        
        if current_n <= 5:
            break
        
        # Count for each vertex how many 4-sets it is the UNIQUE dominator for
        unique_count = [0] * current_n
        
        for a in range(current_n):
            for b in range(a + 1, current_n):
                for c in range(b + 1, current_n):
                    for d in range(c + 1, current_n):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        common = current_in_bits[a] & current_in_bits[b] & current_in_bits[c] & current_in_bits[d]
                        dom = common & ~mask
                        if dom != 0 and (dom & (dom - 1)) == 0:
                            v = dom.bit_length() - 1
                            if v < current_n:
                                unique_count[v] += 1
        
        # Find vertices with unique_count == 0
        removable = [v for v in range(current_n) if unique_count[v] == 0]
        
        if not removable:
            break
        
        # Remove the first removable vertex
        v = removable[0]
        
        # Build new in_bits without v
        new_n = current_n - 1
        if new_n < 5:
            break
            
        new_in_bits = [0] * new_n
        for old_i in range(current_n):
            if old_i == v:
                continue
            new_i = old_i if old_i < v else old_i - 1
            new_bits = 0
            for old_j in range(current_n):
                if old_j == v:
                    continue
                if (current_in_bits[old_i] >> old_j) & 1:
                    new_j = old_j if old_j < v else old_j - 1
                    new_bits |= (1 << new_j)
            new_in_bits[new_i] = new_bits
        
        current_n = new_n
        current_in_bits = new_in_bits
    
    return current_n, current_in_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with Paley(67)
    orientation = _get_paleyn_orientation(67)
    if orientation is None:
        n = 83
        orientation = _get_paleyn_orientation(83)
        if orientation is None:
            n = 90
            orientation = [0] * (n * (n - 1) // 2)
        return {'n': n, 'orientation': orientation}
    
    n = 67
    in_bits = _orientation_to_in_bits(n, orientation)
    
    # Validate Paley(67)
    if not _validate_tournament(n, in_bits):
        return {'n': n, 'orientation': orientation}
    
    # Try greedy pruning
    n_pruned, in_bits_pruned = _greedy_prune(n, in_bits, deadline)
    
    if n_pruned < n:
        n = n_pruned
        in_bits = in_bits_pruned
    
    # Try local search to reduce n further by checking if we can remove more vertices
    # After pruning, try to see if we can find a valid tournament on fewer vertices
    # by starting from a random tournament and using local search
    if time.time() < deadline - 20:
        # Try to find a valid tournament on n-1, n-2, ... vertices
        for target_n in range(n - 1, max(30, n - 10), -1):
            if time.time() > deadline - 20:
                break
            
            # Start with a random tournament
            random.seed(42)
            for attempt in range(20):
                if time.time() > deadline - 20:
                    break
                
                random_orientation = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
                random_in_bits = _orientation_to_in_bits(target_n, random_orientation)
                
                # Use local search to try to fix bad sets
                _, fixed_in_bits = _local_search(target_n, random_in_bits, deadline)
                
                if _validate_tournament(target_n, fixed_in_bits):
                    n = target_n
                    in_bits = fixed_in_bits
                    break
            
            if n < n_pruned:
                break
    
    orientation = _in_bits_to_orientation(n, in_bits)
    return {'n': n, 'orientation': orientation}

# EVOLVE-BLOCK-END
