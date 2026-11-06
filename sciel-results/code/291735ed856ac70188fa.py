import time
import random
import itertools

def _pale_tournament(p):
    """Construct Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j iff (j - i) % p is a square
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _random_tournament(n):
    """Construct a random tournament on n vertices."""
    orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    return orientation

def _orientation_to_bitsets(orientation, n):
    """
    Convert orientation list to list of bitsets for predecessors.
    P[v] is a bitset where bit u is 1 if u -> v.
    """
    P = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j, so i is a predecessor of j
                P[j] |= (1 << i)
            else:
                # j -> i, so j is a predecessor of i
                P[i] |= (1 << j)
    return P

def _count_unique_dominators(P, n, active_mask):
    """
    Count for each vertex v how many 4-subsets S (subset of active vertices)
    have v as their unique dominator.
    """
    counts = [0] * n
    active_vertices = [i for i in range(n) if (active_mask >> i) & 1]
    m = len(active_vertices)
    
    # Iterate over all 4-subsets of active vertices
    for i in range(m - 3):
        s1 = active_vertices[i]
        p1 = P[s1]
        for j in range(i + 1, m - 2):
            s2 = active_vertices[j]
            p12 = p1 & P[s2]
            if p12 == 0: continue
            for k in range(j + 1, m - 1):
                s3 = active_vertices[k]
                p123 = p12 & P[s3]
                if p123 == 0: continue
                for l in range(k + 1, m):
                    s4 = active_vertices[l]
                    doms = p123 & P[s4] & active_mask
                    c = doms.bit_count()
                    if c == 1:
                        d = doms.bit_length() - 1
                        counts[d] += 1
    return counts

def _reconstruct_orientation(P, active_mask, n):
    """Reconstruct orientation list from P and active_mask."""
    active_vertices = [i for i in range(n) if (active_mask >> i) & 1]
    old_to_new = {old: new for new, old in enumerate(active_vertices)}
    new_n = len(active_vertices)
    new_orientation = []
    for i in range(new_n):
        old_i = active_vertices[i]
        for j in range(i + 1, new_n):
            old_j = active_vertices[j]
            # Check direction in original P
            # old_i -> old_j if bit old_i is set in P[old_j]
            if (P[old_j] >> old_i) & 1:
                new_orientation.append(0) # old_i -> old_j
            else:
                new_orientation.append(1) # old_j -> old_i
    return new_n, new_orientation

def _prune_tournament(orientation, n, max_restarts=5, time_limit=100.0):
    """
    Attempt to prune the tournament multiple times with different random choices.
    Returns the best (smallest n) orientation found.
    """
    start_time = time.time()
    best_n = n
    best_orientation = orientation
    
    P = _orientation_to_bitsets(orientation, n)
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        active_mask = (1 << n) - 1
        current_n = n
        
        # Pruning loop
        while time.time() < start_time + time_limit:
            counts = _count_unique_dominators(P, n, active_mask)
            
            # Find removable vertices (count == 0)
            removable = [i for i in range(n) if (active_mask >> i) & 1 and counts[i] == 0]
            
            if not removable:
                break
            
            # Pick a vertex to remove
            v_to_remove = random.choice(removable)
            
            # Update active mask
            active_mask &= ~(1 << v_to_remove)
            current_n -= 1
            
            # Update best if improved
            if current_n < best_n:
                best_n = current_n
                # Reconstruct orientation for best state
                # Note: P is static, active_mask changes
                # We only reconstruct at the end or when we find a new best?
                # Reconstructing is O(n^2). Doing it every step is O(n^3) total.
                # Better to just store the mask and reconstruct at the end of the restart.
                pass
        
        # End of pruning loop for this restart
        if current_n < best_n:
            best_n = current_n
            new_n, new_orientation = _reconstruct_orientation(P, active_mask, n)
            best_orientation = new_orientation

    return best_n, best_orientation

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 90
    best_orientation = []
    
    # Strategy 1: Paley(67)
    p = 67
    paley_orientation = _pale_tournament(p)
    n_p, o_p = _prune_tournament(paley_orientation, p, max_restarts=10, time_limit=40.0)
    if n_p < best_n:
        best_n = n_p
        best_orientation = o_p
        
    if time.time() > deadline:
        return {'n': best_n, 'orientation': best_orientation}
        
    # Strategy 2: Random(83)
    n_r = 83
    random_orientation = _random_tournament(n_r)
    n_r2, o_r = _prune_tournament(random_orientation, n_r, max_restarts=10, time_limit=30.0)
    if n_r2 < best_n:
        best_n = n_r2
        best_orientation = o_r
        
    if time.time() > deadline:
        return {'n': best_n, 'orientation': best_orientation}
        
    # Strategy 3: Random(75)
    n_r3 = 75
    random_orientation3 = _random_tournament(n_r3)
    n_r4, o_r3 = _prune_tournament(random_orientation3, n_r3, max_restarts=5, time_limit=20.0)
    if n_r4 < best_n:
        best_n = n_r4
        best_orientation = o_r3
        
    return {'n': best_n, 'orientation': best_orientation}
