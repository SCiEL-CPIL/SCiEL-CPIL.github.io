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

def _remove_vertex(P, n, active_mask, v):
    """Remove vertex v from active set, return new active_mask and n_active."""
    new_mask = active_mask & ~(1 << v)
    return new_mask, new_mask.bit_count()

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Paley(67)
    p = 67
    orientation = _pale_tournament(p)
    n = p
    P = _orientation_to_bitsets(orientation, n)
    active_mask = (1 << n) - 1
    
    best_n = n
    best_mask = active_mask
    
    # Pruning loop
    while time.time() < deadline:
        counts = _count_unique_dominators(P, n, active_mask)
        
        # Find removable vertices (count == 0)
        removable = [i for i in range(n) if (active_mask >> i) & 1 and counts[i] == 0]
        
        if not removable:
            break
        
        # Pick a vertex to remove
        v_to_remove = random.choice(removable)
        
        # Update active mask
        active_mask, n_active = _remove_vertex(P, n, active_mask, v_to_remove)
        
        # Update best if improved
        if n_active < best_n:
            best_n = n_active
            best_mask = active_mask
            
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Reconstruct orientation from best state
    new_n, new_orientation = _reconstruct_orientation(P, best_mask, n)
    
    return {'n': new_n, 'orientation': new_orientation}
