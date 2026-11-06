import time
import itertools
import random

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

def _bit_count(x):
    return x.bit_count()

def _count_unique_dominators(P, n, active_mask):
    """
    Count for each vertex v how many 4-subsets S (subset of active vertices)
    have v as their unique dominator.
    """
    counts = [0] * n
    active_vertices = [i for i in range(n) if (active_mask >> i) & 1]
    m = len(active_vertices)
    
    # Iterate over all 4-subsets of active vertices
    # Using indices in the active list to avoid checking inactive bits
    for i in range(m - 3):
        s1 = active_vertices[i]
        p1 = P[s1]
        for j in range(i + 1, m - 2):
            s2 = active_vertices[j]
            p12 = p1 & P[s2]
            if p12 == 0: continue # No common predecessors for s1, s2
            for k in range(j + 1, m - 1):
                s3 = active_vertices[k]
                p123 = p12 & P[s3]
                if p123 == 0: continue
                for l in range(k + 1, m):
                    s4 = active_vertices[l]
                    doms = p123 & P[s4] & active_mask
                    # doms contains all active vertices that dominate s1, s2, s3, s4
                    # Since s1..s4 are active and in the set, they are in active_mask.
                    # But s_i cannot dominate itself, so s_i is not in P[s_i].
                    # Thus doms is disjoint from {s1, s2, s3, s4}.
                    c = _bit_count(doms)
                    if c == 1:
                        # Find the unique dominator
                        d = doms.bit_length() - 1
                        counts[d] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Start with Paley(67)
    p = 67
    orientation = _pale_tournament(p)
    n = p
    P = _orientation_to_bitsets(orientation, n)
    active_mask = (1 << n) - 1
    
    best_n = n
    best_orientation = orientation
    best_P = P
    best_mask = active_mask
    
    # Pruning loop
    while time.time() < deadline:
        counts = _count_unique_dominators(P, n, active_mask)
        
        # Find removable vertices (count == 0)
        removable = [i for i in range(n) if (active_mask >> i) & 1 and counts[i] == 0]
        
        if not removable:
            break
            
        # Pick a vertex to remove
        # Heuristic: pick one with highest out-degree in the current sub-tournament?
        # Or just random.
        v_to_remove = random.choice(removable)
        
        # Update active mask
        active_mask &= ~(1 << v_to_remove)
        n_active = _bit_count(active_mask)
        
        # Update best if improved
        if n_active < best_n:
            best_n = n_active
            best_mask = active_mask
            # We don't update best_orientation yet, we reconstruct at the end
            # Or we can just keep track of the mask and reconstruct orientation at the end
            # Reconstructing orientation from P and mask is easy.
            
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Reconstruct orientation from best state
    # We need to map the active vertices to 0..best_n-1
    active_vertices = [i for i in range(n) if (best_mask >> i) & 1]
    # Map old index to new index
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
                
    return {'n': new_n, 'orientation': new_orientation}
