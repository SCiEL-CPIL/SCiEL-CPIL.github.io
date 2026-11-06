import time
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def _paley_orientation(p):
    """Return (n, orientation list) for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient

def _orient_to_outbits(orient, n):
    """Convert orientation list to list of out-degree bitmasks (Python ints)."""
    outbits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orient[idx]
            idx += 1
            if bit == 0:
                outbits[i] |= (1 << j)
            else:
                outbits[j] |= (1 << i)
    return outbits

def _outbits_to_orient(outbits, n):
    """Convert out-degree bitmasks to orientation list."""
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            orient.append(0 if (outbits[i] >> j) & 1 else 1)
    return orient

def _count_violations(outbits, n):
    """Count violating 4-sets."""
    count = 0
    for a, b, c, d in combinations(range(n), 4):
        ma = outbits[a]
        if not ((ma >> b) & 1 and (ma >> c) & 1 and (ma >> d) & 1):
            mb = outbits[b]
            if not ((mb >> a) & 1 and (mb >> c) & 1 and (mb >> d) & 1):
                mc = outbits[c]
                if not ((mc >> a) & 1 and (mc >> b) & 1 and (mc >> d) & 1):
                    md = outbits[d]
                    if not ((md >> a) & 1 and (md >> b) & 1 and (md >> c) & 1):
                        count += 1
    return count

def _delta_violations(outbits, n, u, v):
    """
    Calculate the change in the number of violations if we flip the arc between u and v.
    Returns (delta, new_violation_count_if_flipped).
    Only 4-sets containing both u and v are affected.
    """
    # The 4-sets affected are {u, v, x, y} where x, y are distinct and != u, v.
    # There are C(n-2, 2) such sets.
    # For each set, we check if it is a violation BEFORE and AFTER the flip.
    
    # Precompute: For a set {u, v, x, y}, a vertex z in the set dominates the set 
    # if z has arcs to the other 3.
    # Note: u and v are in the set.
    
    # Let's identify the other vertices
    others = [i for i in range(n) if i != u and i != v]
    
    delta = 0
    
    # Current state: 
    # Is u -> v?
    u_to_v = (outbits[u] >> v) & 1
    
    for i in range(len(others)):
        x = others[i]
        for j in range(i+1, len(others)):
            y = others[j]
            
            # Set S = {u, v, x, y}
            
            # Check if S is a violation BEFORE flip
            is_viol_before = True
            # Check u
            if ((outbits[u] >> v) & 1) and ((outbits[u] >> x) & 1) and ((outbits[u] >> y) & 1):
                is_viol_before = False
            # Check v
            if ((outbits[v] >> u) & 1) and ((outbits[v] >> x) & 1) and ((outbits[v] >> y) & 1):
                is_viol_before = False
            # Check x
            if ((outbits[x] >> u) & 1) and ((outbits[x] >> v) & 1) and ((outbits[x] >> y) & 1):
                is_viol_before = False
            # Check y
            if ((outbits[y] >> u) & 1) and ((outbits[y] >> v) & 1) and ((outbits[y] >> x) & 1):
                is_viol_before = False
            
            # Check if S is a violation AFTER flip
            # If u_to_v was 1, it becomes 0 (v -> u). If 0, it becomes 1 (u -> v).
            new_u_to_v = 1 - u_to_v
            
            is_viol_after = True
            # Check u
            # u dominates if u->v, u->x, u->y
            if new_u_to_v and ((outbits[u] >> x) & 1) and ((outbits[u] >> y) & 1):
                is_viol_after = False
            # Check v
            # v dominates if v->u, v->x, v->y
            # v->u is 1 - new_u_to_v
            if (1 - new_u_to_v) and ((outbits[v] >> x) & 1) and ((outbits[v] >> y) & 1):
                is_viol_after = False
            # Check x
            if ((outbits[x] >> u) & 1) and ((outbits[x] >> v) & 1) and ((outbits[x] >> y) & 1):
                is_viol_after = False
            # Check y
            if ((outbits[y] >> u) & 1) and ((outbits[y] >> v) & 1) and ((outbits[y] >> x) & 1):
                is_viol_after = False
            
            if is_viol_before and not is_viol_after:
                delta -= 1
            elif not is_viol_before and is_viol_after:
                delta += 1
                
    return delta

def _flip_arc(outbits, u, v):
    """Flip the arc between u and v in-place."""
    if (outbits[u] >> v) & 1:
        outbits[u] &= ~(1 << v)
        outbits[v] |= (1 << u)
    else:
        outbits[v] &= ~(1 << u)
        outbits[u] |= (1 << v)

def _remove_vertex(outbits, n, rem):
    """Remove vertex rem, return new outbits for n-1 vertices."""
    new_n = n - 1
    new_outbits = [0] * new_n
    for i in range(n):
        if i == rem:
            continue
        i_new = i if i < rem else i - 1
        mask = outbits[i]
        new_mask = 0
        for j in range(n):
            if j == rem:
                continue
            if (mask >> j) & 1:
                j_new = j if j < rem else j - 1
                new_mask |= (1 << j_new)
        new_outbits[i_new] = new_mask
    return new_outbits

def _fast_local_search(outbits, n, time_limit=10.0, max_steps=100000):
    """
    Local search with delta calculation for speed.
    """
    start_time = time.time()
    cur = outbits[:]
    cur_viol = _count_violations(cur, n)
    
    if cur_viol == 0:
        return True, cur
    
    best = cur[:]
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.01
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random pair to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        while u == v:
            v = random.randint(0, n - 1)
        
        delta = _delta_violations(cur, n, u, v)
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < (1.0 / (1.0 + delta / T)):
            _flip_arc(cur, u, v)
            cur_viol += delta
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
            if cur_viol == 0:
                return True, cur
            
            # Cool down
            T = max(T * 0.9995, T_min)
        else:
            # Reject
            T = max(T * 0.9995, T_min)
            
    if best_viol == 0:
        return True, best
    return False, best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(42)
    
    # Step 1: Verify the known n=67 Paley tournament
    p = 67
    n67, orient67 = _paley_orientation(p)
    outbits67 = _orient_to_outbits(orient67, n67)
    
    if _count_violations(outbits67, n67) == 0:
        # It is valid. Now try to reduce n.
        pass
    else:
        # Should not happen
        return {'n': n67, 'orientation': orient67}

    # Step 2: Try to find n=66
    # Strategy: Remove each vertex from 67, then run fast local search.
    for rem in range(67):
        if time.time() > deadline - 20:
            break
        
        new_outbits = _remove_vertex(outbits67, 67, rem)
        new_n = 66
        
        if _count_violations(new_outbits, new_n) == 0:
            new_orient = _outbits_to_orient(new_outbits, new_n)
            return {'n': new_n, 'orientation': new_orient}
        
        # Run fast local search
        # We have ~20s left per attempt? No, total 120s.
        # Let's allocate 1.5s per attempt for 67 attempts = ~100s.
        success, repaired = _fast_local_search(new_outbits, new_n, time_limit=1.5, max_steps=50000)
        if success:
            new_orient = _outbits_to_orient(repaired, new_n)
            return {'n': new_n, 'orientation': new_orient}
            
    # Step 3: Try to find n=65
    # Strategy: Remove 2 vertices from 67, then run fast local search.
    # There are C(67,2) = 2211 pairs. Too many to try all with long search.
    # Try a random subset of pairs.
    if time.time() < deadline - 10:
        pairs = list(combinations(range(67), 2))
        random.shuffle(pairs)
        
        for rem1, rem2 in pairs[:50]: # Try 50 pairs
            if time.time() > deadline - 5:
                break
            
            # Remove rem1, then rem2 (adjusted)
            tmp = _remove_vertex(outbits67, 67, rem1)
            rem2_adj = rem2 if rem2 < rem1 else rem2 - 1
            if rem2_adj < 0 or rem2_adj >= 66:
                continue
                
            new_outbits = _remove_vertex(tmp, 66, rem2_adj)
            new_n = 65
            
            if _count_violations(new_outbits, new_n) == 0:
                new_orient = _outbits_to_orient(new_outbits, new_n)
                return {'n': new_n, 'orientation': new_orient}
                
            success, repaired = _fast_local_search(new_outbits, new_n, time_limit=0.5, max_steps=20000)
            if success:
                new_orient = _outbits_to_orient(repaired, new_n)
                return {'n': new_n, 'orientation': new_orient}

    # Fallback: Return 67
    return {'n': n67, 'orientation': orient67}

# EVOLVE-BLOCK-END
