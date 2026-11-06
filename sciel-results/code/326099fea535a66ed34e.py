"""
Strategy for Schütte's problem k=4 (every 4 vertices have a common dominator):

1. The known valid tournament is the Paley tournament on p=67 vertices.
2. We use fast integer-bitmask verification.
3. Structural change: instead of random local search, use a vertex-removal
   strategy from the known valid 67-vertex tournament, combined with a
   greedy "dominator-aware" repair heuristic.
4. Also try: construct tournaments via a "dominator-first" greedy approach
   where we ensure certain high-outdegree vertices dominate many 4-sets.

Key insight for speed: use integer bitmasks for the adjacency sets.
For n <= 90, each row fits in a Python int. Checking if vertex v dominates
a 4-set {a,b,c,d} is: (a in Nv) and (b in Nv) and (c in Nv) and (d in Nv),
which is just checking bits in an integer.

Counting violations: A 4-set {a,b,c,d} is bad iff none of the four vertices
dominates the set. We can iterate over all C(n,4) 4-sets, but with bitmasks
the inner check is 4 bit-AND operations on integers, which is fast.

For n=66: C(66,4) = 66*65*64*63/24 = 741,435. With ~20ns per check, that's
~15ms per full count. Very fast.

The repair heuristic:
- Find a violating 4-set.
- For each of the 6 arcs in it, check if flipping reduces violations.
- Accept the best flip; if none helps, accept a random flip (simulated annealing).

But the structural change: start from the 67-vertex Paley tournament, try
removing each vertex one by one, and repair the resulting 66-vertex tournament.
This is much more likely to succeed than starting from random.
"""
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
                # i -> j
                outbits[i] |= (1 << j)
            else:
                # j -> i
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
    """Count violating 4-sets. A 4-set is bad if no vertex in it dominates the other 3."""
    count = 0
    for a, b, c, d in combinations(range(n), 4):
        # Check if a dominates b,c,d
        ma = outbits[a]
        if not ((ma >> b) & 1 and (ma >> c) & 1 and (ma >> d) & 1):
            # Check if b dominates a,c,d
            mb = outbits[b]
            if not ((mb >> a) & 1 and (mb >> c) & 1 and (mb >> d) & 1):
                # Check if c dominates a,b,d
                mc = outbits[c]
                if not ((mc >> a) & 1 and (mc >> b) & 1 and (mc >> d) & 1):
                    # Check if d dominates a,b,c
                    md = outbits[d]
                    if not ((md >> a) & 1 and (md >> b) & 1 and (md >> c) & 1):
                        count += 1
    return count

def _find_violation(outbits, n):
    """Find one violating 4-set, or None."""
    for a, b, c, d in combinations(range(n), 4):
        ma = outbits[a]
        if not ((ma >> b) & 1 and (ma >> c) & 1 and (ma >> d) & 1):
            mb = outbits[b]
            if not ((mb >> a) & 1 and (mb >> c) & 1 and (mb >> d) & 1):
                mc = outbits[c]
                if not ((mc >> a) & 1 and (mc >> b) & 1 and (mc >> d) & 1):
                    md = outbits[d]
                    if not ((md >> a) & 1 and (md >> b) & 1 and (md >> c) & 1):
                        return (a, b, c, d)
    return None

def _flip_arc(outbits, u, v):
    """Flip the arc between u and v in-place. Returns the new outbits."""
    # Determine current direction
    if (outbits[u] >> v) & 1:
        # u -> v, flip to v -> u
        outbits[u] &= ~(1 << v)
        outbits[v] |= (1 << u)
    else:
        # v -> u, flip to u -> v
        outbits[v] &= ~(1 << u)
        outbits[u] |= (1 << v)
    return outbits

def _remove_vertex(outbits, n, rem):
    """Remove vertex rem, return new outbits for n-1 vertices."""
    new_n = n - 1
    new_outbits = [0] * new_n
    for i in range(n):
        if i == rem:
            continue
        i_new = i if i < rem else i - 1
        mask = outbits[i]
        # Build new mask: for each j != rem, map to new index
        new_mask = 0
        for j in range(n):
            if j == rem:
                continue
            if (mask >> j) & 1:
                j_new = j if j < rem else j - 1
                new_mask |= (1 << j_new)
        new_outbits[i_new] = new_mask
    return new_outbits

def _local_search(outbits, n, time_limit=10.0, max_steps=50000):
    """
    Local search with simulated annealing to minimize violations.
    Returns (success, outbits).
    """
    start_time = time.time()
    cur = outbits[:]
    cur_viol = _count_violations(cur, n)
    
    if cur_viol == 0:
        return True, cur
    
    best = cur[:]
    best_viol = cur_viol
    
    # Temperature schedule
    T = float(cur_viol)
    T_min = 0.1
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Find a violation to guide the flip
        viol = _find_violation(cur, n)
        if viol is None:
            return True, cur
        
        a, b, c, d = viol
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        
        # Try all 6 flips in the violating set
        best_flip = None
        best_new_viol = cur_viol
        
        for u, v in pairs:
            old_u = cur[u]
            old_v = cur[v]
            _flip_arc(cur, u, v)
            new_viol = _count_violations(cur, n)
            # Restore
            cur[u] = old_u
            cur[v] = old_v
            
            if new_viol < best_new_viol:
                best_new_viol = new_viol
                best_flip = (u, v)
        
        if best_flip is not None:
            u, v = best_flip
            _flip_arc(cur, u, v)
            cur_viol = best_new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
            if cur_viol == 0:
                return True, cur
        else:
            # No improving move; accept a random flip with probability based on temperature
            u, v = random.choice(pairs)
            old_u = cur[u]
            old_v = cur[v]
            _flip_arc(cur, u, v)
            new_viol = _count_violations(cur, n)
            
            # Accept if better, or with probability exp(-(new_viol - cur_viol) / T)
            delta = new_viol - cur_viol
            if delta < 0 or random.random() < (T / (T + abs(delta) + 1e-10)):
                cur_viol = new_viol
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best = cur[:]
                if cur_viol == 0:
                    return True, cur
            else:
                # Reject: restore
                cur[u] = old_u
                cur[v] = old_v
            
            # Cool down
            T = max(T * 0.999, T_min)
        
        # Periodic restart: if stuck, try a few random flips
        if step % 5000 == 4999 and cur_viol > 0:
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    _flip_arc(cur, u, v)
            cur_viol = _count_violations(cur, n)
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
    
    if best_viol == 0:
        return True, best
    return False, best

def _greedy_dominator_tournament(n, time_limit=5.0):
    """
    Greedy construction: build a tournament by ensuring high-outdegree vertices
    dominate many 4-sets. Start with a random tournament, then for each vertex
    in order of decreasing out-degree, try to increase its out-degree by
    flipping arcs, as long as it doesn't create new violations.
    """
    start_time = time.time()
    
    # Start with random tournament
    outbits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                outbits[i] |= (1 << j)
            else:
                outbits[j] |= (1 << i)
    
    cur_viol = _count_violations(outbits, n)
    if cur_viol == 0:
        return True, outbits
    
    best = outbits[:]
    best_viol = cur_viol
    
    # Greedy: for each vertex, try to increase its out-degree
    for _ in range(100):  # iterations
        if time.time() - start_time > time_limit:
            break
        
        # Sort vertices by current out-degree descending
        outdegs = [bin(outbits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -outdegs[i])
        
        for v in order:
            if time.time() - start_time > time_limit:
                break
            
            # Try flipping arcs where v is losing to gain
            for u in range(n):
                if u == v:
                    continue
                if (outbits[u] >> v) & 1:
                    # u -> v, try flipping to v -> u
                    old_u = outbits[u]
                    old_v = outbits[v]
                    _flip_arc(outbits, u, v)
                    new_viol = _count_violations(outbits, n)
                    if new_viol > cur_viol:
                        # Restore
                        outbits[u] = old_u
                        outbits[v] = old_v
                    else:
                        cur_viol = new_viol
                        if cur_viol < best_viol:
                            best_viol = cur_viol
                            best = outbits[:]
                        if cur_viol == 0:
                            return True, outbits
    
    if best_viol == 0:
        return True, best
    return False, best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    random.seed(42)  # Reproducibility
    
    # Step 1: Verify the known n=67 Paley tournament
    p = 67
    n67, orient67 = _paley_orientation(p)
    outbits67 = _orient_to_outbits(orient67, n67)
    
    # Quick verification
    viol67 = _find_violation(outbits67, n67)
    if viol67 is not None:
        # Paley 67 should be valid; if not, try other primes
        for p in [71, 79, 83, 89]:
            n, orient = _paley_orientation(p)
            ob = _orient_to_outbits(orient, n)
            if _find_violation(ob, n) is None:
                return {'n': n, 'orientation': orient}
        # Fallback: return the 67-vertex one even if not verified
        return {'n': n67, 'orientation': orient67}
    
    # Step 2: Try vertex-removal from the 67-vertex tournament
    # Remove each vertex one by one and repair
    for rem in range(67):
        if time.time() > deadline - 40:
            break
        
        new_outbits = _remove_vertex(outbits67, 67, rem)
        new_n = 66
        
        # Check if already valid
        if _find_violation(new_outbits, new_n) is None:
            new_orient = _outbits_to_orient(new_outbits, new_n)
            return {'n': new_n, 'orientation': new_orient}
        
        # Local search to repair
        success, repaired = _local_search(new_outbits, new_n, time_limit=8.0, max_steps=20000)
        if success:
            new_orient = _outbits_to_orient(repaired, new_n)
            return {'n': new_n, 'orientation': new_orient}
    
    # Step 3: Try greedy dominator tournament for n=66
    if time.time() < deadline - 20:
        success, result = _greedy_dominator_tournament(66, time_limit=10.0)
        if success:
            new_orient = _outbits_to_orient(result, 66)
            return {'n': 66, 'orientation': new_orient}
    
    # Step 4: Try local search from random starts for n=66
    if time.time() < deadline - 15:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            rand_outbits = [0] * 66
            for i in range(66):
                for j in range(i + 1, 66):
                    if random.random() < 0.5:
                        rand_outbits[i] |= (1 << j)
                    else:
                        rand_outbits[j] |= (1 << i)
            
            success, repaired = _local_search(rand_outbits, 66, time_limit=5.0, max_steps=10000)
            if success:
                new_orient = _outbits_to_orient(repaired, 66)
                return {'n': 66, 'orientation': new_orient}
    
    # Step 5: Try to remove 2 vertices from 67 (giving 65)
    if time.time() < deadline - 30:
        for rem1 in range(67):
            if time.time() > deadline - 20:
                break
            for rem2 in range(rem1 + 1, 67):
                if time.time() > deadline - 15:
                    break
                
                # Remove rem1 first, then rem2
                tmp = _remove_vertex(outbits67, 67, rem1)
                # Now remove rem2 from the 66-vertex tournament
                # But rem2 might have shifted
                rem2_adjusted = rem2 if rem2 < rem1 else rem2 - 1
                if rem2_adjusted >= 0 and rem2_adjusted < 66:
                    new_outbits = _remove_vertex(tmp, 66, rem2_adjusted)
                    new_n = 65
                    
                    if _find_violation(new_outbits, new_n) is None:
                        new_orient = _outbits_to_orient(new_outbits, new_n)
                        return {'n': new_n, 'orientation': new_orient}
                    
                    success, repaired = _local_search(new_outbits, new_n, time_limit=3.0, max_steps=5000)
                    if success:
                        new_orient = _outbits_to_orient(repaired, new_n)
                        return {'n': new_n, 'orientation': new_orient}
    
    # Step 6: Try n=65 with greedy
    if time.time() < deadline - 15:
        success, result = _greedy_dominator_tournament(65, time_limit=8.0)
        if success:
            new_orient = _outbits_to_orient(result, 65)
            return {'n': 65, 'orientation': new_orient}
    
    # Final fallback: return the verified 67-vertex Paley tournament
    return {'n': n67, 'orientation': orient67}

# EVOLVE-BLOCK-END
