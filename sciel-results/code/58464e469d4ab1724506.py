import time
import random
import math

_START = time.time()
_LIMIT = 120.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _edge_idx(n, u, v):
    if u > v:
        u, v = v, u
    return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)

def _simulated_annealing(n, max_time=10.0):
    """
    Attempt to find a valid 5-coloring of K_n using Simulated Annealing.
    Energy = sum over colors c and pairs (u,v) of C(|N_c(u) intersect N_c(v)|, 2).
    """
    start_time = time.time()
    m = n * (n - 1) // 2
    
    # Initialize with random colors
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Data structures for fast updates
    # nbrs[c][u] is a set of neighbors of u in color c
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    
    # common[c][u][v] = |N_c(u) intersect N_c(v)| for u < v
    # We only need this for u < v to compute energy, but updates affect both (u,v) and (v,u)
    # Let's store intersection counts for all pairs to make updates symmetric and simple
    # common[c] is a 2D list of ints
    common = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def _build_structures():
        for c in range(5):
            for u in range(n):
                nbrs[c][u].clear()
                for v in range(n):
                    common[c][u][v] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                nbrs[c][i].add(j)
                nbrs[c][j].add(i)
                idx += 1
        
        # Calculate initial common counts
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Intersection of nbrs[c][u] and nbrs[c][v]
                    # Since n is small, we can just iterate
                    cnt = 0
                    for w in nbrs[c][u]:
                        if w in nbrs[c][v]:
                            cnt += 1
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    
    _build_structures()
    
    def _energy():
        e = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    k = common[c][u][v]
                    if k >= 2:
                        e += k * (k - 1) // 2
        return e

    current_energy = _energy()
    if current_energy == 0:
        return {'n': n, 'colors': colors}

    T = 50.0
    cooling = 0.995
    best_energy = current_energy
    best_colors = colors[:]
    
    while time.time() - start_time < max_time and _time_left() > 5:
        # Pick a random edge
        idx = random.randint(0, m - 1)
        # Decode edge
        # Inverse of _edge_idx
        # k = u * (n - 1) - u * (u +  1) // 2 + (v - u - 1)
        # This is tricky to invert directly. Let's precompute edge list.
        pass # We will use a different strategy: pick u, v randomly
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        idx = _edge_idx(n, u, v)
        c_old = colors[idx]
        
        # Pick a new color
        c_new = random.randint(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate delta energy
        # Removing edge (u,v) from c_old
        # Adding edge (u,v) to c_new
        
        delta = 0
        
        # Helper to get change in C(k,2) when k changes by dk
        def _delta_c2(k, dk):
            # C(k+dk, 2) - C(k, 2)
            # (k+dk)(k+dk-1)/2 - k(k-1)/2
            return ((k + dk) * (k + dk - 1) - k * (k - 1)) // 2
        
        # Process removal from c_old
        # u loses neighbor v in c_old
        # v loses neighbor u in c_old
        
        # For all w != u, v:
        # common[c_old][u][w] changes if v in nbrs[c_old][w]
        # common[c_old][v][w] changes if u in nbrs[c_old][w]
        
        # We need to be careful with the order of updates to not double count or miss counts.
        # The energy term for a pair (x,y) is C(common[c][x][y], 2).
        # When we remove (u,v) from c_old:
        # 1. The pair (u,v) itself: common[c_old][u][v] decreases by 1? 
        #    No, common[c_old][u][v] is the number of common neighbors. 
        #    Removing edge (u,v) does NOT change the set of common neighbors of u and v.
        #    It changes the adjacency of u and v, but the definition of common neighbors 
        #    depends on the edges (u,w) and (v,w).
        #    So common[c_old][u][v] is UNCHANGED by removing edge (u,v).
        #    Wait, the energy is sum over pairs (x,y) of C(common[c][x][y], 2).
        #    The pair (u,v) contributes C(common[c_old][u][v], 2). This value doesn't change.
        #    
        # 2. However, the sets N_c(u) and N_c(v) change.
        #    N_c_old(u) loses v. N_c_old(v) loses u.
        #    This affects common[c_old][u][w] for all w.
        #    common[c_old][u][w] = |N_c_old(u) intersect N_c_old(w)|.
        #    Since N_c_old(u) loses v, common[c_old][u][w] decreases by 1 if v in N_c_old(w).
        #    Similarly, common[c_old][v][w] decreases by 1 if u in N_c_old(w).
        #    
        #    Also, we must update the actual sets nbrs[c_old][u] and nbrs[c_old][v].
        
        # Let's collect the changes
        changes = []
        
        # Removal from c_old
        # For each w in nbrs[c_old][u] (excluding v, but v is in there):
        # If we remove v from nbrs[c_old][u], then for any w that is a neighbor of v in c_old,
        # the intersection N_c_old(u) intersect N_c_old(w) loses v.
        # So common[c_old][u][w] decreases by 1 for all w in nbrs[c_old][v].
        # Similarly, common[c_old][v][w] decreases by 1 for all w in nbrs[c_old][u].
        
        # Note: u is in nbrs[c_old][v] and v is in nbrs[c_old][u].
        # So common[c_old][u][u] is not defined/used. common[c_old][v][v] not used.
        # But common[c_old][u][v] is affected?
        # common[c_old][u][v] = |N_c_old(u) intersect N_c_old(v)|.
        # N_c_old(u) loses v. Is v in N_c_old(v)? No.
        # N_c_old(v) loses u. Is u in N_c_old(u)? No.
        # So common[c_old][u][v] is NOT affected by the removal of edge (u,v) itself?
        # Let's check.
        # N_c(u) = {w | (u,w) in E_c}.
        # If we remove (u,v), v is no longer in N_c(u).
        # Does v affect the intersection N_c(u) intersect N_c(v)?
        # v is in N_c(u) (before removal). Is v in N_c(v)? No.
        # So v was not in the intersection.
        # Similarly u was not in the intersection.
        # So common[c_old][u][v] is unchanged.
        
        # However, for any other w:
        # If v in N_c_old(w), then v was in N_c_old(u) intersect N_c_old(w).
        # After removal, v is not in N_c_old(u), so it is no longer in the intersection.
        # So common[c_old][u][w] decreases by 1.
        
        # So:
        # For all w in nbrs[c_old][v]:
        #    if w != u:
        #       common[c_old][u][w] -= 1
        #       common[c_old][w][u] -= 1
        #       update energy for pair (min(u,w), max(u,w))
        # For all w in nbrs[c_old][u]:
        #    if w != v:
        #       common[c_old][v][w] -= 1
        #       common[c_old][w][v] -= 1
        #       update energy for pair (min(v,w), max(v,w))
        
        # And we must update the sets:
        # nbrs[c_old][u].remove(v)
        # nbrs[c_old][v].remove(u)
        
        # Addition to c_new
        # Similar logic but with +1.
        # For all w in nbrs[c_new][v]:
        #    if w != u:
        #       common[c_new][u][w] += 1
        #       ...
        # For all w in nbrs[c_new][u]:
        #    if w != v:
        #       common[c_new][v][w] += 1
        #       ...
        # nbrs[c_new][u].add(v)
        # nbrs[c_new][v].add(u)
        
        # Implementation of delta calculation
        
        # 1. Removal from c_old
        # We need to iterate over neighbors BEFORE updating the sets.
        
        # To avoid double counting energy updates for the same pair (x,y), 
        # we can just calculate the new energy for the affected pairs.
        # Affected pairs in c_old: (u, w) for w in nbrs[c_old][v] and (v, w) for w in nbrs[c_old][u].
        # Note that u is in nbrs[c_old][v] and v is in nbrs[c_old][u].
        # The pair (u,v) is affected in the sense that it's in the lists, but we established 
        # common[c_old][u][v] doesn't change. 
        # Wait, if w=u in the first loop (w in nbrs[c_old][v]), we look at pair (u,u) which is invalid.
        # So we skip w=u.
        
        # Let's store the old values and new values for the affected common counts.
        # To be safe and simple, let's just recalculate the energy for the affected pairs.
        
        affected_pairs_c_old = set()
        for w in nbrs[c_old][v]:
            if w != u:
                affected_pairs_c_old.add((min(u, w), max(u, w)))
        for w in nbrs[c_old][u]:
            if w != v:
                affected_pairs_c_old.add((min(v, w), max(v, w)))
                
        # Also, the pair (u,v) itself? No, common count doesn't change.
        
        # Calculate delta for c_old
        d_old = 0
        for x, y in affected_pairs_c_old:
            old_val = common[c_old][x][y]
            # Determine new val
            # If (x,y) is (u,w), and w in nbrs[c_old][v], then common decreases by 1.
            # But what if w is in BOTH nbrs[c_old][u] and nbrs[c_old][v]?
            # i.e., w is a common neighbor of u and v.
            # Then common[c_old][u][w] decreases by 1 (because v left N_c(u)) AND 
            # common[c_old][v][w] decreases by 1 (because u left N_c(v)).
            # These are different pairs (u,w) and (v,w).
            # So we just need to know if the pair is (u,w) or (v,w).
            
            # Let's just apply the changes to common[][] and then recalculate the energy for the pair.
            # But we need the OLD energy to compute delta.
            
            # Let's compute the change in common count for this pair.
            # If x == u and y == w (where w in nbrs[c_old][v]):
            #   change = -1
            # If x == v and y == w (where w in nbrs[c_old][u]):
            #   change = -1
            # Can a pair be affected twice?
            # Pair (u, w) is affected if w in nbrs[c_old][v].
            # Pair (v, w) is affected if w in nbrs[c_old][u].
            # They are distinct pairs unless u==v (false) or w==u/v.
            
            # Let's just update the common matrix and track the delta.
            pass
        
        # This manual delta tracking is error-prone. 
        # Given n=26, we can afford to be a bit less efficient.
        # Let's just update the structures and recalculate the energy for the affected pairs.
        
        # Save old energy contributions
        old_e = 0
        for x, y in affected_pairs_c_old:
            k = common[c_old][x][y]
            old_e += k * (k - 1) // 2
            
        # Apply changes to common and nbrs for c_old
        # First, determine the exact changes
        # We know:
        # For w in nbrs[c_old][v] \ {u}: common[c_old][u][w] -= 1
        # For w in nbrs[c_old][u] \ {v}: common[c_old][v][w] -= 1
        
        # Let's apply to common matrix
        for w in list(nbrs[c_old][v]):
            if w != u:
                common[c_old][u][w] -= 1
                common[c_old][w][u] -= 1
        for w in list(nbrs[c_old][u]):
            if w != v:
                common[c_old][v][w] -= 1
                common[c_old][w][v] -= 1
                
        # Update sets
        nbrs[c_old][u].remove(v)
        nbrs[c_old][v].remove(u)
        
        # Calculate new energy contribution
        new_e = 0
        for x, y in affected_pairs_c_old:
            k = common[c_old][x][y]
            new_e += k * (k - 1) // 2
            
        delta -= (new_e - old_e)
        
        # 2. Addition to c_new
        affected_pairs_c_new = set()
        for w in nbrs[c_new][v]:
            if w != u:
                affected_pairs_c_new.add((min(u, w), max(u, w)))
        for w in nbrs[c_new][u]:
            if w != v:
                affected_pairs_c_new.add((min(v, w), max(v, w)))
                
        old_e_new = 0
        for x, y in affected_pairs_c_new:
            k = common[c_new][x][y]
            old_e_new += k * (k - 1) // 2
            
        # Apply changes
        for w in list(nbrs[c_new][v]):
            if w != u:
                common[c_new][u][w] += 1
                common[c_new][w][u] += 1
        for w in list(nbrs[c_new][u]):
            if w != v:
                common[c_new][v][w] += 1
                common[c_new][w][v] += 1
                
        # Update sets
        nbrs[c_new][u].add(v)
        nbrs[c_new][v].add(u)
        
        new_e_new = 0
        for x, y in affected_pairs_c_new:
            k = common[c_new][x][y]
            new_e_new += k * (k - 1) // 2
            
        delta += (new_e_new - old_e_new)
        
        # Accept/Reject
        if delta <= 0 or random.random() < math.exp(-delta / T):
            colors[idx] = c_new
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors[:]
                if best_energy == 0:
                    return {'n': n, 'colors': best_colors}
        else:
            # Revert
            # Revert c_new changes
            for w in list(nbrs[c_new][v]):
                if w != u:
                    common[c_new][u][w] -= 1
                    common[c_new][w][u] -= 1
            for w in list(nbrs[c_new][u]):
                if w != v:
                    common[c_new][v][w] -= 1
                    common[c_new][w][v] -= 1
            nbrs[c_new][u].remove(v)
            nbrs[c_new][v].remove(u)
            
            # Revert c_old changes
            for w in list(nbrs[c_old][v]):
                if w != u:
                    common[c_old][u][w] += 1
                    common[c_old][w][u] += 1
            for w in list(nbrs[c_old][u]):
                if w != v:
                    common[c_old][v][w] += 1
                    common[c_old][w][v] += 1
            nbrs[c_old][u].add(v)
            nbrs[c_old][v].add(u)
            
            # Note: The sets nbrs[c_old] and nbrs[c_new] were modified in the "apply" step.
            # In the revert step, I used list(nbrs[...]) which is the CURRENT state (after apply).
            # So removing/adding back should restore it.
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return {'n': n, 'colors': best_colors} if best_energy == 0 else None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try cyclic constructions for various n
    for n in [25, 20, 15, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res
    
    # Try Simulated Annealing for n=26
    if best['n'] < 26 and _time_left() > 10:
        # Run SA for a few seconds
        sa_time = min(_time_left() - 5, 30)
        res = _simulated_annealing(26, max_time=sa_time)
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res

    # If we didn't find 26, try 25?
    if best['n'] < 26 and _time_left() > 10:
        sa_time = min(_time_left() - 5, 20)
        res = _simulated_annealing(25, max_time=sa_time)
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res

    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
