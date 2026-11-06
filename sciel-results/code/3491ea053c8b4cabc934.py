import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """
    Greedy growth with some randomization to avoid getting stuck.
    Returns a list of tuples representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score the site
            # 1. Immediate H-H contact gain
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        # Check if it's a chain neighbor (i-1)
                        if idx == i - 1:
                            continue
                        if sequence[idx] == 'H':
                            gain += 1
            
            # 2. Free space (to avoid trapping)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # 3. Random tie-breaker
            rand = rng.random()
            
            # Weighted score: Gain is most important, free space second
            score = gain * 10.0 + free * 0.1 + rand
            options.append((score, site))
            
        if not options:
            # Dead end, try to backtrack? For simplicity, return None or random valid
            # In a robust implementation, we'd backtrack. Here, we just fail this init.
            return None
            
        # Pick best
        best_site = max(options)[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total energy of a chain."""
    if not chain:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, chain, k, new_segment, occupied_rest, is_h):
    """
    Calculate the change in energy when replacing segment k: of chain with new_segment.
    occupied_rest contains indices 0..k-1.
    """
    n = len(chain)
    
    # Count contacts involving the old segment (k..n-1) with the rest (0..k-1)
    # and internal contacts within the old segment
    old_contacts = 0
    
    # Internal contacts in old segment
    old_seg = chain[k:]
    for i in range(len(old_seg)):
        idx = k + i
        if is_h[idx] == 0:
            continue
        pos = old_seg[i]
        # Check neighbors within the old segment
        for j in range(i + 2, len(old_seg)):
            j_idx = k + j
            if is_h[j_idx] == 0:
                continue
            j_pos = old_seg[j]
            # Check if pos and j_pos are neighbors
            # Since they are not adjacent in chain (j > i+1), if they are neighbors, it's a contact
            # We need an efficient way to check neighbor.
            # Given the small size, simple check:
            # But doing this for all pairs is O(L^2).
            # Better: use a set of occupied positions for the old segment?
            # Actually, we can just check the 6 neighbors of pos in the global grid,
            # but we only care about those in the old segment.
            
            # Let's use a different approach for delta calculation which is faster:
            # Delta E = E_new - E_old
            # E = -Contacts
            # Delta Contacts = New Contacts - Old Contacts
            
            # Contacts are pairs (i, j) with |i-j| > 1 and dist(pos_i, pos_j) = 1.
            
            # Let's count contacts for the new segment vs old segment separately?
            # No, we need to compare the total.
            
            # Efficient way:
            # 1. Count contacts between (k..n-1) and (0..k-1) for old config.
            # 2. Count contacts within (k..n-1) for old config.
            # 3. Count contacts between new_segment and (0..k-1).
            # 4. Count contacts within new_segment.
            # 5. Delta = (3+4) - (1+2)
            
            pass # This logic is getting messy in a helper. Let's inline or simplify.

    # Let's rewrite this helper to be cleaner and correct.
    # We will count:
    # C_old_boundary: contacts between old_seg and rest
    # C_old_internal: contacts within old_seg
    # C_new_boundary: contacts between new_seg and rest
    # C_new_internal: contacts within new_seg
    
    # Helper to count contacts between a segment (list of positions) and a set of occupied positions (dict pos->idx)
    # where the segment starts at index k in the chain.
    def count_boundary(seg_positions, seg_start_idx):
        c = 0
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if is_h[idx] == 0:
                continue
            for nb in _neighbours(pos):
                if nb in occupied_rest:
                    nb_idx = occupied_rest[nb]
                    # nb_idx is in 0..k-1
                    if is_h[nb_idx] == 1:
                        c += 1
        return c

    # Helper to count internal contacts within a segment
    def count_internal(seg_positions, seg_start_idx):
        c = 0
        # Create a set of positions for fast lookup?
        # Or just check pairs.
        # Since segment length can be up to N, O(L^2) might be slow if L is large.
        # But we can use the fact that contacts are only between neighbors.
        # So for each pos in seg, check its 6 neighbors.
        # If a neighbor is also in seg and not adjacent in chain, it's a contact.
        
        # Map position to index in seg for fast lookup
        pos_to_idx_in_seg = {}
        for i, p in enumerate(seg_positions):
            pos_to_idx_in_seg[p] = i
            
        for i_offset, pos in enumerate(seg_positions):
            idx = seg_start_idx + i_offset
            if is_h[idx] == 0:
                continue
            for nb in _neighbours(pos):
                if nb in pos_to_idx_in_seg:
                    j_offset = pos_to_idx_in_seg[nb]
                    j_idx = seg_start_idx + j_offset
                    # Check if they are adjacent in chain
                    if abs(idx - j_idx) > 1:
                        if is_h[j_idx] == 1:
                            c += 1
        # Note: this counts each contact twice (once for i, once for j).
        # So we should return c // 2? 
        # Let's check: if i sees j and j sees i, c is incremented twice.
        # Yes, so return c // 2.
        return c // 2

    old_seg = chain[k:]
    
    c_old_b = count_boundary(old_seg, k)
    c_old_i = count_internal(old_seg, k)
    
    c_new_b = count_boundary(new_segment, k)
    c_new_i = count_internal(new_segment, k)
    
    old_total = c_old_b + c_old_i
    new_total = c_new_b + c_new_i
    
    delta_contacts = new_total - old_total
    delta_energy = -delta_contacts
    
    return delta_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Time budget per call
    # Total 330s for 26 sequences. 330/26 ~ 12.7s.
    # Use 11.5s to be safe.
    max_time = 11.5
    start_time = time.time()
    
    rng = random.Random(42 + len(sequence))
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0 # 0 is no contacts, energy is negative, so lower is better
    
    # Try multiple restarts
    while True:
        if time.time() - start_time > max_time:
            break
            
        # 1. Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: Straight line
            chain = [(i, 0, 0) for i in range(n)]
            
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
            
        # 2. Simulated Annealing
        # Temperature schedule
        T_max = 5.0
        T_min = 0.01
        
        # Run SA for a certain number of steps or time
        # Let's run until time slice for this restart is up
        # We don't know how many restarts we'll get, so just run until global time is up?
        # No, better to have a fixed number of steps per restart to ensure diversity.
        # But if we have time, more steps is better.
        # Let's do: If first restart, run longer. Subsequent restarts, shorter.
        
        # Estimate steps/sec. ~5000-10000 steps/sec in Python for N=100?
        # Let's just use a time check inside the loop.
        
        steps = 0
        while True:
            if time.time() - start_time > max_time * 0.95:
                break
            
            # Choose move type
            # 70% Pivot, 30% Move (Translation)
            move_type = rng.random()
            
            # Choose pivot index k (1 to n-1)
            k = rng.randint(1, n - 1)
            
            old_segment = chain[k:]
            old_positions = old_segment
            
            # Remove old segment from occupied
            for p in old_positions:
                del occupied[p]
            
            new_segment = None
            valid = True
            
            if move_type < 0.7:
                # PIVOT
                # Rotate segment k..n-1 around chain[k-1]
                pivot_site = chain[k-1]
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                
                new_segment = []
                for p in old_positions:
                    v = [p[i] - pivot_site[i] for i in range(dimension)]
                    # Rotate v by 90 deg
                    if axis == 0:
                        if direction == 1:
                            v_new = [v[0], -v[2], v[1]]
                        else:
                            v_new = [v[0], v[2], -v[1]]
                    elif axis == 1:
                        if direction == 1:
                            v_new = [v[2], v[1], -v[0]]
                        else:
                            v_new = [-v[2], v[1], v[0]]
                    else: # axis 2
                        if direction == 1:
                            v_new = [-v[1], v[0], v[2]]
                        else:
                            v_new = [v[1], -v[0], v[2]]
                    
                    new_p = tuple(pivot_site[i] + v_new[i] for i in range(dimension))
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            else:
                # MOVE (Translation)
                # Translate segment by unit vector
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                offset = [0, 0, 0]
                offset[axis] = direction
                
                new_segment = []
                for p in old_positions:
                    new_p = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
                    new_segment.append(new_p)
                    
                    if new_p in occupied:
                        valid = False
                        break
            
            if not valid:
                # Restore old
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
                continue
            
            # Calculate Energy Delta
            delta_energy = _calc_delta_energy(sequence, chain, k, new_segment, occupied, is_h)
            
            # Acceptance
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                # Temperature
                # Let's use a constant T that is moderate, e.g., 2.0, 
                # and rely on multiple restarts for exploration.
                T = 2.0 
                try:
                    prob = math.exp(-delta_energy / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                # Update state
                for p, idx in zip(new_segment, range(k, n)):
                    occupied[p] = idx
                    chain[idx] = p
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Restore old
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
            
            steps += 1
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
