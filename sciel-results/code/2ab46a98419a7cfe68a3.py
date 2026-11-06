import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """Fast greedy growth to initialize."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _rotate_segment(segment, pivot, axis, direction, dimension):
    """
    Rotate segment (list of tuples) around pivot by 90 degrees.
    Returns new list of tuples.
    """
    new_segment = []
    for p in segment:
        v = [p[i] - pivot[i] for i in range(dimension)]
        # Rotate v
        if axis == 0:
            if direction == 1:
                # x, -z, y
                v_new = [v[0], -v[2], v[1]]
            else:
                # x, z, -y
                v_new = [v[0], v[2], -v[1]]
        elif axis == 1:
            if direction == 1:
                # z, y, -x
                v_new = [v[2], v[1], -v[0]]
            else:
                # -z, y, x
                v_new = [-v[2], v[1], v[0]]
        else: # axis 2
            if direction == 1:
                # -y, x, z
                v_new = [-v[1], v[0], v[2]]
            else:
                # y, -x, z
                v_new = [v[1], -v[0], v[2]]
        
        new_p = tuple(pivot[i] + v_new[i] for i in range(dimension))
        new_segment.append(new_p)
    return new_segment

def fold(sequence, dimension):
    """
    Simulated Annealing with Pivots.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Global time budget per call.
    # The prompt says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # This implies isolation. However, to be safe against a shared process scenario,
    # we will use a conservative limit of 10 seconds per call, assuming ~26 calls.
    # 26 * 10 = 260s, leaving 40s buffer for overhead/launch.
    # If it is truly isolated, 10s is a good balance for SA.
    
    start_time = time.time()
    max_time = 10.0
    
    best_chain = None
    best_energy = 0  # 0 is unfolded (no contacts), energy is negative
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Try multiple restarts if time permits
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time * 0.95:
            break
            
        # 1. Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: simple random walk or straight line
            # Straight line is always valid but energy 0
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = chain.copy()
            best_energy = current_energy
            
        # 2. Simulated Annealing
        # We will run SA for a fixed number of steps or until time slice ends.
        # Let's allocate time per restart based on restart index.
        # First restart gets more time.
        
        # Estimate steps possible.
        # Python is slow. Let's aim for ~1000-2000 steps/sec.
        # 10s * 1000 = 10000 steps total.
        # If we do 10 restarts, 1000 steps each.
        
        # Let's do 5 restarts with 2000 steps each?
        # Or 1 long run?
        # SA often benefits from long runs to escape local minima.
        # But multiple restarts help explore different basins.
        # Let's try 5 restarts.
        
        # Time per restart:
        # If restart == 0: 40% of time
        # Else: 12% of time each (4 * 12 = 48%, total 88%)
        
        if restart == 0:
            time_slice = max_time * 0.4
        else:
            time_slice = max_time * 0.12
        
        t_start = time.time()
        
        # Temperature schedule
        # E_range is roughly -N/2 to 0.
        # Let's set T_max = 5.0, T_min = 0.01
        T_max = 5.0
        T_min = 0.01
        
        # We don't know exact steps, so we run until time_slice is up.
        # But we need a step counter for T cooling.
        # Let's assume ~1000 steps/sec.
        # steps_estimate = time_slice * 1000
        # T = T_max * (1 - step/steps_estimate)
        
        # To be robust, let's just run until time is up and use a fixed large step count for T calculation,
        # or adapt T based on time elapsed.
        # Better: use a fixed number of steps for the SA loop, estimated from time.
        # If we exceed time, we break.
        
        steps_estimate = int(time_slice * 800) # Conservative estimate
        
        for step in range(steps_estimate):
            if time.time() - t_start > time_slice:
                break
                
            # Choose move: Pivot
            # Pivot index k: 1 to n-1
            # We rotate segment k..n-1 around chain[k-1]
            k = rng.randint(1, n-1)
            pivot_site = chain[k-1]
            
            # Extract segment
            segment = chain[k:]
            seg_len = len(segment)
            
            # Remove segment from occupied
            old_positions = segment
            for p in old_positions:
                del occupied[p]
            
            # Choose rotation
            # 6 possible rotations around pivot_site
            # Axis: 0, 1, 2. Step: +1, -1 (90 deg)
            axis = rng.randint(0, dimension-1)
            direction = rng.choice([-1, 1])
            
            new_segment = _rotate_segment(old_positions, pivot_site, axis, direction, dimension)
            
            # Check collision with occupied (which is R)
            valid = True
            for new_p in new_segment:
                if new_p in occupied:
                    valid = False
                    break
            
            if not valid:
                # Restore
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
                continue
            
            # Calculate Energy Delta
            # We need to check contacts between new_segment and R (indices 0..k-1)
            # And old_segment and R.
            # Internal energy of segment is invariant under rotation.
            
            # Helper to count H-H contacts between a segment (list of pos) and occupied (dict pos->idx)
            # where the segment indices start at seg_start_idx.
            # We only count contacts where the neighbor is in 'occupied' (which is R).
            
            def count_boundary_contacts(seg_positions, seg_start_idx):
                e = 0
                for i_offset, pos in enumerate(seg_positions):
                    idx = seg_start_idx + i_offset
                    if is_h[idx] == 0:
                        continue
                    for nb in _neighbours(pos):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx] == 1:
                                # Since we only look from segment to R, and R is disjoint,
                                # each pair is counted exactly once (from the segment side).
                                # We must ensure we don't count the same pair twice if we iterated both sides,
                                # but we only iterate segment side here.
                                e += 1
                return e
            
            old_contacts = count_boundary_contacts(old_positions, k)
            new_contacts = count_boundary_contacts(new_segment, k)
            
            delta_contacts = new_contacts - old_contacts
            # Energy = -contacts
            # delta_energy = -delta_contacts
            delta_energy = -delta_contacts
            
            # SA Acceptance
            if delta_energy < 0:
                accept = True
            else:
                # T is needed.
                # T = T_max * (1 - step/steps_estimate)
                # If steps_estimate is 0, T = T_max
                if steps_estimate > 0:
                    T = T_max * (1.0 - step / steps_estimate)
                else:
                    T = T_max
                if T < T_min: T = T_min
                
                try:
                    prob = math.exp(-delta_energy / T)
                except OverflowError:
                    prob = 0.0
                
                accept = rng.random() < prob
            
            if accept:
                # Update chain and occupied
                # We already removed old from occupied.
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
                # chain is already correct for old positions (we didn't change it in reject)
                pass

    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
