# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Standard energy calculation for verification/fallback."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """
    Randomized greedy chain growth.
    We use a mix of contact-seeking and random moves to generate diverse starting points.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Calculate gain: H-H contacts formed
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            
            # Free space: number of empty neighbors
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: Strong bias to contacts, weak bias to free space, plus noise
            # Noise helps escape local minima in the growth process itself
            score = gain * 10.0 + free * 0.5 + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _mc_optimize(sequence, chain, dimension, rng, steps, deadline):
    """
    Monte Carlo optimization with incremental energy updates.
    Uses pivot moves and point moves.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 4:
        return chain

    # Pre-compute H mask for faster checks
    is_h = [1 if c == 'H' else 0 for c in sequence]
    
    # Current state
    curr_chain = list(chain)
    curr_energy = _energy(sequence, curr_chain)
    best_chain = list(curr_chain)
    best_energy = curr_energy
    
    # Occupied map for fast collision checks
    occ = {p: i for i, p in enumerate(curr_chain)}
    
    # Helper to calculate delta energy for a move
    # We need to know which contacts are broken and which are formed.
    # For a point move i:
    # 1. Remove i from old position. Check neighbors of old pos. If they are H and i is H, we lose contacts.
    # 2. Place i at new position. Check neighbors of new pos. If they are H and i is H, we gain contacts.
    
    def point_move_delta(i, old_pos, new_pos):
        delta = 0
        if is_h[i]:
            # Losing contacts from old_pos
            for nb in _neighbours(old_pos):
                j = occ.get(nb)
                if j is not None and j != i and abs(i - j) > 1 and is_h[j]:
                    delta -= 1 # We lose this contact (energy goes up, so delta is negative in terms of 'goodness' if we define delta as new-old? 
                               # Wait. Energy = -contacts.
                               # If we lose a contact, contacts decrease, Energy increases (becomes less negative).
                               # So Delta Energy = +1.
                               # Let's stick to Delta Energy.
                               # If contact lost: Energy increases by 1. Delta = +1.
                    delta += 1
            
            # Gaining contacts from new_pos
            for nb in _neighbours(new_pos):
                j = occ.get(nb)
                if j is not None and j != i and abs(i - j) > 1 and is_h[j]:
                    # Note: We must ensure the neighbor is still there. 
                    # In a point move, only i moves. So occ for others is valid.
                    delta -= 1 # Gain contact -> Energy decreases. Delta = -1.
        return delta

    # Pivot move is harder to incrementalize because many residues move.
    # For pivot, we usually just recalculate energy O(N) if the move is valid (no collisions).
    # Given N <= 136, O(N) is very fast.
    # However, collision checking for pivot is O(N) as well.
    # So total cost per pivot trial is O(N).
    # Point move trial is O(1) for delta (if we check neighbors), but O(N) for collision check if we use set?
    # Actually, for point move:
    # 1. Check if new_pos is in occ. If yes, invalid. O(1).
    # 2. If valid, apply move. Update occ. O(1).
    # 3. Calculate delta. O(1) (6 neighbors).
    # So point moves are very fast.
    
    # Strategy: Mix point moves (fast) and pivot moves (robust).
    # Since pivot moves change the topology significantly, they are good for escaping local minima.
    # Point moves are good for fine-tuning.
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Point Move
            # Pick a random index i (not 0 or n-1 to keep ends fixed? No, usually 1..n-2)
            i = rng.randint(1, n - 2)
            old_pos = curr_chain[i]
            
            # Pick a random neighbor of old_pos
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            
            moved = False
            for new_pos in neighbors:
                if new_pos in occ:
                    continue
                
                # Check if the move is valid (no collision)
                # Since we remove old_pos first, we only need to check if new_pos is free.
                # But we must ensure the bond to i-1 and i+1 is preserved?
                # Point move in lattice protein:
                # Residue i moves to a neighbor of its current position.
                # This breaks bonds (i-1, i) and (i, i+1) and creates new ones.
                # Wait, a "point move" usually means moving a single residue to an adjacent empty site.
                # This changes the chain connectivity!
                # Standard HP lattice moves:
                # 1. Pivot: Rotate tail.
                # 2. Point: Move a single residue to an adjacent site.
                #    This is only valid if the new position is adjacent to BOTH i-1 and i+1?
                #    No, that's a "kink" move or specific local move.
                #    If i moves to a neighbor, the bonds (i-1, i) and (i, i+1) might break.
                #    The problem states: "unit steps between consecutive residues".
                #    So, if i moves, it MUST remain adjacent to i-1 and i+1.
                #    This is very restrictive. The only valid point moves are those where the new position
                #    is adjacent to both previous and next residues.
                #    This is rare.
                
                # Let's check validity:
                prev_pos = curr_chain[i-1]
                next_pos = curr_chain[i+1]
                
                dist_prev = sum(abs(new_pos[k] - prev_pos[k]) for k in range(dimension))
                dist_next = sum(abs(new_pos[k] - next_pos[k]) for k in range(dimension))
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                # Valid move!
                # Calculate Delta Energy
                delta = point_move_delta(i, old_pos, new_pos)
                
                # Acceptance criterion: Metropolis or Greedy?
                # Greedy is safer for time limits.
                # If delta < 0, accept.
                # If delta == 0, accept with some probability (to allow diffusion).
                # If delta > 0, reject (for greedy).
                
                if delta < 0 or (delta == 0 and rng.random() < 0.5):
                    # Apply move
                    del occ[old_pos]
                    curr_chain[i] = new_pos
                    occ[new_pos] = i
                    curr_energy += delta
                    moved = True
                    break
            
            if moved and curr_energy < best_energy:
                best_energy = curr_energy
                best_chain = list(curr_chain)
                
        else:
            # Pivot Move
            # Pick pivot index k (1..n-2)
            k = rng.randint(1, n - 2)
            p_prev = curr_chain[k-1]
            p_pivot = curr_chain[k]
            
            # Bond axis
            v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
            bond_axis = -1
            for i in range(dimension):
                if v[i] != 0:
                    bond_axis = i
                    break
            
            if bond_axis == -1:
                continue
                
            rot_axes = [i for i in range(dimension) if i != bond_axis]
            if not rot_axes:
                continue
                
            r_axis = rng.choice(rot_axes)
            angle = rng.choice([1, -1])
            
            # Rotate tail [k+1..n-1] around axis r_axis passing through p_pivot
            # Note: p_pivot is the center of rotation.
            
            def rotate_point(point, center, axis, ang):
                p = list(point)
                c = list(center)
                p[0] -= c[0]
                p[1] -= c[1]
                p[2] -= c[2]
                
                if axis == 0: # X
                    if ang == 1:
                        p[1], p[2] = -p[2], p[1]
                    else:
                        p[1], p[2] = p[2], -p[1]
                elif axis == 1: # Y
                    if ang == 1:
                        p[0], p[2] = p[2], -p[0]
                    else:
                        p[0], p[2] = -p[2], p[0]
                elif axis == 2: # Z
                    if ang == 1:
                        p[0], p[1] = -p[1], p[0]
                    else:
                        p[0], p[1] = p[1], -p[0]
                
                p[0] += c[0]
                p[1] += c[1]
                p[2] += c[2]
                return tuple(p)
            
            # Check validity
            head_set = set(curr_chain[:k]) # Exclude pivot k? 
            # The tail starts at k+1.
            # The new tail points must not collide with head [0..k-1] or each other.
            # Also, the first new point (k+1) must be adjacent to p_pivot (k).
            # This is guaranteed by rotation if the original was adjacent?
            # Original curr_chain[k+1] was adjacent to curr_chain[k].
            # Rotation preserves distance. So dist(new_k+1, p_pivot) = dist(old_k+1, p_pivot) = 1.
            # So the bond is preserved.
            
            valid = True
            new_tail = []
            occupied_tail = set()
            
            for i in range(k + 1, n):
                orig = curr_chain[i]
                new_p = rotate_point(orig, p_pivot, r_axis, angle)
                
                if new_p in head_set:
                    valid = False
                    break
                if new_p in occupied_tail:
                    valid = False
                    break
                
                # Check self-avoidance of the new tail itself is handled by occupied_tail.
                # Check connectivity?
                # Rotation preserves relative positions, so steps are unit.
                # We just need to ensure no collision with head or existing new tail.
                
                occupied_tail.add(new_p)
                new_tail.append(new_p)
                
            if not valid:
                continue
            
            # Construct candidate
            candidate = curr_chain[:k+1] + new_tail
            
            # Calculate energy (O(N))
            new_energy = _energy(sequence, candidate)
            
            if new_energy < curr_energy or (new_energy == curr_energy and rng.random() < 0.5):
                # Apply move
                curr_chain = candidate
                curr_energy = new_energy
                occ = {p: i for i, p in enumerate(curr_chain)}
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_chain = list(curr_chain)
                    
    return best_chain

def fold(sequence, dimension):
    """
    Parallel multi-start Monte Carlo.
    """
    rng = random.Random(12345 + len(sequence))
    start_time = time.time()
    deadline = start_time + 280 # Safety margin for 300s limit
    
    best_chain = None
    best_energy = 0 # Maximize energy (minimize -energy)
    
    # Number of restarts
    # We don't know how long each optimization takes.
    # We will run restarts until deadline is approached.
    # Each restart: Grow (fast) + Optimize (variable).
    
    n_restarts = 0
    while time.time() < deadline - 5:
        # Generate initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Try a few more times? Or just break?
            # If growth fails, it's likely a very tight sequence or bad luck.
            # Let's try 5 times to grow.
            success = False
            for _ in range(5):
                chain = _grow(sequence, dimension, rng)
                if chain:
                    success = True
                    break
            if not success:
                # Fallback to a straight line if we can't grow?
                # Or just return best so far.
                if best_chain is not None:
                    break
                else:
                    # Absolute fallback
                    best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
                    break
        
        n_restarts += 1
        
        # Determine optimization steps
        # Short chains: more steps? Long chains: fewer steps per restart?
        # Let's scale inversely with length to keep time constant per restart?
        # Or just fixed steps.
        # 1000 steps for N=100 is ~1ms? No, Python is slower.
        # Let's estimate.
        # 1000 steps * (O(N) for pivot check) ~ 1000 * 100 * 6 ~ 600k ops.
        # In Python, 1M ops is ~0.05-0.1s.
        # So 1000 steps ~ 0.1s.
        # We have 280s.
        # So we can do ~2000 restarts?
        # But growth is also O(N^2) roughly? No, O(N) with set lookups.
        
        # Let's use a time-based loop for optimization within a restart.
        # Or fixed steps.
        # Fixed steps is safer for determinism in logic, but time-based is better for adaptation.
        
        # Let's try 2000 steps per restart.
        # If N is small, this is fast. If N is large, it's slower.
        
        opt_steps = 2000
        # If N > 100, reduce steps to allow more restarts?
        # Let's stick to 1000 steps.
        
        optimized_chain = _mc_optimize(sequence, chain, dimension, rng, opt_steps, deadline)
        
        if optimized_chain is not None:
            e = _energy(sequence, optimized_chain)
            if best_chain is None or e < best_energy:
                best_chain = optimized_chain
                best_energy = e
                
        # Check time
        if time.time() > deadline - 10:
            break
            
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
