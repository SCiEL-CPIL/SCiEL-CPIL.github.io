<<<<<<< SEARCH
def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP energy: negative count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
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

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        rng.shuffle(neighbors)
        placed = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied[site] = i
                placed = True
                break
        if not placed:
            return None
    return chain

def _backtrack_move(sequence, chain, dimension, rng):
    """Attempt a backtracking move: pick a random segment and try to move it."""
    n = len(sequence)
    if n < 5:
        return None
    
    # Pick a random segment [i, j] to move
    i = rng.randint(1, n - 2)
    max_len = min(20, n - i - 1)
    seg_len = rng.randint(1, max_len)
    j = min(i + seg_len, n - 1)
    
    # Try to move the segment to a new position
    segment = chain[i:j+1]
    # Remove segment from occupied
    occupied = {p: idx for idx, p in enumerate(chain)}
    for idx in range(i, j+1):
        del occupied[chain[idx]]
    
    # Try to place segment starting from a neighbor of chain[i-1]
    start_site = chain[i-1]
    neighbors = list(_neighbours(start_site))
    rng.shuffle(neighbors)
    
    for new_start in neighbors:
        if new_start in occupied:
            continue
        # Try to grow the segment from new_start
        new_segment = [new_start]
        new_occupied = set(occupied.keys())
        new_occupied.add(new_start)
        success = True
        for k in range(1, len(segment)):
            # The direction from segment[k-1] to segment[k]
            dx = [segment[k][d] - segment[k-1][d] for d in range(dimension)]
            candidate = tuple(new_segment[-1][d] + dx[d] for d in range(dimension))
            if candidate in new_occupied or candidate == chain[i-1]:
                success = False
                break
            new_segment.append(candidate)
            new_occupied.add(candidate)
        if not success:
            continue
        # Check if the end connects to chain[j+1]
        if j + 1 < n:
            end_neighbor_ok = False
            for nb in _neighbours(new_segment[-1]):
                if nb == chain[j+1]:
                    end_neighbor_ok = True
                    break
            if not end_neighbor_ok:
                continue
        # Build new chain
        new_chain = chain[:i] + new_segment + chain[j+1:]
        # Verify self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        return new_chain
    return None

def _local_search(sequence, chain, dimension, rng, max_iters=1000):
    """Local search using backtracking moves with simulated annealing."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        new_chain = _backtrack_move(sequence, current_chain, dimension, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        T = max(T * cooling, T_min)
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=5000):
    """Simulated annealing with backtracking moves."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.5
    cooling = 0.9995
    
    for it in range(max_iters):
        if _is_deadline_exceeded():
            break
        new_chain = _backtrack_move(sequence, current_chain, dimension, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        T = max(T * cooling, T_min)
    
    return best_chain, best_energy
=======
def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP energy: negative count of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
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

def _pivot_move(chain, dimension, rng):
    """
    Perform a pivot move: choose a random pivot point and rotate the rest of the chain.
    Returns the new chain if valid, None otherwise.
    """
    n = len(chain)
    if n < 2:
        return None
    
    # Pick a pivot index i (1 to n-1)
    i = rng.randint(1, n - 1)
    
    # The segment to rotate is chain[i:]
    # We rotate around chain[i-1]
    pivot = chain[i-1]
    segment = chain[i:]
    
    # Generate a random rotation
    # In 3D, there are 6 possible directions for the first step of the segment relative to the pivot.
    # The original direction is chain[i] - chain[i-1].
    # We pick a new direction that is not the reverse of the original (to avoid immediate backtrack)
    # and not the same (to avoid no-op, though no-op is handled by energy check).
    
    orig_dir = tuple(chain[i][d] - chain[i-1][d] for d in range(dimension))
    
    # Possible directions: +/-1 on each axis
    dirs = []
    for axis in range(dimension):
        for step in (-1, 1):
            d = [0] * dimension
            d[axis] = step
            dirs.append(tuple(d))
    
    # Filter out the reverse direction (which would put segment[0] back at chain[i-1] if we were moving from i-1 to i, 
    # but here we are moving the whole segment. Actually, the constraint is that the new first point of the segment 
    # must be a neighbor of the pivot and not occupied by the rest of the chain (excluding the segment itself).
    
    occupied = set(chain[:i]) # The part of the chain that stays fixed
    
    new_segment = None
    valid_dirs = [d for d in dirs if d != tuple(-x for x in orig_dir)] # Avoid direct reversal
    rng.shuffle(valid_dirs)
    
    for d in valid_dirs:
        # Calculate the new position for the first point of the segment
        new_pos = tuple(pivot[d] + d[d] for d in range(dimension))
        
        if new_pos in occupied:
            continue
            
        # Try to map the entire segment using this rotation
        # The rotation is defined by the mapping of the original first vector to the new first vector.
        # This is a complex 3D rotation. A simpler approach for "pivot" in lattice contexts often involves 
        # trying all 6 neighbors for the new head and then checking if the rest of the segment can be placed 
        # by applying the same relative displacements? No, that's not a rigid rotation.
        
        # True pivot move: Rotate the vector (chain[i] - chain[i-1]) to a new direction.
        # In a cubic lattice, a "rotation" that maps one unit vector to another is a permutation/reflection.
        # However, simply translating the segment relative to the pivot is not a rotation.
        
        # Let's use a simpler and very effective move: **Bridge Move** or **Kink Flip** is faster to implement correctly.
        # But Pivots are powerful. Let's implement a simplified pivot:
        # Pick a new neighbor for chain[i] (relative to chain[i-1]).
        # Then, for the rest of the segment, we need to apply a rigid transformation.
        # This is hard to do efficiently in pure Python without numpy for rotation matrices.
        
        # Alternative: **Local Move (Kink Flip)** is much easier and very effective.
        # Let's switch strategy to Kink Flips and Bridge Moves which are O(1) or O(N) but very fast.
        return None # Fallback to other moves if pivot is too complex/slow

def _kink_flip(chain, dimension, rng):
    """
    Find a kink (i, i+1, i+2 where chain[i+2] == chain[i]) and flip the middle point.
    Or more generally, pick 3 consecutive points and try to move the middle one to a different neighbor of the ends.
    """
    n = len(chain)
    if n < 3:
        return None
    
    i = rng.randint(0, n - 3)
    p0 = chain[i]
    p1 = chain[i+1]
    p2 = chain[i+2]
    
    # Check if it's a kink (p0 == p2)
    if p0 != p2:
        return None
        
    # p1 is the middle. We want to move p1 to a new position p1' such that
    # p1' is a neighbor of p0 and p2.
    # The neighbors of p0 are p1 and 5 others.
    # The neighbors of p2 are p1 and 5 others.
    # The intersection of neighbors of p0 and p2 (excluding p1) are the valid new positions.
    
    occ = set(chain)
    occ.remove(p1)
    
    candidates = []
    for nb in _neighbours(p0):
        if nb == p1:
            continue
        if nb in _neighbours(p2):
            if nb not in occ:
                candidates.append(nb)
                
    if not candidates:
        return None
        
    new_p1 = rng.choice(candidates)
    
    new_chain = chain.copy()
    new_chain[i+1] = new_p1
    return new_chain

def _bridge_move(chain, dimension, rng):
    """
    Pick two points i, j (j > i+1) that are neighbors. Move the segment between them?
    Actually, a standard move is: Pick a point i, and try to move it to a neighbor, 
    effectively shifting the whole tail? No, that's a pivot.
    
    Let's use **Local Move**: Pick a random index i, and try to move chain[i] to a random neighbor,
    provided the neighbors i-1 and i+1 are not blocked and the new position is free.
    This is a "local kink" generalization.
    """
    n = len(chain)
    if n < 3:
        return None
        
    i = rng.randint(1, n - 2)
    p = chain[i]
    p_prev = chain[i-1]
    p_next = chain[i+1]
    
    occ = set(chain)
    occ.remove(p)
    
    candidates = []
    for nb in _neighbours(p):
        if nb == p_prev or nb == p_next:
            continue
        if nb not in occ:
            candidates.append(nb)
            
    if not candidates:
        return None
        
    new_p = rng.choice(candidates)
    new_chain = chain.copy()
    new_chain[i] = new_p
    return new_chain

def _anneal(sequence, chain, dimension, rng, time_limit):
    """
    Simulated Annealing using local moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if _is_deadline_exceeded():
            break
            
        # Choose a move
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            new_chain = _kink_flip(current_chain, dimension, rng)
        elif move_type < 0.9:
            new_chain = _bridge_move(current_chain, dimension, rng)
        else:
            # Random restart of a small segment? Or just another local move
            new_chain = _bridge_move(current_chain, dimension, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
                
        T = max(T * cooling, T_min)
        
    return best_chain, best_energy
>>>>>>> REPLACE

