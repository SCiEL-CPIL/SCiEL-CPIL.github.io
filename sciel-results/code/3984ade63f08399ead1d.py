"""HP lattice protein folding on the square lattice using CSP-guided search with
crankshaft and pivot moves for efficient local optimization."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues
    on neighbouring lattice sites. Returns a negative number (more negative = better)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1:
                contacts += 1
    return -contacts

def _incremental_energy_change(sequence, chain, i, new_pos, old_pos):
    """Compute the change in energy if residue i moves from old_pos to new_pos."""
    n = len(chain)
    where = {p: idx for idx, p in enumerate(chain)}
    
    # Remove old contacts involving residue i
    delta = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1  # removing a contact increases energy
    
    # Add new contacts involving residue i
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1  # adding a contact decreases energy
    
    return -delta  # energy change = -delta_contacts

def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy chain growth preferring sites that create H-H contacts."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Count free neighbours for flexibility
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: primarily gain, secondarily free space, tertiary random
            score = gain * 10.0 + free * 0.3 + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Select the best candidate with some randomness
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.85:
            best_site = candidates[0][1]
        else:
            best_site = rng.choice(candidates[:max(1, len(candidates) // 3)])[1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _try_crankshaft(chain, i, direction, occupied_set):
    """Try a crankshaft move: rotate a 3-residue segment around residue i.
    Returns new chain if valid, else None."""
    n = len(chain)
    if i < 2 or i > n - 3:
        return None
    
    # The segment is chain[i-1], chain[i], chain[i+1]
    # We rotate around chain[i], moving chain[i-1] and chain[i+1]
    center = chain[i]
    left = chain[i-1]
    right = chain[i+1]
    
    # Determine the direction vector from center to left
    dx = left[0] - center[0]
    dy = left[1] - center[1]
    
    # Possible rotations: 90 degrees clockwise or counter-clockwise
    if dx == 1:  # left is to the right of center
        new_left = (center[0], center[1] - 1) if direction == 1 else (center[0], center[1] + 1)
    elif dx == -1:  # left is to the left of center
        new_left = (center[0], center[1] + 1) if direction == 1 else (center[0], center[1] - 1)
    elif dy == 1:  # left is above center
        new_left = (center[0] - 1, center[1]) if direction == 1 else (center[0] + 1, center[1])
    elif dy == -1:  # left is below center
        new_left = (center[0] + 1, center[1]) if direction == 1 else (center[0] - 1, center[1])
    else:
        return None
    
    # Check if new_left is valid
    if new_left in occupied_set and new_left not in (left, right):
        return None
    
    # Check adjacency
    if abs(new_left[0] - center[0]) + abs(new_left[1] - center[1]) != 1:
        return None
    
    new_chain = list(chain)
    new_chain[i-1] = new_left
    return new_chain

def _try_pivot_move(chain, i, occupied_set):
    """Try a pivot move: rotate a segment starting at i around chain[i-1].
    Returns new chain if valid, else None."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    pivot = chain[i-1]
    seg_start = chain[i]
    
    # Determine direction from pivot to seg_start
    dx = seg_start[0] - pivot[0]
    dy = seg_start[1] - pivot[1]
    
    # Possible rotations: 90 degrees clockwise or counter-clockwise
    if dx == 1:  # segment starts to the right
        new_start = (pivot[0], pivot[1] - 1)
    elif dx == -1:  # segment starts to the left
        new_start = (pivot[0], pivot[1] + 1)
    elif dy == 1:  # segment starts above
        new_start = (pivot[0] - 1, pivot[1])
    elif dy == -1:  # segment starts below
        new_start = (pivot[0] + 1, pivot[1])
    else:
        return None
    
    # Check if new_start is valid
    if new_start in occupied_set and new_start != seg_start:
        return None
    
    # Try to rotate the entire segment
    # This is complex; for simplicity, only try single-residue pivots
    if i == n - 1:
        new_chain = list(chain)
        new_chain[i] = new_start
        return new_chain
    
    return None

def _local_search(sequence, chain, max_iters=500, rng=None, time_limit=None):
    """Perform local search using single-residue moves, crankshaft, and pivot moves."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        
        iters += 1
        improved = False
        
        # Strategy 1: Single residue moves
        for _ in range(min(5, n - 2)):
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                occupied_set = set(chain)
                if new_pos in occupied_set:
                    continue
                
                # Check adjacency to neighbors
                prev = chain[i-1]
                nxt = chain[i+1]
                if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                    continue
                if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                    continue
                
                new_chain = list(chain)
                new_chain[i] = new_pos
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    chain = new_chain
                    improved = True
                    no_improve = 0
                    break
                elif new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        # Strategy 2: Crankshaft moves
        if n >= 5 and rng.random() < 0.4:
            i = rng.randint(2, n - 3)
            for direction in [1, -1]:
                new_chain = _try_crankshaft(chain, i, direction, set(chain))
                if new_chain is not None:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        improved = True
                        no_improve = 0
                        break
                    elif new_energy < current_energy:
                        chain = new_chain
                        current_energy = new_energy
                        improved = True
                        no_improve = 0
                        break
            if improved:
                break
        
        # Strategy 3: Pivot moves
        if n >= 4 and rng.random() < 0.3:
            i = rng.randint(1, n - 1)
            new_chain = _try_pivot_move(chain, i, set(chain))
            if new_chain is not None:
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    chain = new_chain
                    improved = True
                    no_improve = 0
                elif new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    improved = True
                    no_improve = 0
        
        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
        else:
            no_improve = 0
    
    return best_chain, best_energy

def _diverse_init(sequence, n, rng):
    """Generate diverse initial conformations using different strategies."""
    chains = []
    
    # Strategy 1: Greedy with different temperatures
    for temp in [0.1, 0.3, 0.7, 1.5]:
        chain = _greedy_grow(sequence, 2, rng, temperature=temp)
        if chain is not None and len(chain) == n:
            chains.append(chain)
    
    # Strategy 2: Random walk initialization
    for _ in range(4):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        valid = True
        for i in range(1, n):
            last = chain[-1]
            options = [s for s in _neighbours(last) if s not in occupied]
            if not options:
                valid = False
                break
            site = rng.choice(options)
            chain.append(site)
            occupied.add(site)
        if valid and len(chain) == n:
            chains.append(chain)
    
    # Strategy 3: Serpentine pattern
    serpentine = []
    for i in range(n):
        row = i // 2
        col = (i % 2) * (row + 1) if i % 2 == 0 else (row + 1) - 1
        serpentine.append((col, row))
    if len(serpentine) == n and len(set(serpentine)) == n:
        chains.append(serpentine)
    
    return chains

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses diverse initialization with local search refinement using crankshaft and pivot moves."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Phase 1: Generate diverse initial conformations
    inits = _diverse_init(sequence, n, rng)
    
    for init_chain in inits:
        energy = _energy(sequence, init_chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(init_chain)
    
    # Phase 2: Local search on each initial conformation
    for init_chain in inits:
        if time.time() > deadline - 40.0:
            break
        
        remaining = deadline - time.time()
        time_per_init = max(5.0, remaining / max(1, len(inits)))
        
        improved_chain, improved_energy = _local_search(
            sequence, init_chain, 
            max_iters=800, 
            rng=rng,
            time_limit=time.time() + time_per_init
        )
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = list(improved_chain)
    
    # Phase 3: More greedy grows with local search
    grows_done = 0
    while time.time() < deadline - 30.0:
        temperature = 0.2 + rng.random() * 2.0
        chain = _greedy_grow(sequence, dimension, rng, temperature=temperature)
        if chain is not None and len(chain) == n:
            grows_done += 1
            energy = _energy(sequence, chain)
            
            # Quick local search on promising chains
            if energy < best_energy or energy < -5:
                time_left = deadline - time.time() - 20.0
                if time_left > 10.0:
                    improved_chain, improved_energy = _local_search(
                        sequence, chain,
                        max_iters=150,
                        rng=rng,
                        time_limit=time.time() + min(15.0, time_left * 0.4)
                    )
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = list(improved_chain)
                elif energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
        
        # Limit total grows to avoid spending too much time
        if grows_done > 3000:
            break
    
    # Phase 4: Final polish on best chain with more iterations
    if best_chain is not None and time.time() < deadline - 10.0:
        final_chain, final_energy = _local_search(
            sequence, best_chain,
            max_iters=500,
            rng=rng,
            time_limit=deadline - 5.0
        )
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
