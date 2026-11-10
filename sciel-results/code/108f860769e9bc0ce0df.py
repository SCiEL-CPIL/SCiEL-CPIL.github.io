"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Generate 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _spatial_hash(point):
    """Hash a 3D point to a spatial grid cell for O(1) lookups."""
    # Use a large prime to reduce collisions
    x, y, z = point
    return (x * 1000003 + y * 10007 + z) % 1000000007

def _energy(sequence, chain):
    """Compute energy using spatial hash for O(1) neighbor lookups."""
    n = len(chain)
    if n < 2:
        return 0
    
    # Build spatial hash: maps hash -> list of (index, point)
    spatial = {}
    for i, p in enumerate(chain):
        h = _spatial_hash(p)
        if h not in spatial:
            spatial[h] = []
        spatial[h].append((i, p))
    
    contacts = 0
    for i in range(n):
        if sequence[i] == "H":
            p = chain[i]
            for q in _neighbours(p):
                h = _spatial_hash(q)
                if h in spatial:
                    for j, q_point in spatial[h]:
                        if q_point == q and j > i + 1 and sequence[j] == "H":
                            contacts += 1
    return -contacts

def _grow_2d_projection(sequence, dimension, rng):
    """
    Generate initial conformation using 2D projection heuristic.
    First fold in 2D (x,y plane) using known good patterns, then add z-dimension.
    """
    n = len(sequence)
    if n <= 2:
        return [(0,0,0)] + [(1,0,0)] * (n-1)
    
    # Start with a 2D spiral-like pattern for compactness
    chain = [(0, 0, 0)]
    occupied = {(0,0,0): 0}
    
    # Direction changes for 2D growth
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    dir_idx = 0
    steps_in_dir = 0
    max_steps = 2  # Change direction every 2 steps for spiral
    
    for i in range(1, n):
        # Try to continue in current direction, or change if blocked
        placed = False
        for attempt in range(4):
            dx, dy = directions[dir_idx]
            new_x = chain[-1][0] + dx
            new_y = chain[-1][1] + dy
            new_z = chain[-1][2]
            site = (new_x, new_y, new_z)
            
            if site not in occupied:
                # Check if this creates H-H contacts
                if sequence[i] == "H":
                    gain = 0
                    for qx, qy, qz in _neighbours(site):
                        if (qx, qy, qz) in occupied:
                            j = occupied[(qx, qy, qz)]
                            if j < i - 1 and sequence[j] == "H":
                                gain += 1
                else:
                    gain = 0
                
                # Add randomness for diversity
                score = gain * 2 + rng.random()
                
                chain.append(site)
                occupied[site] = i
                placed = True
                break
            
            # Change direction
            dir_idx = (dir_idx + 1) % 4
            steps_in_dir = 0
        
        if not placed:
            # Try adding in z-direction
            for dz in [-1, 1]:
                site = (chain[-1][0], chain[-1][1], chain[-1][2] + dz)
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    placed = True
                    break
        
        if not placed:
            return None
        
        steps_in_dir += 1
        if steps_in_dir >= max_steps:
            dir_idx = (dir_idx + 1) % 4
            steps_in_dir = 0
    
    return chain

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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
            # Slightly more random to ensure diversity
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Local search with single-residue moves, 2-residue cranks, and 3-residue cranks.
    Uses spatial hash for efficient neighbor lookups.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        improved = False
        
        # Choose move type: 0=single, 1=2-crank, 2=3-crank
        # Probabilities: 0.35, 0.35, 0.3
        r = rng.random()
        if r < 0.35:
            move_type = 0
        elif r < 0.70:
            move_type = 1
        else:
            move_type = 2

        if move_type == 0:
            # Single residue move
            if n < 3:
                continue
            i = rng.randrange(1, n - 1)
            old_site = chain[i]
            
            for new_site in _neighbours(old_site):
                if new_site in occupied:
                    continue
                
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                        continue
                if i < n - 1:
                    nxt = chain[i+1]
                    if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                        continue
                
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours(old_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta += 1
                    for q in _neighbours(new_site):
                        j = occupied.get(q)
                        if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                            delta -= 1
                
                new_energy = current_energy + delta
                if new_energy < current_energy:
                    occupied.pop(old_site)
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    improved = True
                    break
        
        elif move_type == 1:
            # 2-residue crank
            if n < 4:
                continue
            i = rng.randrange(1, n - 2)
            j = i + 1
            old_i = chain[i]
            old_j = chain[j]
            
            for new_i in _neighbours(old_i):
                if new_i in occupied:
                    occ = occupied[new_i]
                    if occ != i and occ != j:
                        continue
                
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                        continue
                
                for new_j in _neighbours(new_i):
                    if new_j in occupied:
                        occ = occupied[new_j]
                        if occ != i and occ != j:
                            continue
                    
                    if j < n - 1:
                        nxt = chain[j+1]
                        if abs(new_j[0]-nxt[0]) + abs(new_j[1]-nxt[1]) + abs(new_j[2]-nxt[2]) != 1:
                            continue
                    
                    if new_i == old_j or new_j == old_i:
                        continue
                    
                    delta = 0
                    if sequence[i] == "H":
                        for q in _neighbours(old_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(i-k) > 1:
                                delta += 1
                        for q in _neighbours(new_i):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(i-k) > 1:
                                delta -= 1
                    if sequence[j] == "H":
                        for q in _neighbours(old_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(j-k) > 1:
                                delta += 1
                        for q in _neighbours(new_j):
                            k = occupied.get(q)
                            if k is not None and k != i and k != j and sequence[k] == "H" and abs(j-k) > 1:
                                delta -= 1
                    
                    new_energy = current_energy + delta
                    if new_energy < current_energy:
                        occupied.pop(old_i)
                        occupied.pop(old_j)
                        occupied[new_i] = i
                        occupied[new_j] = j
                        chain[i] = new_i
                        chain[j] = new_j
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                        improved = True
                        break
                if improved:
                    break
        
        else:
            # 3-residue crank: move i, i+1, i+2
            if n < 5:
                continue
            i = rng.randrange(1, n - 3)
            j = i + 1
            k = i + 2
            old_i = chain[i]
            old_j = chain[j]
            old_k = chain[k]
            
            for new_i in _neighbours(old_i):
                if new_i in occupied:
                    occ = occupied[new_i]
                    if occ not in (i, j, k):
                        continue
                
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                        continue
                
                for new_j in _neighbours(new_i):
                    if new_j in occupied:
                        occ = occupied[new_j]
                        if occ not in (i, j, k):
                            continue
                    
                    if new_j == old_k:
                        continue
                    
                    for new_k in _neighbours(new_j):
                        if new_k in occupied:
                            occ = occupied[new_k]
                            if occ not in (i, j, k):
                                continue
                        
                        if k < n - 1:
                            nxt = chain[k+1]
                            if abs(new_k[0]-nxt[0]) + abs(new_k[1]-nxt[1]) + abs(new_k[2]-nxt[2]) != 1:
                                continue
                        
                        if new_i == old_j or new_j == old_i or new_j == old_k or new_k == old_j:
                            continue
                        
                        delta = 0
                        for idx, old_pos, new_pos in [(i, old_i, new_i), (j, old_j, new_j), (k, old_k, new_k)]:
                            if sequence[idx] == "H":
                                for q in _neighbours(old_pos):
                                    m = occupied.get(q)
                                    if m is not None and m not in (i, j, k) and sequence[m] == "H" and abs(idx-m) > 1:
                                        delta += 1
                                for q in _neighbours(new_pos):
                                    m = occupied.get(q)
                                    if m is not None and m not in (i, j, k) and sequence[m] == "H" and abs(idx-m) > 1:
                                        delta -= 1
                        
                        new_energy = current_energy + delta
                        if new_energy < current_energy:
                            occupied.pop(old_i)
                            occupied.pop(old_j)
                            occupied.pop(old_k)
                            occupied[new_i] = i
                            occupied[new_j] = j
                            occupied[new_k] = k
                            chain[i] = new_i
                            chain[j] = new_j
                            chain[k] = new_k
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_chain = list(chain)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved and iters % 100 == 0:
            if time.time() > time_limit:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using 2D projection heuristic + diverse growth + local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Adaptive number of growths based on remaining time
    if n <= 48:
        num_growths = 200
    elif n <= 80:
        num_growths = 120
    else:
        num_growths = 80
    
    candidate_chains = []
    
    # First, try 2D projection heuristic for better initial structures
    for attempt in range(min(50, num_growths // 4)):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow_2d_projection(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
        candidate_chains.append((value, chain))
    
    # Then, diverse randomized growth
    for attempt in range(num_growths - len(candidate_chains)):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
        candidate_chains.append((value, chain))
    
    candidate_chains.sort(key=lambda x: x[0])
    
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        # Allocate more time to local search
        num_ls_runs = min(len(candidate_chains), max(3, int(remaining / 12)))
        per_run_time = (remaining - 5.0) / num_ls_runs
        
        for idx in range(num_ls_runs):
            if _remaining_time() < 5.0:
                break
            
            # Use diverse starting chains
            if idx == 0:
                start_chain = candidate_chains[0][1]
            elif idx < len(candidate_chains) // 2:
                start_chain = candidate_chains[idx][1]
            else:
                # For later runs, pick from a random subset for diversity
                pool = candidate_chains[len(candidate_chains) // 2:]
                if pool:
                    start_chain = pool[rng.randrange(len(pool))][1]
                else:
                    start_chain = candidate_chains[0][1]
            
            ls_time_limit = time.time() + per_run_time
            max_ls_iters = max(1000, min(30000, 150 * len(sequence)))
            improved_chain = _local_search(sequence, list(start_chain), max_ls_iters, rng, ls_time_limit)
            improved_energy = _energy(sequence, improved_chain)
            if improved_energy < best_energy:
                best, best_energy = improved_chain, improved_energy
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
