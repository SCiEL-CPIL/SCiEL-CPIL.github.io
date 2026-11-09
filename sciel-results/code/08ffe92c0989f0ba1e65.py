"""HP lattice protein folding with parallel-restart strategy and contact-map local search."""
# EVOLVE-BLOCK-START
import random
import time
import math


# Pre-compute neighbor offsets for 3D cubic lattice
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _neighbours(point):
    """Generate 6 neighbours of a 3D lattice point."""
    x, y, z = point
    for dx, dy, dz in NEIGHBOR_OFFSETS:
        yield (x + dx, y + dy, z + dz)

def _get_neighbours_fast(x, y, z):
    """Fast neighbor generation using precomputed offsets."""
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]


def _grow(sequence, dimension, rng, bias=0.7):
    """Randomized greedy chain growth with tunable contact bias."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _get_neighbours_fast(*chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _get_neighbours_fast(*site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _get_neighbours_fast(*site) if q not in occupied)
            # Mix contact gain and free space with tunable bias
            score = gain * bias * 10.0 + free * (1 - bias) * 0.5 + rng.random() * 0.1
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _get_neighbours_fast(*p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_fast(sequence, chain, pos_map, i, new_pos):
    """Fast delta energy computation using precomputed position map."""
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == "H"
    if not is_h:
        return 0
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    for q in _get_neighbours_fast(*old_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    for q in _get_neighbours_fast(*new_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
            new_contacts += 1
    
    return -(new_contacts - old_contacts)


def _local_search_contact_map(sequence, chain, rng, max_iter=5000):
    """
    Contact-map-based local search: iteratively try to improve H-H contacts
    by moving H residues to positions that create new contacts.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Identify H positions
    h_indices = [i for i in range(n) if sequence[i] == "H"]
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each H residue to a better position
        for i in h_indices:
            if i == 0 or i == n - 1:
                continue  # Skip endpoints
            
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            # Try all valid neighbor positions
            for new_pos in _get_neighbours_fast(*old_pos):
                # Check unit steps
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                
                # Check self-avoidance
                if new_pos in pos_map:
                    continue
                
                # Compute delta
                delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)
                
                if delta < 0:  # Energy improves
                    # Accept immediately
                    pos_map.pop(old_pos)
                    pos_map[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Move to next H residue
        
        if not improved:
            break
    
    return best_chain


def _anneal_short(sequence, chain, dimension, rng, time_budget):
    """Short simulated annealing with time budget."""
    n = len(chain)
    if n < 4:
        return chain
    
    start_time = time.time()
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Temperature schedule
    T0 = max(1.0, abs(current_energy) * 0.3)
    T_min = 0.05
    
    # Estimate steps
    est_steps_per_sec = 8000
    total_steps = int(time_budget * est_steps_per_sec)
    total_steps = max(200, min(total_steps, 30000))
    
    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0
    
    pos_set = set(current_chain)
    
    for step in range(total_steps):
        if time.time() - start_time >= time_budget - 0.1:
            break
        
        T *= cooling_factor
        if T < T_min:
            T = T_min
        
        move_type = rng.random()
        
        if move_type < 0.55:
            # Flip move
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            valid_options = []
            for new_pos in _get_neighbours_fast(*old_pos):
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos in pos_set:
                    continue
                valid_options.append(new_pos)
            
            if not valid_options:
                continue
            
            new_pos = rng.choice(valid_options)
            pos_map = {p: idx for idx, p in enumerate(current_chain) if idx != i}
            delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                pos_set.discard(old_pos)
                pos_set.add(new_pos)
                current_chain[i] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        elif move_type < 0.80:
            # Pivot move
            i = rng.randint(1, n - 3)
            axis = rng.randint(0, 2)
            sign = rng.choice([1, -1])
            
            pivot = current_chain[i]
            tail = current_chain[i + 1:]
            
            new_tail = []
            for pos in tail:
                rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
                if axis == 0:
                    if sign == 1:
                        new_rel = (rel[0], -rel[2], rel[1])
                    else:
                        new_rel = (rel[0], rel[2], -rel[1])
                elif axis == 1:
                    if sign == 1:
                        new_rel = (rel[2], rel[1], -rel[0])
                    else:
                        new_rel = (-rel[2], rel[1], rel[0])
                else:
                    if sign == 1:
                        new_rel = (-rel[1], rel[0], rel[2])
                    else:
                        new_rel = (rel[1], -rel[0], rel[2])
                new_tail.append((pivot[0] + new_rel[0], pivot[1] + new_rel[1], pivot[2] + new_rel[2]))
            
            new_chain = current_chain[:i + 1] + new_tail
            
            if len(set(new_chain)) != n:
                continue
            
            diff = abs(new_chain[i + 1][0] - new_chain[i][0]) + abs(new_chain[i + 1][1] - new_chain[i][1]) + abs(new_chain[i + 1][2] - new_chain[i][2])
            if diff != 1:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        else:
            # Reversal move
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            
            new_chain = list(current_chain)
            new_chain[a:b + 1] = reversed(new_chain[a:b + 1])
            
            if len(set(new_chain)) != n:
                continue
            
            valid = True
            for k in range(n - 1):
                d = abs(new_chain[k + 1][0] - new_chain[k][0]) + abs(new_chain[k + 1][1] - new_chain[k][1]) + abs(new_chain[k + 1][2] - new_chain[k][2])
                if d != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Parallel-restart strategy: multiple independent growth+anneal cycles,
    each with a time budget, plus contact-map local search refinement.
    """
    deadline = time.time() + 235.0  # 235s budget with safety margin
    n = len(sequence)
    
    # Number of restarts scales with sequence length
    if n <= 50:
        num_restarts = 8
    elif n <= 80:
        num_restarts = 5
    elif n <= 120:
        num_restarts = 3
    else:
        num_restarts = 2
    
    best = None
    best_energy = 0  # 0 = no contacts (unfolded)
    
    # Time per restart
    time_per_restart = (deadline - time.time() - 5.0) / num_restarts  # leave 5s for final refinement
    time_per_restart = max(2.0, min(time_per_restart, 40.0))
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        rng = random.Random(1000 + restart * 7919 + len(sequence))
        
        # Growth with varying bias
        bias = 0.5 + 0.4 * (restart % 3) / 2.0  # Vary between 0.5 and 0.9
        chain = _grow(sequence, dimension, rng, bias=bias)
        
        if chain is None:
            # Try with different bias
            chain = _grow(sequence, dimension, rng, bias=0.7)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Short anneal
        remaining = deadline - time.time() - 3.0
        if remaining > 1.0:
            anneal_budget = min(time_per_restart - 0.5, remaining * 0.6)
            annealed = _anneal_short(sequence, chain, dimension, rng, anneal_budget)
            anneal_energy = _energy(sequence, annealed)
            
            if anneal_energy < energy:
                chain = annealed
                energy = anneal_energy
        
        # Contact-map local search
        if time.time() < deadline - 2.0:
            refined = _local_search_contact_map(sequence, chain, rng, max_iter=2000)
            refine_energy = _energy(sequence, refined)
            
            if refine_energy < energy:
                chain = refined
                energy = refine_energy
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Final refinement with remaining time
    remaining = deadline - time.time() - 1.0
    if remaining > 2.0:
        rng = random.Random(99999 + len(sequence))
        final = _anneal_short(sequence, best, dimension, rng, remaining * 0.7)
        final_energy = _energy(sequence, final)
        
        if final_energy < best_energy:
            best = final
            best_energy = final_energy
        
        # One more contact-map refinement
        if time.time() < deadline - 0.5:
            refined = _local_search_contact_map(sequence, best, rng, max_iter=3000)
            refine_energy = _energy(sequence, refined)
            if refine_energy < best_energy:
                best = refined
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
