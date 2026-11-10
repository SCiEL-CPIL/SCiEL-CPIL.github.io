"""HP lattice protein folding on cubic lattice - replica exchange SA with diverse initializations."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # 315s hard limit, 15s safety margin


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _neighbours_set(point):
    """Return set of 6 neighbours."""
    return set(_neighbours(point))


def _are_adjacent(a, b):
    """Check if two lattice points are adjacent (unit distance)."""
    return sum(abs(a[i] - b[i]) for i in range(len(a))) == 1


def _compute_energy(sequence, chain, occupied=None):
    """Compute energy = -number of H-H contacts."""
    if occupied is None:
        occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, mode='greedy'):
    """
    Generate initial chain.
    mode: 'greedy' - prefer H-H contacts
          'compact' - prefer compactness
          'random' - random walk
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            if mode == 'greedy':
                gain = 0
                if sequence[i] == 'H':
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 2.0 + 0.1 * free + rng.random() * 0.5
            elif mode == 'compact':
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                cz = sum(p[2] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
                gain = 0
                if sequence[i] == 'H':
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 3.0 - dist * 0.5 + free * 0.1 + rng.random() * 0.3
            else:  # random
                score = rng.random()
            
            options.append((score, site))
        
        if not options:
            return None
        
        if mode == 'greedy':
            site = max(options)[1]
        else:
            options.sort(reverse=True)
            k = min(3, len(options))
            site = options[rng.randint(0, k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _compute_move_delta(sequence, chain, occupied, i, new_site):
    """
    Compute energy delta for moving residue i to new_site.
    Returns (delta, valid).
    """
    n = len(chain)
    if i == 0 or i == n - 1:
        return 0, False
    if new_site in occupied:
        return 0, False
    if not _are_adjacent(chain[i - 1], new_site):
        return 0, False
    if not _are_adjacent(chain[i + 1], new_site):
        return 0, False
    
    old_site = chain[i]
    if sequence[i] != 'H':
        return 0, True
    
    # Compute lost contacts at old_site
    lost = 0
    for q in _neighbours(old_site):
        j = occupied.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == 'H':
                lost += 1
    
    # Compute gained contacts at new_site
    gained = 0
    for q in _neighbours(new_site):
        j = occupied.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1:
            if sequence[j] == 'H':
                gained += 1
    
    delta = gained - lost
    return delta, True


def _local_search(sequence, chain, rng, deadline, max_stagnation=300):
    """
    Simulated annealing local search with single-site moves.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    stagnation = 0
    temp = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline and stagnation < max_stagnation:
        # Choose a random residue to move
        i = rng.randint(1, n - 2)
        candidates = list(_neighbours(current_chain[i]))
        if not candidates:
            continue
        new_site = candidates[rng.randint(0, len(candidates) - 1)]
        
        delta, valid = _compute_move_delta(sequence, current_chain, occupied, i, new_site)
        if not valid:
            continue
        
        accept = False
        if delta <= 0:
            accept = True
        else:
            if temp > 0.001:
                accept = rng.random() < math.exp(-delta / temp)
        
        if accept:
            old_site = current_chain[i]
            occupied[old_site] = -1
            occupied[new_site] = i
            current_chain[i] = new_site
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
        
        temp *= cooling_rate
        if temp < 0.005:
            temp = 0.5  # Reheat
    
    return best_chain, best_energy


def _perturb(sequence, chain, rng, num_moves):
    """
    Apply random perturbations to chain.
    Returns new_chain or None if invalid.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    new_chain = list(chain)
    occupied = {p: i for i, p in enumerate(new_chain)}
    
    for _ in range(num_moves):
        i = rng.randint(1, n - 2)
        candidates = list(_neighbours(new_chain[i]))
        if not candidates:
            continue
        new_site = candidates[rng.randint(0, len(candidates) - 1)]
        
        if new_site in occupied:
            continue
        if not _are_adjacent(new_chain[i - 1], new_site):
            continue
        if not _are_adjacent(new_chain[i + 1], new_site):
            continue
        
        old_site = new_chain[i]
        occupied[old_site] = -1
        occupied[new_site] = i
        new_chain[i] = new_site
    
    return new_chain


def fold(sequence, dimension):
    """
    Best conformation via diverse initializations + iterative local search with perturbation.
    Uses replica exchange strategy: multiple diverse initializations, then SA refinement.
    """
    start_time = time.time()
    # 240s total for 26 sequences, so ~9.2s each. Use 8.5s per sequence for safety.
    total_deadline = start_time + 8.5
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Generate multiple diverse initial chains
    initial_chains = []
    modes = ['greedy', 'compact', 'random', 'greedy', 'compact']
    
    for mode in modes:
        if time.time() > total_deadline - 2.0:
            break
        chain = _grow(sequence, dimension, rng, mode=mode)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        initial_chains.append(chain)
    
    # Phase 2: Run local search on each initial chain, keep the best
    best_chain = initial_chains[0]
    best_energy = _compute_energy(sequence, best_chain)
    
    for chain in initial_chains:
        if time.time() > total_deadline - 1.0:
            break
        ls_deadline = min(total_deadline - 0.5, time.time() + 1.2)
        if ls_deadline <= time.time():
            break
        
        new_chain, new_energy = _local_search(sequence, chain, rng, ls_deadline)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = list(new_chain)
    
    # Phase 3: Iterative improvement: perturb and local search
    iteration = 0
    while time.time() < total_deadline - 0.5:
        # Perturb the best chain
        num_perturb = min(10, len(sequence) // 4)
        perturbed = _perturb(sequence, best_chain, rng, num_perturb)
        if perturbed is None:
            break
        
        # Local search on perturbed chain
        iter_deadline = min(total_deadline - 0.3, time.time() + 1.0)
        if iter_deadline <= time.time():
            break
        
        new_chain, new_energy = _local_search(sequence, perturbed, rng, iter_deadline)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = list(new_chain)
        
        iteration += 1
        if iteration > 15:  # Safety limit
            break
    
    return [list(p) for p in best_chain]
