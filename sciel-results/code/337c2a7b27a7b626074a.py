"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Improved heuristic: weight contacts more heavily, moderate free-space bonus
            score = gain * 2.0 + free * 0.5 + rng.random()
            options.append((score, site))
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
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _incremental_energy_change(sequence, chain, occupied, i, old_pos, new_pos):
    """Compute energy delta when moving residue i from old_pos to new_pos."""
    delta = 0
    if sequence[i] == 'H':
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1
    if sequence[i] == 'H':
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1
    return delta


def _is_valid_move(occupied, i, new_pos):
    """Check if moving residue i to new_pos preserves self-avoidance."""
    if new_pos in occupied:
        return False
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return False
    return True


def _local_search(sequence, chain, max_iter=5000, rng=None):
    """Improve conformation via single-residue moves with early stopping on stagnation."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    no_improve = 0
    max_no_improve = 50
    
    for _ in range(max_iter):
        if no_improve > max_no_improve:
            break
        
        improved = False
        # Try moving each internal residue
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if not _is_valid_move(occupied, i, new_pos):
                    continue
                
                delta = _incremental_energy_change(sequence, best_chain, occupied, i, current_pos, new_pos)
                new_energy = best_energy + delta
                
                if new_energy < best_energy:
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy = new_energy
                    no_improve = 0
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, max_iter=10000, initial_temp=5.0, final_temp=0.1, rng=None):
    """Simulated annealing for HP lattice protein folding."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, max_iter))
    
    for iteration in range(max_iter):
        temp = initial_temp * (cooling_rate ** iteration)
        i = rng.randint(1, n - 2)
        current_pos = current_chain[i]
        
        for new_pos in _neighbours2d(current_pos):
            if not _is_valid_move(occupied, i, new_pos):
                continue
            
            delta = _incremental_energy_change(sequence, current_chain, occupied, i, current_pos, new_pos)
            new_energy = current_energy + delta
            
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                current_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growths, top-K refinement, and simulated annealing."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Adjust number of restarts based on sequence length
    if n <= 30:
        num_restarts = 400
    elif n <= 60:
        num_restarts = 250
    else:
        num_restarts = 120
    
    # Keep top-K candidates for multi-basin refinement
    top_k = 5 if n <= 50 else 3
    candidates = []  # list of (energy, chain)
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 30.0:  # Reserve time for refinement
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Quick local search to get a decent starting point
        chain, energy = _local_search(sequence, chain, max_iter=min(2000, max(500, 5000 // n)), rng=rng)
        
        # Insert into top-K
        candidates.append((energy, chain))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > top_k:
            candidates.pop()
    
    if not candidates:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
        return [list(p) for p in best]
    
    # Refine top-K candidates with deeper local search + SA
    best_chain, best_energy = candidates[0]
    
    for energy, chain in candidates:
        if time.time() > deadline - 10.0:
            break
        
        # Deeper local search
        chain, energy = _local_search(sequence, chain, max_iter=min(8000, max(2000, 20000 // n)), rng=rng)
        
        if energy < best_energy:
            best_chain, best_energy = chain, energy
    
    # Final SA polish on the best
    if time.time() < deadline - 5.0:
        sa_time_budget = min(25.0, deadline - time.time() - 3.0)
        sa_iter = int(min(40000, max(1000, sa_time_budget * 400)))
        best_chain, best_energy = _simulated_annealing(
            sequence, best_chain, max_iter=sa_iter, initial_temp=3.0, final_temp=0.05, rng=rng
        )
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
