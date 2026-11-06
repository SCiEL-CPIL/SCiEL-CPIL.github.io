"""HP lattice protein folding on the cubic lattice - simulated annealing approach."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_walk(sequence, dimension, rng):
    """
    Build an initial self-avoiding walk using a heuristic that tries to
    cluster H residues. Uses a modified greedy approach with look-ahead.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        current = chain[-1]
        candidates = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            # Score this site
            score = rng.random() * 0.1
            if sequence[i] == 'H':
                # Count H neighbours already placed
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        score += 1.0
            # Slight preference for sites with more free space (avoid trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            candidates.append((score, site))
        
        if not candidates:
            # Dead end - try to backtrack or just fail
            return None
        
        # Pick best candidate
        candidates.sort(key=lambda x: -x[0])
        site = candidates[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain


def _compute_energy(sequence, chain, occupied=None):
    """Compute the energy (negative number of H-H contacts)."""
    if occupied is None:
        occupied = {p: i for i, p in enumerate(chain)}
    n = len(chain)
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _simulated_annealing(sequence, dimension, time_budget, rng):
    """
    Simulated annealing to optimize the conformation.
    Moves: relocate a single residue to an adjacent free site.
    """
    n = len(sequence)
    
    # Get initial conformation
    chain = _initial_walk(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain, occupied)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 100.0
    t_final = 0.01
    # Number of iterations we can afford
    # Estimate: each iteration takes ~microseconds, so we can do millions
    # Let's aim for a reasonable number based on time budget
    max_iters = min(500000, int(time_budget * 50000))  # ~50k iters per second
    if max_iters < 1000:
        max_iters = 1000
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.95:
            break
        
        # Temperature schedule: exponential cooling
        if it < max_iters:
            alpha = it / max_iters
            temp = t0 * (t_final / t0) ** alpha
        else:
            temp = t_final
        
        # Pick a random residue to move (not endpoints for simplicity, but try all)
        i = rng.randint(0, n - 1)
        old_pos = chain[i]
        
        # Try to move to a neighbouring free site
        neighbours = list(_neighbours(old_pos))
        rng.shuffle(neighbours)
        
        moved = False
        for new_pos in neighbours:
            if new_pos in occupied:
                continue
            
            # Check if moving i to new_pos keeps the chain connected
            # The chain must remain a valid self-avoiding walk with unit steps
            # So new_pos must be adjacent to chain[i-1] (if i>0) and chain[i+1] (if i<n-1)
            
            valid = True
            if i > 0:
                prev = chain[i - 1]
                if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2]) != 1:
                    valid = False
            if valid and i < n - 1:
                nxt = chain[i + 1]
                if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Compute energy change
            # Remove old_pos from occupied temporarily
            occupied.pop(old_pos)
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _compute_energy(sequence, chain, occupied)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
                # Accept
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                moved = True
                break
            else:
                # Reject: undo
                chain[i] = old_pos
                occupied.pop(new_pos)
                occupied[old_pos] = i
        
        if not moved:
            continue
    
    return best_chain, best_energy


def _try_multiple_starts(sequence, dimension, time_budget, rng):
    """Try multiple initializations and pick the best."""
    best_chain = None
    best_energy = 0
    
    # Number of restarts based on time budget
    num_restarts = max(1, min(5, int(time_budget / 2)))
    time_per_restart = time_budget / num_restarts
    
    for r in range(num_restarts):
        remaining = time_budget - (time.time() - _try_multiple_starts._start_time) if hasattr(_try_multiple_starts, '_start_time') else time_budget
        if remaining < 0.1:
            break
        chain, energy = _simulated_annealing(sequence, dimension, time_per_restart, rng)
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using simulated annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time: longer sequences get more time
    # Total budget is 240s for all 26 sequences
    # We don't know exactly how many sequences will be called, so be conservative
    # Assume ~26 sequences, so ~9 seconds each on average
    # But give longer sequences proportionally more time
    base_time = 5.0
    time_budget = base_time + (n / 100.0) * 3.0  # up to ~19s for 136-residue
    time_budget = min(time_budget, 20.0)
    
    _try_multiple_starts._start_time = time.time()
    chain, energy = _try_multiple_starts(sequence, dimension, time_budget, rng)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
