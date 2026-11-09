"""HP lattice protein folding on the cubic lattice.
Combines deterministic greedy growth with local tail re-growth refinement
and a global deadline guard to stay within the wall-clock budget.
"""
import random
import time

# Global deadline: evaluator kills at 330s wall-clock. Use 320s safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute the total energy of the chain. Energy = -num_HH_contacts."""
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
    One randomized greedy chain growth.
    Heuristic: prefer sites with new H-H contacts, then free space, with small noise.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Free space penalty/reward to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: contacts are most important, free space is secondary
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _regrow_tail(sequence, chain, i, rng):
    """
    Re-grow the tail from index i onwards, given the prefix up to i-1.
    Returns (new_chain, new_energy) or (None, None) if stuck.
    """
    prefix = chain[:i]
    occupied = {p: idx for idx, p in enumerate(prefix)}
    new_chain = list(prefix)
    
    for k in range(i, len(sequence)):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < k - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None, None
        
        best_score, best_site = max(options)
        new_chain.append(best_site)
        occupied[best_site] = k
    
    energy = _energy(sequence, new_chain)
    return new_chain, energy


def fold(sequence, dimension):
    """
    Fold the sequence on the cubic lattice.
    Phase 1: Multiple randomized greedy growths to find a good initial conformation.
    Phase 2: Local optimization via tail re-growth to refine the conformation.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Check remaining time
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        # Not enough time, return a simple valid chain (straight line)
        return [[i, 0, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts)
    
    # Phase 1: Multiple greedy growths
    # Allocate time based on sequence length and remaining budget
    # Longer sequences need more time per growth
    # Estimate time per growth: ~n * constant
    # Let's allocate a fraction of remaining time to Phase 1
    # For short sequences, we can do many fast growths.
    # For long sequences, fewer growths but each is slower.
    
    # Heuristic: allocate 40% of remaining time to Phase 1
    phase1_time_budget = time_remaining * 0.4
    # Estimate growth time: ~0.001s per residue? Let's say 1ms per residue.
    # So growth_time ~ n * 0.001
    # num_growths ~ phase1_time_budget / (n * 0.001)
    est_growth_time = n * 0.002
    num_growths = max(10, int(phase1_time_budget / est_growth_time))
    # Cap num_growths to avoid excessive time for very short sequences
    num_growths = min(num_growths, 5000)
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE - 10:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local optimization via tail re-growth
    current_chain = list(best_chain)
    current_energy = best_energy
    
    # Allocate remaining time to Phase 2
    time_remaining = _DEADLINE - time.time()
    if time_remaining < 1.0:
        return [list(p) for p in best_chain]
    
    # Each local step involves re-growing a tail of average length n/2
    # So time per step ~ (n/2) * 0.002
    est_step_time = (n / 2.0) * 0.002
    num_local_steps = max(10, int(time_remaining / est_step_time))
    # Cap to avoid too many steps if time estimate is off
    num_local_steps = min(num_local_steps, 50000)
    
    for step in range(num_local_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a random position to re-grow the tail from
        # Bias towards earlier positions for more impact, but also try later ones
        # Sometimes pick a position close to the end for fine-tuning
        if rng.random() < 0.5:
            # Re-grow a significant part of the chain
            i = rng.randint(1, max(1, n // 2))
        else:
            # Re-grow a smaller part
            i = rng.randint(n // 2, n - 1)
        
        i = min(i, n - 1)
        
        new_chain, new_energy = _regrow_tail(sequence, current_chain, i, rng)
        if new_chain is None:
            continue
        
        # Accept if energy improves
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        elif rng.random() < 0.05:
            # Occasionally accept worse solutions to escape local optima
            # This is a simple form of simulated annealing
            current_chain = new_chain
            current_energy = new_energy
    
    # Ensure we return the best chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
