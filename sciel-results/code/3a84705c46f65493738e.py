"""Rewritten program for HP lattice protein folding using a model-and-optimize approach."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Deadline guard: 210s wall-clock, keep a safety margin
_DEADLINE = 205.0
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    """Yield 4-neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    One randomized greedy chain growth with simulated annealing-like acceptance.
    Prefer sites with new H-H contacts, then free space, with some randomness.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: H-H contact gain + free space + randomness
            gain = 0.0
            if sequence[i] == 'H':
                x, y = site
                for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1.0
            # Count free neighbours for flexibility
            free = 0
            x, y = site
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.15 * free + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best (with some randomness via temperature already in score)
        options.sort(key=lambda x: -x[0])
        # With temperature, sometimes pick second best
        if temperature > 0 and len(options) > 1 and rng.random() < temperature * 0.3:
            site = options[1][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_steps=200):
    """
    Local search: try to improve energy by moving a single residue to an adjacent free site.
    Returns improved chain or original if no improvement found.
    """
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    steps = 0
    while steps < max_steps and _remaining_time() > 5.0:
        steps += 1
        # Pick a random interior residue to try moving
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        for new_pos in _neighbours(old_pos):
            if new_pos in set(chain):
                continue
            # Check that moving doesn't create overlap
            if new_pos == chain[i-1] or new_pos == chain[i+1]:
                continue
            
            # Tentatively move
            chain[i] = new_pos
            
            # Check self-avoidance
            valid = True
            for j in range(n):
                if j != i and chain[j] == new_pos:
                    valid = False
                    break
            if not valid:
                chain[i] = old_pos
                continue
            
            new_energy = _energy(sequence, chain)
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                chain[i] = old_pos
    
    return best_chain, best_energy


def _fold_with_restarts(sequence, dimension, n_restarts=50, max_ls_steps=100):
    """
    Fold sequence using multiple randomized greedy growths followed by local search.
    Returns best chain found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0  # 0 means no valid chain yet
    
    for r in range(n_restarts):
        if _remaining_time() < 2.0:
            break
        
        # Vary temperature per restart
        temp = 0.5 + 0.5 * (r / max(1, n_restarts - 1))
        rng = random.Random(12345 + len(sequence) * 100 + r)
        
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Local search improvement
        if _remaining_time() > 3.0:
            chain, energy = _local_search(sequence, chain, rng, max_steps=max_ls_steps)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation found within time budget.
    Returns list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    
    # Determine number of restarts based on sequence length and time budget
    # Shorter sequences can afford more restarts
    if n <= 30:
        n_restarts = 100
        max_ls = 150
    elif n <= 60:
        n_restarts = 60
        max_ls = 100
    else:
        n_restarts = 40
        max_ls = 80
    
    best_chain = _fold_with_restarts(sequence, dimension, n_restarts=n_restarts, max_ls_steps=max_ls)
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
