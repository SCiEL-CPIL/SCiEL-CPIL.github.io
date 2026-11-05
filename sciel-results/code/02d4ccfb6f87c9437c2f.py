"""HP lattice protein folding on the cubic lattice - from-scratch design."""
# EVOLVE-BLOCK-START
import random
import heapq
import math


def _neighbours(point):
    """Yield the 6 neighbours of a point on the 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of non-consecutive H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.5):
    """
    Grow the chain using a randomized greedy heuristic.
    At each step, score candidate sites by:
      - number of new H-H contacts formed
      - number of free neighbours (to avoid trapping)
      - random noise
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Count new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Count free neighbours (escape routes)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: prefer high gain, then high free, with noise
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        candidates.sort(key=lambda x: -x[0])
        site = candidates[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, dimension, max_iter=5000, rng=None):
    """
    Perform local search by trying to move individual residues to nearby free sites.
    This is a simple hill-climbing approach.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iter):
        # Pick a random residue to move
        i = rng.randint(1, n - 2)  # Don't move endpoints
        old_pos = chain[i]
        
        # Try moving to a neighbouring free site
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            
            # Check if the move is valid (no collisions)
            # Remove old position, add new position
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            
            # Compute energy change efficiently
            # We only need to check contacts involving residue i and its neighbours
            old_contacts = 0
            new_contacts = 0
            
            # Check H-H contacts for residue i
            if sequence[i] == "H":
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                        new_contacts += 1
            
            # Also check if any other H residue loses a contact with i
            # This is approximate but good enough for local search
            
            energy_change = -new_contacts  # Simplified; full recomputation below
            
            # Full recomputation for correctness
            new_energy = _energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_pos
                del where[new_pos]
                where[old_pos] = i
        
        # Occasionally restart from best found
        if rng.random() < 0.1:
            chain = list(best_chain)
            where = {p: i for i, p in enumerate(chain)}
            current_energy = best_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find the best conformation using multiple randomized greedy growths
    followed by local search refinement.
    """
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so start at 0 (worst)
    
    # Number of restarts based on chain length
    if n <= 50:
        n_restarts = 100
    elif n <= 80:
        n_restarts = 50
    else:
        n_restarts = 25
    
    for restart in range(n_restarts):
        # Vary the temperature slightly for diversity
        temp = 0.3 + 0.4 * (restart / max(1, n_restarts - 1))
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
        
        # Refine with local search
        refined_chain, refined_energy = _local_search(sequence, chain, dimension, 
                                                       max_iter=2000, rng=rng)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # If no valid chain found, return a straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
