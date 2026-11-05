"""HP lattice protein folding with multi-start local search and deadline guard."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.85  # Use 85% of the 330s budget to be safe


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE


def _is_deadline_exceeded():
    return time.time() > _get_deadline()


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (for dimension=3)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts) of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_grow(sequence, dimension, rng):
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_iters=5000):
    """Improve a chain using local search with single-step moves."""
    if chain is None:
        return None
    occupied = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = energy
    
    # Precompute contact map for faster energy updates
    contact_map = defaultdict(int)  # (i, j) -> 1 if H-H contact
    
    def _compute_contacts(seq, ch):
        where = {p: i for i, p in enumerate(ch)}
        contacts = set()
        for i, p in enumerate(ch):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    contacts = _compute_contacts(sequence, chain)
    energy = -len(contacts)
    best_energy = energy
    best_chain = list(chain)
    
    for iteration in range(max_iters):
        if _is_deadline_exceeded():
            break
        if iteration % 100 == 0 and _is_deadline_exceeded():
            break
            
        # Pick a random position to move
        if len(chain) <= 2:
            break
        pos = random.randint(1, len(chain) - 2)  # Don't move ends
        
        old_site = chain[pos]
        # Find a random neighbour to move to
        moves = list(_neighbours(old_site))
        random.shuffle(moves)
        moved = False
        for new_site in moves:
            if new_site in occupied and occupied[new_site] != pos:
                continue
            # Check if moving is valid (no self-intersection)
            # The site must be free or be the old site
            if new_site == old_site:
                continue
            if new_site in occupied:
                continue
            
            # Tentatively move
            occupied.pop(old_site, None)
            occupied[new_site] = pos
            chain[pos] = new_site
            
            # Recompute energy quickly
            new_contacts = _compute_contacts(sequence, chain)
            new_energy = -len(new_contacts)
            
            if new_energy < energy:
                energy = new_energy
                contacts = new_contacts
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                moved = True
                break
            else:
                # Revert
                occupied.pop(new_site, None)
                occupied[old_site] = pos
                chain[pos] = old_site
        
        if not moved and iteration % 1000 == 0:
            # Stuck, try a different strategy: random restart of local search
            break
    
    return best_chain


def _local_search_simple(sequence, chain, max_iters=2000):
    """Simpler local search: try moving each residue to a random free neighbour."""
    if chain is None:
        return None
    occupied = {p: i for i, p in enumerate(chain)}
    energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = energy
    
    for iteration in range(max_iters):
        if _is_deadline_exceeded():
            break
        if iteration % 50 == 0 and _is_deadline_exceeded():
            break
            
        pos = random.randint(1, len(chain) - 2)
        old_site = chain[pos]
        moves = list(_neighbours(old_site))
        random.shuffle(moves)
        
        for new_site in moves:
            if new_site in occupied and occupied[new_site] != pos:
                continue
            if new_site == old_site:
                continue
            
            # Tentatively move
            occupied.pop(old_site, None)
            occupied[new_site] = pos
            chain[pos] = new_site
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < energy:
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(chain)
                break
            else:
                # Revert
                occupied.pop(new_site, None)
                occupied[old_site] = pos
                chain[pos] = old_site
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multi-start local search with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded)
    
    # First, try several initializations
    n_initializations = min(50, max(10, 300 // max(1, len(sequence) // 10)))
    
    for init_idx in range(n_initializations):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search
        improved = _local_search_simple(sequence, chain, max_iters=1000)
        if improved is not None:
            energy = _energy(sequence, improved)
            if energy < best_energy:
                best_energy = energy
                best_chain = improved
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
