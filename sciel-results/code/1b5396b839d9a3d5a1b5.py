"""HP lattice protein folding on the cubic lattice.
Uses beam search with a composite score (contacts + compactness) to guide growth.
"""
import time
import random
import sys

_DEADLINE_MARGIN = 10.0  # seconds of safety margin
_START_TIME = None


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _beam_score(sequence, chain, occupied, current_index):
    """Compute a score for a partial chain: H-H contacts + compactness bonus."""
    contacts = 0
    for i in range(current_index + 1):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    # Compactness: negative sum of squared distances from centroid
    if current_index > 0:
        cx = sum(p[0] for p in chain) / (current_index + 1)
        cy = sum(p[1] for p in chain) / (current_index + 1)
        cz = sum(p[2] for p in chain) / (current_index + 1)
        compactness = -sum((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 for p in chain)
    else:
        compactness = 0.0
    return contacts * 10.0 + compactness


def _beam_search(sequence, dimension, beam_width, max_time):
    """Beam search for HP protein folding. Returns best chain found."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Each beam entry: (chain_list, occupied_dict, score)
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_score = _beam_score(sequence, initial_chain, initial_occupied, 0)
    
    beam = [(initial_chain, initial_occupied, initial_score)]
    best_chain = initial_chain
    best_score = initial_score
    
    for i in range(1, n):
        if time.time() > max_time:
            break
        
        new_beam = []
        seen = set()
        
        for chain, occupied, score in beam:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Check for self-avoidance (already handled by occupied set)
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                # Compute new score
                new_score = _beam_score(sequence, new_chain, new_occupied, i)
                options.append((new_score, site, new_chain, new_occupied))
            
            if not options:
                continue
            
            # Sort by score descending and keep top candidates
            options.sort(key=lambda x: x[0], reverse=True)
            
            # Add top candidates from this parent to new beam
            for opt_score, site, new_chain, new_occupied in options[:max(1, beam_width // len(beam))]:
                # Use a hash of the chain to avoid duplicates
                key = tuple(new_chain)
                if key not in seen:
                    seen.add(key)
                    new_beam.append((new_chain, new_occupied, opt_score))
        
        if not new_beam:
            break
        
        # Keep only the top beam_width entries
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
        
        # Update best
        if beam[0][2] > best_score:
            best_score = beam[0][2]
            best_chain = beam[0][0]
    
    return best_chain


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
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


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, max_iterations, max_time):
    """Simple local search: try moving individual residues to nearby sites."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iterations):
        if time.time() > max_time:
            break
        
        improved = False
        # Try moving each residue (except endpoints) to a random neighbour
        for i in range(1, n - 1):
            if time.time() > max_time:
                break
            
            for new_site in _neighbours(tuple(chain[i])):
                # Check if new_site is valid (not occupied by other residues)
                valid = True
                for j, p in enumerate(chain):
                    if j != i and tuple(p) == new_site:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check unit step constraints
                if i > 0:
                    dx = abs(new_site[0] - chain[i-1][0]) + abs(new_site[1] - chain[i-1][1]) + abs(new_site[2] - chain[i-1][2])
                    if dx != 1:
                        continue
                if i < n - 1:
                    dx = abs(new_site[0] - chain[i+1][0]) + abs(new_site[1] - chain[i+1][1]) + abs(new_site[2] - chain[i+1][2])
                    if dx != 1:
                        continue
                
                # Try the move
                old_site = chain[i]
                chain[i] = list(new_site)
                new_energy = _energy(sequence, chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    improved = True
                else:
                    chain[i] = list(old_site)
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search + local search + greedy restarts."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    max_time = _START_TIME + 330.0 - _DEADLINE_MARGIN
    n = len(sequence)
    
    if n == 0:
        return []
    
    # Strategy 1: Beam search with varying widths
    best_chain = None
    best_energy = 0  # We want minimum energy (most negative)
    
    # Try beam search with different beam widths
    for bw in [5, 10, 20]:
        if time.time() > max_time:
            break
        
        chain = _beam_search(sequence, dimension, bw, max_time)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
        
        # Local search on the beam search result
        if chain is not None and len(chain) == n and time.time() < max_time:
            ls_chain, ls_energy = _local_search(sequence, chain, 100, max_time)
            if ls_energy < best_energy:
                best_energy = ls_energy
                best_chain = ls_chain
    
    # Strategy 2: Greedy restarts
    rng = random.Random(12345 + len(sequence))
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > max_time:
            break
        
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            
            # Local search on greedy result
            if time.time() < max_time:
                ls_chain, ls_energy = _local_search(sequence, chain, 50, max_time)
                if ls_energy < best_energy:
                    best_energy = ls_energy
                    best_chain = ls_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
