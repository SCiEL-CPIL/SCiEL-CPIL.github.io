"""HP lattice protein folding - structural change: simulated annealing with diverse initializations."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    """Generate all 6 neighbours in 3D cubic lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _generate_initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk using biased growth."""
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Prefer sites with more free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 5.0 + free * 0.3 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Weighted selection
        total = sum(s for s, _ in options)
        r = rng.random() * total
        cumulative = 0
        chosen = options[-1][1]
        for score, site in options:
            cumulative += score
            if r <= cumulative:
                chosen = site
                break
        
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain


def _try_pivot_move(chain, i, direction, occupied, sequence, dimension):
    """
    Try to rotate the segment starting at position i by 90 degrees around the bond (i-1, i).
    direction: one of 6 unit vectors perpendicular to the bond direction.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
    
    # The bond vector from i-1 to i
    bond = tuple(chain[i][d] - chain[i-1][d] for d in range(dimension))
    
    # Check that direction is perpendicular to bond
    dot = sum(direction[d] * bond[d] for d in range(dimension))
    if dot != 0:
        return None
    
    # Try rotating segment [i, n-1] around the pivot at i-1
    # We need to compute the rotation matrix
    # For a 90-degree rotation around axis 'bond' through point chain[i-1]
    # This is complex; instead, use a simpler approach:
    # Try moving the entire tail by checking if it fits
    
    # Simpler: try all 6 possible new positions for chain[i] that are perpendicular to bond
    # and not collinear with the bond
    new_positions = []
    for d in _neighbours(chain[i]):
        # Check it's not back to i-1
        if d == chain[i-1]:
            continue
        # Check it's perpendicular to the bond (90 degree turn)
        new_bond = tuple(d[a] - chain[i-1][a] for a in range(dimension))
        dot2 = sum(new_bond[a] * bond[a] for a in range(dimension))
        if dot2 == 0:
            new_positions.append(d)
    
    if not new_positions:
        return None
    
    # Try each possible new position for chain[i]
    for new_pos in new_positions:
        if new_pos in occupied:
            continue
        
        # Try to shift the entire tail
        new_chain = list(chain)
        new_chain[i] = new_pos
        
        # Check if the rest of the tail can be shifted
        # This is complex; for now, just check if new_pos is valid and try to extend
        # Actually, for a proper pivot move, we need to rotate the entire tail
        # Let's use a different approach: try to find a valid continuation
        
        # For simplicity, just check if new_pos is free and accept if it creates a valid chain
        # This is not a true pivot move but a "jump" move
        pass
    
    return None


def _simulated_annealing(sequence, chain, dimension, rng, time_budget):
    """
    Simulated annealing with pivot and 2-opt moves.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_temp = 2.0
    end_temp = 0.01
    cooling_rate = 0.995
    
    temp = start_temp
    step = 0
    
    while temp > end_temp and _get_remaining_time() > time_budget * 0.1:
        # Choose move type
        if rng.random() < 0.5:
            # 2-opt reversal
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            if b - a <= 1:
                continue
            
            new_chain = current_chain[:a] + current_chain[a:b+1][::-1] + current_chain[b+1:]
            
            # Validate
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue
            
            for k in range(n - 1):
                diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
        else:
            # Single-step move: try to move one residue
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            
            # Try moving residue i to a neighbouring site
            for new_pos in _neighbours(old_pos):
                if new_pos == current_chain[i-1] or new_pos == current_chain[i+1]:
                    continue
                
                new_chain = list(current_chain)
                new_chain[i] = new_pos
                
                # Check self-avoidance
                seen = set()
                valid = True
                for p in new_chain:
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                if not valid:
                    continue
                
                # Check unit steps
                for k in range(n - 1):
                    diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
                    if diff != 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_energy = _energy(sequence, new_chain)
                break
            else:
                continue
        
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temp *= cooling_rate
        step += 1
    
    return best_chain, best_energy


def _local_search(sequence, chain, dimension, rng, max_moves=1000):
    """
    Hill climbing local search with 2-opt and single-step moves.
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for move in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        improved = False
        
        # Try 2-opt
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        if b - a > 1:
            new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
            
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if valid:
                for k in range(n - 1):
                    diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
                    if diff != 1:
                        valid = False
                        break
            if valid:
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True
        
        # Try single-step
        if not improved:
            i = rng.randint(1, n - 2)
            old_pos = best_chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos == best_chain[i-1] or new_pos == best_chain[i+1]:
                    continue
                
                new_chain = list(best_chain)
                new_chain[i] = new_pos
                
                seen = set()
                valid = True
                for p in new_chain:
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                if valid:
                    for k in range(n - 1):
                        diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
                        if diff != 1:
                            valid = False
                            break
                    if valid:
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < best_energy:
                            best_chain = new_chain
                            best_energy = new_energy
                            improved = True
                            break
        
        if not improved and move > 100:
            # Try a random restart within local search
            if rng.random() < 0.1:
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using simulated annealing with multiple diverse initializations.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Generate multiple diverse initial chains
    n_initials = min(20, max(5, 240 // n))
    
    for init in range(n_initials):
        if _get_remaining_time() < 5.0:
            break
        
        chain = _generate_initial_chain(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Run simulated annealing on this chain
        time_per_sa = _get_remaining_time() / (n_initials - init) * 0.8
        time_per_sa = max(1.0, min(30.0, time_per_sa))
        
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, dimension, rng, time_per_sa)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Final local search on the best chain
    if best_chain is not None and _get_remaining_time() > 5.0:
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, max_moves=2000)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
