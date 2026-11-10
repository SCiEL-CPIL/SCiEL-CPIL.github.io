"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 165.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute energy from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, occupied, i, new_site):
    """Compute energy delta for moving residue i to new_site. Returns delta or None if invalid."""
    n = len(chain)
    old_site = chain[i]
    
    # Check if new_site is free (not occupied by another residue)
    if new_site in occupied and occupied[new_site] != i:
        return None
    
    # Check connectivity with neighbors
    if i > 0:
        px, py = chain[i-1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None
    if i < n - 1:
        px, py = chain[i+1]
        nx, ny = new_site
        if abs(nx - px) + abs(ny - py) != 1:
            return None
    
    # Compute delta: remove old contacts involving i, add new contacts involving i
    delta = 0
    is_h = sequence[i] == "H"
    
    # Remove contacts that residue i had at old_site
    if is_h:
        for q in _neighbours2d(old_site):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                if abs(i - j) > 1:
                    delta += 1  # losing a contact increases energy (less negative)
    
    # Add contacts that residue i will have at new_site
    if is_h:
        for q in _neighbours2d(new_site):
            j = occupied.get(q)
            if j is not None and j != i and sequence[j] == "H":
                if abs(i - j) > 1:
                    delta -= 1  # gaining a contact decreases energy (more negative)
    
    return delta


def _grow_diverse(sequence, dimension, rng, strategy=0):
    """Randomized chain growth with different strategies."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied:
                    free += 1
            
            if strategy == 0:
                # Greedy with some randomness
                score = gain * 3.0 + 0.1 * free + rng.random() * 0.5
            elif strategy == 1:
                # More random
                score = gain * 1.0 + 0.05 * free + rng.random() * 2.0
            elif strategy == 2:
                # Prefer compactness (closer to origin)
                dist = abs(site[0]) + abs(site[1])
                score = gain * 2.0 - 0.1 * dist + rng.random() * 0.5
            else:
                score = gain * 2.0 + 0.1 * free + rng.random()
            
            options.append((score, site))
        
        if not options:
            return None
        
        if strategy == 1 and len(options) > 1:
            # Pick randomly from top candidates
            options.sort(reverse=True)
            top_k = min(3, len(options))
            _, site = options[rng.randint(0, top_k - 1)]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _simulated_annealing(sequence, chain, occupied, t_start=1.0, t_end=0.01, max_iters=2000, rng=None):
    """Simulated annealing with single-residue and segment moves."""
    n = len(chain)
    if n < 3:
        return chain, occupied, _compute_energy(sequence, chain)
    if rng is None:
        rng = random.Random()
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_occupied = dict(occupied)
    best_energy = current_energy
    
    t = t_start
    cooling = (t_start - t_end) / max(1, max_iters)
    
    for it in range(max_iters):
        if time.time() >= _DEADLINE:
            break
        
        # Choose move type
        if rng.random() < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)
            x, y = chain[i]
            candidates = []
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                new_site = (x + dx, y + dy)
                if new_site in occupied and occupied[new_site] != i:
                    continue
                # Check connectivity
                if i > 0:
                    px, py = chain[i-1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                if i < n - 1:
                    px, py = chain[i+1]
                    if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                        continue
                candidates.append(new_site)
            
            if not candidates:
                continue
            
            new_site = candidates[rng.randint(0, len(candidates) - 1)]
            delta = _energy_delta(sequence, chain, occupied, i, new_site)
            if delta is None:
                continue
            
            # Metropolis criterion
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                # Accept move
                old_site = chain[i]
                del occupied[old_site]
                chain[i] = new_site
                occupied[new_site] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
        else:
            # Segment reversal move
            if n < 4:
                continue
            # Pick a segment to reverse
            seg_len = rng.randint(2, min(6, n - 2))
            start_idx = rng.randint(1, n - 1 - seg_len)
            end_idx = start_idx + seg_len - 1
            
            # Check if reversal is valid
            new_segment = list(reversed(chain[start_idx:end_idx+1]))
            
            # Check connectivity at boundaries
            valid = True
            if start_idx > 0:
                prev_point = chain[start_idx - 1]
                first_new = new_segment[0]
                if abs(first_new[0] - prev_point[0]) + abs(first_new[1] - prev_point[1]) != 1:
                    valid = False
            if valid and end_idx < n - 1:
                next_point = chain[end_idx + 1]
                last_new = new_segment[-1]
                if abs(last_new[0] - next_point[0]) + abs(last_new[1] - next_point[1]) != 1:
                    valid = False
            if not valid:
                continue
            
            # Check self-avoidance: the reversed segment must not collide with itself or other parts
            # Remove the old segment positions from occupied temporarily for checking
            old_positions = set(chain[start_idx:end_idx+1])
            new_positions = set(new_segment)
            
            # Check if new positions collide with parts outside the segment
            for pos in new_positions:
                if pos in occupied:
                    occ_idx = occupied[pos]
                    if not (start_idx <= occ_idx <= end_idx):
                        valid = False
                        break
            if not valid:
                continue
            
            # Check internal collisions in new_segment
            if len(new_positions) != len(new_segment):
                continue
            
            # Compute new energy
            new_chain = chain[:start_idx] + new_segment + chain[end_idx+1:]
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
                # Accept
                for idx in range(start_idx, end_idx + 1):
                    del occupied[chain[idx]]
                chain[start_idx:end_idx+1] = new_segment
                for idx in range(start_idx, end_idx + 1):
                    occupied[chain[idx]] = idx
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
        
        t = max(t_end, t - cooling)
    
    return best_chain, best_occupied, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search using SA."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Will be negative
    
    # Try multiple growth strategies and SA refinements
    strategies = [0, 1, 2, 3]
    attempts_per_strategy = 5
    
    for strat in strategies:
        for attempt in range(attempts_per_strategy):
            if time.time() >= _DEADLINE:
                break
            
            chain = _grow_diverse(sequence, dimension, rng, strategy=strat)
            if chain is None:
                continue
            
            occupied = {p: i for i, p in enumerate(chain)}
            
            # Simulated annealing
            t_start = 1.0 if n <= 50 else 0.5
            max_iters = max(500, min(5000, 200 * n))
            chain, occupied, energy = _simulated_annealing(sequence, chain, occupied, t_start, 0.01, max_iters, rng)
            
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
            
            if time.time() >= _DEADLINE:
                break
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
