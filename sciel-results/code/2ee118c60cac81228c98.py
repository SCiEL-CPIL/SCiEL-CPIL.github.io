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
            # Heuristic: strongly prefer high gain, then moderate free space
            score = gain * 3.0 + free * 0.3 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Lower is better (more negative)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, occupied, i, old_pos, new_pos):
    """
    Calculate the change in energy when moving residue i from old_pos to new_pos.
    Assumes occupied reflects the state where i is at old_pos.
    """
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

def _is_valid_move(chain, occupied, i, new_pos):
    """Check if moving residue i to new_pos is valid (self-avoiding and no invalid adjacencies)."""
    if new_pos in occupied:
        return False
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return False
    return True

def _regrow_segment(sequence, chain, occupied, i_start, i_end, rng, max_steps=50):
    """
    Re-grow the segment from index i_start to i_end (inclusive).
    The prefix (0..i_start-1) and suffix (i_end+1..n-1) remain fixed.
    Returns (new_segment, success) where new_segment is a list of positions for i_start..i_end.
    """
    n = len(chain)
    
    # Build fixed_occupied as a dict: position -> index
    fixed_occupied = {}
    for i, p in enumerate(chain):
        if i < i_start or i > i_end:
            fixed_occupied[p] = i
    
    # Anchor is the position just before the segment
    if i_start == 0:
        return None, False # Skip regrow from start for simplicity/robustness
    
    anchor_pos = chain[i_start - 1]
    
    # Initialize current_occupied with fixed positions
    current_occupied = dict(fixed_occupied)
    
    segment = []
    current_pos = anchor_pos
    
    for idx in range(i_start, i_end + 1):
        options = []
        for site in _neighbours2d(current_pos):
            if site in current_occupied:
                continue
            
            # Check adjacency with fixed residues and segment residues
            valid = True
            for q in _neighbours2d(site):
                if q in current_occupied:
                    j = current_occupied[q]
                    if abs(j - idx) > 1:
                        valid = False
                        break
            if not valid:
                continue
            
            # Score: gain from H-H contacts
            gain = 0
            if sequence[idx] == "H":
                for q in _neighbours2d(site):
                    if q in current_occupied:
                        j = current_occupied[q]
                        if abs(j - idx) > 1 and sequence[j] == "H":
                            gain += 1
            
            free = sum(1 for q in _neighbours2d(site) if q not in current_occupied)
            score = gain * 3.0 + free * 0.3 + rng.random()
            options.append((score, site))
        
        if not options:
            return None, False
        
        site = max(options)[1]
        current_occupied[site] = idx
        segment.append(site)
        current_pos = site
    
    return segment, True

def _local_search(sequence, chain, max_iter=5000, rng=None):
    """Local search with single-residue moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    no_improve_count = 0
    max_no_improve = 150
    
    for _ in range(max_iter):
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        i = rng.randint(1, n - 2)
        current_pos = best_chain[i]
        for new_pos in _neighbours2d(current_pos):
            if not _is_valid_move(best_chain, occupied, i, new_pos):
                continue
            
            delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
            new_energy = best_energy + delta
            
            if new_energy < best_energy:
                best_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                best_energy = new_energy
                improved = True
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    return best_chain, best_energy

def _simulated_annealing_with_regrow(sequence, chain, max_iter=10000, initial_temp=5.0, final_temp=0.01, rng=None, deadline=None):
    """Simulated Annealing with both single-residue moves and segment regrowth."""
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
        if deadline is not None and time.time() > deadline:
            break
        temp = initial_temp * (cooling_rate ** iteration)
        if temp < 0.001:
            break
        
        # Choose move type: 70% single, 30% regrow
        if n >= 6 and rng.random() < 0.3:
            # Regrow a random segment of length 3-8
            seg_len = min(rng.randint(3, 8), n - 2)
            i_start = rng.randint(1, n - seg_len - 1)
            i_end = i_start + seg_len - 1
            
            new_segment, success = _regrow_segment(sequence, current_chain, occupied, i_start, i_end, rng)
            if not success:
                continue
            
            # Compute energy change by rebuilding
            new_chain = current_chain[:i_start] + new_segment + current_chain[i_end + 1:]
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                current_chain = new_chain
                occupied = {p: i for i, p in enumerate(current_chain)}
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
        else:
            # Single residue move
            i = rng.randint(1, n - 2)
            current_pos = current_chain[i]
            for new_pos in _neighbours2d(current_pos):
                if not _is_valid_move(current_chain, occupied, i, new_pos):
                    continue
                
                delta = _calc_delta_energy(sequence, occupied, i, current_pos, new_pos)
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
    """Best conformation using randomized greedy growths, local search, and SA with regrow."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Adjust parameters based on sequence length
    if n <= 30:
        num_restarts = 600
        ls_iter = 1500
    elif n <= 60:
        num_restarts = 300
        ls_iter = 800
    else:
        num_restarts = 120
        ls_iter = 400
    
    # Phase 1: Multiple restarts with local search
    for attempt in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        chain, energy = _local_search(sequence, chain, max_iter=ls_iter, rng=rng)
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # Phase 2: SA with regrow on the best solution
    if best is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        # Estimate iterations per second
        est_iter_per_sec = 800 if n < 40 else 400
        sa_iter = int(remaining_time * est_iter_per_sec)
        sa_iter = max(2000, min(sa_iter, 80000))
        
        best, best_energy = _simulated_annealing_with_regrow(
            sequence, best, max_iter=sa_iter, initial_temp=4.0, final_temp=0.01, rng=rng,
            deadline=deadline - 2.0
        )
    
    # Phase 3: Final local search polish
    if best is not None and time.time() < deadline - 1.0:
        best, best_energy = _local_search(sequence, best, max_iter=2000, rng=rng)
    
    if best is None:
        best = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
