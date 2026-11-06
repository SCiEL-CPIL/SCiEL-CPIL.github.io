"""Improved HP lattice folding with beam growth and multi-neighborhood local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta(sequence, chain, old_positions, new_positions):
    """
    Compute the change in energy when moving residues at indices in old_positions
    to new_positions. old_positions and new_positions are dicts: index -> point.
    Returns the delta (new_energy - old_energy).
    """
    if not old_positions:
        return 0
    
    # Build the set of positions occupied by residues that are NOT being moved
    # and the set of positions occupied by residues that ARE being moved
    n = len(chain)
    
    # Old state
    old_occ = {p: i for i, p in enumerate(chain)}
    new_chain = list(chain)
    for idx, pos in new_positions.items():
        new_chain[idx] = pos
    new_occ = {p: i for i, p in enumerate(new_chain)}
    
    # To compute delta efficiently, we only need to consider contacts involving
    # the moved residues. But for simplicity and correctness with small moves,
    # let's just recompute energy on the affected region.
    # Actually, for segment moves, recomputing full energy is O(n) which is fine for n<=100.
    # But for many moves, we want it fast.
    # Let's use the full recompute for now since n is small.
    return _energy(sequence, new_chain) - _energy(sequence, chain)

def _grow_beam(sequence, dimension, rng, beam_width=20):
    """
    Beam search growth: keep the top beam_width partial chains at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Each entry in the beam is (chain, energy_so_far)
    # chain is a list of points
    beam = [([(0, 0)], 0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, energy in beam:
            current = chain[-1]
            occupied = set(chain)
            
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                new_chain = chain + [site]
                new_energy = energy
                
                # Check for new contacts formed by adding residue i
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find the index of q in the chain
                            # This is O(n) but n is small
                            for j, p in enumerate(chain):
                                if p == q and j < i - 1 and sequence[j] == 'H':
                                    new_energy -= 1
                                    break
                
                next_beam.append((new_chain, new_energy))
        
        if not next_beam:
            return None
        
        # Sort by energy (lower is better) and keep top beam_width
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]
    
    # Return the best chain
    if beam:
        return beam[0][0]
    return None

def _grow_greedy(sequence, dimension, rng):
    """
    Greedy growth with randomization.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        # Sometimes pick the second best to add diversity
        if len(options) > 1 and rng.random() < 0.2:
            site = options[1][1]
        else:
            site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, rng, max_steps=2000):
    """
    Local search with multiple move types:
    1. Single residue relocation
    2. Segment relocation (short segments)
    3. 180-degree flip of a segment
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        # Choose a move type
        move_type = rng.choice(['single', 'segment', 'flip'])
        
        if move_type == 'single':
            # Try to move a single residue (not endpoints)
            idx = rng.randint(1, n - 2)
            old_pos = best_chain[idx]
            left = best_chain[idx - 1]
            right = best_chain[idx + 1]
            
            # Candidate positions: neighbors of left and right that are free
            candidates = set()
            for site in _neighbours(left):
                if site != old_pos and site != right:
                    # Check if site is occupied by any other residue
                    occupied = {p: i for i, p in enumerate(best_chain) if i != idx}
                    if site not in occupied:
                        candidates.add(site)
            
            for site in _neighbours(right):
                if site != old_pos and site != left:
                    occupied = {p: i for i, p in enumerate(best_chain) if i != idx}
                    if site not in occupied:
                        candidates.add(site)
            
            for new_pos in candidates:
                new_chain = list(best_chain)
                new_chain[idx] = new_pos
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    break
        
        elif move_type == 'segment':
            # Try to move a short segment (length 2-5)
            seg_len = rng.randint(2, min(5, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            left_neighbor = best_chain[start_idx - 1]
            right_neighbor = best_chain[end_idx + 1]
            
            # Find candidate start positions
            occupied_by_others = {p: i for i, p in enumerate(best_chain) 
                                  if i < start_idx or i > end_idx}
            
            start_candidates = []
            for site in _neighbours(left_neighbor):
                if site not in occupied_by_others:
                    start_candidates.append(site)
            
            if not start_candidates:
                continue
            
            # Try to find a valid path for the segment
            # Use a simple DFS with limited depth
            for start_pos in start_candidates:
                # DFS to find a path of length seg_len ending near right_neighbor
                visited = {start_pos}
                stack = [(start_pos, [start_pos])]
                found = False
                
                steps = 0
                max_steps_dfs = 50
                while stack and steps < max_steps_dfs:
                    steps += 1
                    current, path = stack.pop()
                    
                    if len(path) == seg_len:
                        # Check if current is adjacent to right_neighbor
                        for q in _neighbours(current):
                            if q == right_neighbor:
                                # Found a valid segment
                                new_chain = list(best_chain)
                                for i, p in enumerate(path):
                                    new_chain[start_idx + i] = p
                                new_energy = _energy(sequence, new_chain)
                                if new_energy < best_energy:
                                    best_chain = new_chain
                                    best_energy = new_energy
                                    found = True
                                break
                        if found:
                            break
                        continue
                    
                    for site in _neighbours(current):
                        if site in visited or site in occupied_by_others:
                            continue
                        visited.add(site)
                        stack.append((site, path + [site]))
                
                if found:
                    break
        
        elif move_type == 'flip':
            # Try to flip a short segment (reverse direction)
            seg_len = rng.randint(2, min(5, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            # Reverse the segment
            new_chain = list(best_chain)
            segment = new_chain[start_idx:end_idx + 1]
            segment.reverse()
            new_chain[start_idx:end_idx + 1] = segment
            
            # Check self-avoidance
            if len(set(new_chain)) == len(new_chain):
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
    
    return best_chain

def fold(sequence, dimension):
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam search growth (best initial solution)
    if _time_remaining() > 5.0:
        beam_chain = _grow_beam(sequence, dimension, rng, beam_width=20)
        if beam_chain is not None:
            energy = _energy(sequence, beam_chain)
            best_chain = beam_chain
            best_energy = energy
    
    # Phase 2: Greedy growth restarts
    growth_deadline = _time_remaining() * 0.5
    iterations = 0
    max_iterations = 5000
    
    while _time_remaining() > growth_deadline and iterations < max_iterations:
        iterations += 1
        
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Local search refinement
    if _time_remaining() > 3.0:
        best_chain = _local_search(sequence, best_chain, rng, max_steps=3000)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 4: Second local search pass with fresh randomness
    if _time_remaining() > 5.0:
        rng2 = random.Random(99999 + len(sequence))
        best_chain = _local_search(sequence, best_chain, rng2, max_steps=3000)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
