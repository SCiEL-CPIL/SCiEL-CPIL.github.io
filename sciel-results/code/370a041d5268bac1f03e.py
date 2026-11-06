"""HP lattice protein folding with multi-start growth and local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 205.0 - (time.time() - _START_TIME)

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

def _grow_greedy(sequence, dimension, rng, temperature=1.0):
    """Grow the chain using a greedy/randomized approach."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
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
            
            # Count free neighbors to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort by score and pick the best with some randomness
        options.sort(key=lambda x: x[0], reverse=True)
        
        # Sometimes pick a non-best option to explore more
        if temperature > 0.5 and rng.random() < 0.2:
            idx = min(len(options) - 1, max(0, int(rng.gauss(0, 1))))
            site = options[idx][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, rng, time_limit=5.0):
    """
    Local search using single-residue moves (1-0-1 moves) and segment moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    start_time = time.time()
    
    while time.time() - start_time < time_limit and _time_remaining() > 1.0:
        improved = False
        
        # Try 1-0-1 moves (move a single residue to an adjacent empty spot)
        for i in range(1, n - 1):
            if time.time() - start_time > time_limit:
                break
            
            old_pos = best_chain[i]
            for new_pos in _neighbours(old_pos):
                # Check if new_pos is free (not occupied by any other residue)
                if new_pos == best_chain[i-1] or new_pos == best_chain[i+1]:
                    continue
                # Check if new_pos is occupied by any other residue
                occupied = set(best_chain)
                occupied.discard(old_pos)
                if new_pos in occupied:
                    continue
                
                # Try the move
                best_chain[i] = new_pos
                new_energy = _energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                else:
                    best_chain[i] = old_pos
        
        # Try segment moves (move a short segment)
        if not improved and n >= 5:
            for _ in range(50):  # Try a few segment moves
                if time.time() - start_time > time_limit:
                    break
                
                seg_len = rng.randint(2, min(5, n - 2))
                start_idx = rng.randint(1, n - seg_len - 1)
                end_idx = start_idx + seg_len - 1
                
                # Get the segment
                segment = best_chain[start_idx:end_idx+1]
                
                # Try to relocate the segment
                left_neighbor = best_chain[start_idx - 1]
                right_neighbor = best_chain[end_idx + 1]
                
                # Try starting from neighbors of left_neighbor
                for start_pos in _neighbours(left_neighbor):
                    if start_pos in best_chain or start_pos == right_neighbor:
                        continue
                    
                    # Check if start_pos is free
                    occupied = set(best_chain)
                    for p in segment:
                        occupied.discard(p)
                    
                    if start_pos in occupied:
                        continue
                    
                    # Try to grow a path from start_pos to a neighbor of right_neighbor
                    path = [start_pos]
                    visited = {start_pos}
                    current = start_pos
                    success = False
                    
                    for step in range(1, seg_len):
                        neighbors = [p for p in _neighbours(current) 
                                     if p not in visited and p not in occupied]
                        if not neighbors:
                            break
                        next_pos = rng.choice(neighbors)
                        path.append(next_pos)
                        visited.add(next_pos)
                        current = next_pos
                    
                    if len(path) == seg_len:
                        # Check if we ended at a neighbor of right_neighbor
                        if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                            # Apply the move
                            new_chain = list(best_chain)
                            for i, p in enumerate(path):
                                new_chain[start_idx + i] = p
                            
                            new_energy = _energy(sequence, new_chain)
                            if new_energy < best_energy:
                                best_chain = new_chain
                                best_energy = new_energy
                                improved = True
        
        if not improved:
            # No improvement found, break to avoid infinite loop
            break
    
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
    
    # Phase 1: Multi-start growth
    # Use different temperatures and strategies
    growth_deadline = _time_remaining() * 0.6
    
    temperatures = [0.1, 0.5, 1.0, 2.0]
    temp_idx = 0
    
    while _time_remaining() > growth_deadline:
        temp = temperatures[temp_idx % len(temperatures)]
        temp_idx += 1
        
        chain = _grow_greedy(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement
    if _time_remaining() > 2.0:
        # Allocate remaining time to local search
        ls_time = min(_time_remaining() - 1.0, 10.0)
        best_chain = _local_search(sequence, best_chain, rng, time_limit=ls_time)
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
