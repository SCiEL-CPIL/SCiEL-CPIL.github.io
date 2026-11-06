"""HP lattice protein folding on cubic lattice - CSP backtracking with dynamic variable ordering."""
# EVOLVE-BLOCK-START
import time
import random
import math

START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a full chain (negative number of H-H contacts)."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=50, time_budget=15.0):
    """
    Beam search for HP folding.
    Returns best full chain or None if timeout.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    deadline = START_TIME + time_budget
    
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_energy)]
    
    for i in range(1, n):
        next_beam = []
        
        for chain, occupied, energy in beam:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            new_energy -= 1
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_energy))
        
        if not next_beam:
            return None
        
        # Prune to beam_width
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
        
        # Time check
        if time.time() > deadline:
            return None
    
    if not beam:
        return None
        
    return beam[0][0]

def _greedy_grow(sequence, dimension, rng):
    """Randomized greedy growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
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
            score = gain * 2.0 + free * 0.3 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            break
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        chosen = random.choice(options[:top_k])[1]
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain

def _local_refine(sequence, chain, dimension, max_iters=50):
    """
    Local refinement: try to improve energy by moving individual residues.
    """
    n = len(chain)
    if n == 0:
        return chain, 0
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for iteration in range(max_iters):
        improved = False
        occupied = {p: i for i, p in enumerate(best_chain)}
        
        for i in range(n):
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
            
            pos = best_chain[i]
            
            for new_pos in _neighbours(pos):
                if new_pos in occupied:
                    continue
                
                valid = True
                if i > 0:
                    prev_pos = best_chain[i-1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, prev_pos))
                    if dist != 1:
                        valid = False
                
                if valid and i < n-1:
                    next_pos = best_chain[i+1]
                    dist = sum(abs(a-b) for a, b in zip(new_pos, next_pos))
                    if dist != 1:
                        valid = False
                
                if not valid:
                    continue
                
                old_pos = best_chain[i]
                best_chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                if len(occupied) != n:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
                    continue
                
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                else:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
            
            if time.time() > START_TIME + TIME_LIMIT * 0.85:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _csp_backtrack(sequence, dimension, max_time=10.0):
    """
    CSP-based backtracking search with dynamic variable ordering.
    Variables: residue positions (0 to n-1)
    Constraints: self-avoidance, connectivity (unit steps)
    Objective: maximize H-H contacts (minimize energy)
    """
    n = len(sequence)
    if n == 0:
        return []
    
    deadline = START_TIME + max_time
    rng = random.Random(12345)
    
    # We'll do a guided backtracking: place residues in an order that
    # minimizes the number of valid positions first (MRV heuristic)
    # But for a linear chain, the natural order is 0,1,2,...,n-1
    # So we use a hybrid: try to place residues that have fewer options first
    
    # Start with residue 0 at origin
    chain = [(0,) * dimension]
    occupied = {(0,) * dimension: 0}
    current_energy = 0
    
    # For simplicity and effectiveness, we'll do a depth-first search with
    # backtracking and a node limit, guided by energy
    
    best_chain = None
    best_energy = 0
    nodes_visited = 0
    max_nodes = 50000  # Limit to prevent infinite loops
    
    def backtrack(i, chain, occupied, energy):
        nonlocal best_chain, best_energy, nodes_visited
        
        if time.time() > deadline or nodes_visited > max_nodes:
            return
        
        nodes_visited += 1
        
        if i == n:
            # Complete chain found
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            return
        
        last_pos = chain[-1]
        # Get all valid positions for residue i
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Calculate energy gain
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            candidates.append((gain, site))
        
        # Sort by gain (higher is better) with some randomness
        candidates.sort(key=lambda x: -x[0])
        rng.shuffle(candidates)  # Add randomness to break ties
        
        for gain, site in candidates:
            new_chain = chain + [site]
            new_occupied = dict(occupied)
            new_occupied[site] = i
            new_energy = energy - gain
            
            backtrack(i + 1, new_chain, new_occupied, new_energy)
            
            if time.time() > deadline or nodes_visited > max_nodes:
                return
    
    backtrack(1, chain, occupied, current_energy)
    
    if best_chain is not None:
        return best_chain
    
    # Fallback: return the straight line if backtracking didn't find anything
    return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

def fold(sequence, dimension):
    """
    Best conformation using CSP backtracking, beam search, greedy growth, and local refinement.
    """
    n = len(sequence)
    rng = random.Random(42 + n)
    
    # Determine time budget allocation
    if n <= 48:
        csp_time = 15.0
        beam_time = 10.0
        beam_width = 80
    elif n <= 80:
        csp_time = 12.0
        beam_time = 8.0
        beam_width = 60
    else:
        csp_time = 8.0
        beam_time = 6.0
        beam_width = 50
    
    # Try CSP backtracking first
    best_chain = _csp_backtrack(sequence, dimension, csp_time)
    
    if best_chain is None or len(best_chain) != n:
        # Fallback to beam search
        best_chain = _beam_search(sequence, dimension, beam_width, beam_time)
    
    if best_chain is None or len(best_chain) != n:
        # Fallback to greedy growth
        best_chain = _greedy_grow(sequence, dimension, rng)
        
        if len(best_chain) != n:
            # Last resort: straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Local refinement
    best_chain, best_energy = _local_refine(sequence, best_chain, dimension, max_iters=30)
    
    # If we have time, try multiple greedy starts and keep the best
    elapsed = time.time() - START_TIME
    if elapsed < TIME_LIMIT * 0.6:
        rng2 = random.Random(99999 + n)
        
        num_restarts = min(5, int((TIME_LIMIT * 0.6 - (time.time() - START_TIME)) / 2))
        
        for restart in range(max(1, num_restarts)):
            if time.time() - START_TIME > TIME_LIMIT * 0.8:
                break
            
            candidate = _greedy_grow(sequence, dimension, rng2)
            if len(candidate) != n:
                continue
            
            candidate, cand_energy = _local_refine(sequence, candidate, dimension, max_iters=20)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_chain = candidate
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
