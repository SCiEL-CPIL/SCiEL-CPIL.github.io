# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_csp(sequence, dimension, time_limit=10.0, seed=42):
    n = len(sequence)
    start_time = time.time()
    random.seed(seed)
    
    best_chain = None
    best_energy = 0  # We want most negative (most contacts)
    
    # Precompute H positions for faster checking? 
    # is_h = [c == 'H' for c in sequence]
    
    def _search(chain, occupied, current_energy):
        nonlocal best_chain, best_energy, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        i = len(chain)
        if i == n:
            # Complete chain found
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return
            
        last_pos = chain[-1]
        current_residue = sequence[i]
        
        # Generate candidates
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
                
            # Heuristic Scoring
            score = 0.0
            
            # 1. Immediate Contact Gain
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            score += 10.0 # Reduced weight, let's balance with trap avoidance
            
            # 2. Free Space / Trap Avoidance
            free_count = 0
            # Check if next step is possible
            next_free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
                    # Lookahead: if we go to q, how many options do we have?
                    # This is expensive, so only do it if free_count is low?
                    # Or just use free_count as proxy.
            
            # If placing here leaves 0 free neighbors for the next residue (and i < n-1), it's a dead end
            if free_count == 0 and i < n - 1:
                continue # Prune immediately
            
            # Stronger penalty for trapping
            # If free_count is 1, we are forced to go to that specific neighbor.
            # If that neighbor is a dead end (0 free neighbors for the step after), it's very bad.
            if free_count == 1 and i < n - 2:
                # Find the only free neighbor
                only_free = None
                for q in _neighbours(site):
                    if q not in occupied:
                        only_free = q
                        break
                # Check if only_free has any free neighbors (for the step after)
                # Note: 'site' will be occupied, so we check neighbors of only_free excluding 'site'
                has_escape = False
                for r in _neighbours(only_free):
                    if r != site and r not in occupied:
                        has_escape = True
                        break
                if not has_escape:
                    # Dead end detected one step ahead
                    score -= 100.0
                else:
                    # Forced move, moderate penalty
                    score -= 5.0
            
            # Prefer sites with more free space
            score += free_count * 2.0
            
            # 3. Randomness for diversity
            score += random.random() * 1.0 # Reduced noise to rely more on heuristic
            
            candidates.append((score, site))
            
        if not candidates:
            return
            
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            # Try placing at site
            chain.append(site)
            occupied[site] = i
            
            # Calculate new energy increment
            delta_energy = 0
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (improves) by 1 per contact
            
            _search(chain, occupied, current_energy + delta_energy)
            
            # Backtrack
            chain.pop()
            del occupied[site]
            
            if time.time() - start_time > time_limit:
                return

    # Run search
    init_chain = [(0, 0)]
    init_occupied = {(0, 0): 0}
    
    if n > 1:
        init_chain.append((1, 0))
        init_occupied[(1, 0)] = 1
        _search(init_chain, init_occupied, 0)
    else:
        best_chain = init_chain[:]
        best_energy = 0

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence using CSP-inspired search with local refinement."""
    n = len(sequence)
    
    # Time budget per sequence - Increased to allow deeper search
    if n <= 20:
        time_limit = 12.0
    elif n <= 40:
        time_limit = 10.0
    elif n <= 60:
        time_limit = 8.0
    else:
        time_limit = 6.0
        
    # Multi-restart strategy
    # For smaller sequences, more restarts with less time each?
    # For larger sequences, fewer restarts with more time each to find a complete solution?
    if n <= 40:
        num_restarts = 8
    else:
        num_restarts = 5
        
    time_per_restart = time_limit / num_restarts
    
    best_chain = None
    best_energy = 0
    global_start = time.time()
    
    for i in range(num_restarts):
        seed = 42 + i
        chain, energy = _fold_csp(sequence, dimension, time_limit=time_per_restart, seed=seed)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Check if we are running out of total time
        if time.time() - global_start > time_limit * 0.95:
            break
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
