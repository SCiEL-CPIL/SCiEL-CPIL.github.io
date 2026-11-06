"""HP lattice protein folding on cubic lattice - Monte Carlo chain growth with simulated annealing."""
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

def _mc_fold(sequence, dimension, rng, time_budget=5.0):
    """
    Monte Carlo chain growth with simulated annealing.
    Grows the chain residue by residue, but at each step can backtrack
    and try alternative moves based on temperature.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    deadline = START_TIME + time_budget
    
    # Start with the first residue at origin
    chain = [(0,) * dimension]
    occupied = {(0,) * dimension: 0}
    energy = 0
    
    # Temperature schedule: start high, cool down
    T_start = 2.0
    T_end = 0.1
    cooling_rate = 0.995
    
    current_T = T_start
    
    # Maximum number of MC steps
    max_mc_steps = 20000
    
    mc_step = 0
    
    while len(chain) < n:
        if time.time() > deadline:
            break
        if mc_step >= max_mc_steps:
            break
        
        mc_step += 1
        
        i = len(chain)
        last_pos = chain[-1]
        
        # Find all valid moves
        valid_moves = []
        for site in _neighbours(last_pos):
            if site not in occupied:
                # Calculate energy gain for this move
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                valid_moves.append((site, -gain))
        
        if not valid_moves:
            # Dead end - need to backtrack
            # Try to find an alternative at an earlier position
            backtrack_found = False
            for backtrack_i in range(len(chain) - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                # Remove from chain to backtrack_i
                for k in range(len(chain) - 1, backtrack_i - 1, -1):
                    pos = chain.pop()
                    del occupied[pos]
                
                # Recalculate energy for the truncated chain
                energy = _compute_energy(sequence, chain)
                
                # Try alternative moves at position backtrack_i
                alt_moves = []
                for site in _neighbours(chain[-1]):
                    if site not in occupied:
                        gain = 0
                        if sequence[backtrack_i] == "H":
                            for q in _neighbours(site):
                                j = occupied.get(q)
                                if j is not None and j < backtrack_i - 1 and sequence[j] == "H":
                                    gain += 1
                        alt_moves.append((site, -gain))
                
                if alt_moves:
                    # Pick a random alternative
                    site, delta_e = rng.choice(alt_moves)
                    chain.append(site)
                    occupied[site] = backtrack_i
                    energy += delta_e
                    
                    # Cool down
                    current_T *= cooling_rate
                    backtrack_found = True
                    break
            
            if not backtrack_found:
                # Complete dead end, restart
                chain = [(0,) * dimension]
                occupied = {(0,) * dimension: 0}
                energy = 0
                current_T = T_start
                mc_step = 0
                continue
        
        # Choose move using simulated annealing
        if len(valid_moves) == 1:
            site, delta_e = valid_moves[0]
        else:
            # Boltzmann sampling
            total_weight = 0.0
            weights = []
            for site, delta_e in valid_moves:
                weight = math.exp(-delta_e / max(current_T, 0.01))
                weights.append(weight)
                total_weight += weight
            
            r = rng.random() * total_weight
            cumulative = 0.0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break
            
            site, delta_e = valid_moves[chosen_idx]
        
        chain.append(site)
        occupied[site] = i
        energy += delta_e
        
        # Cool down
        current_T *= cooling_rate
    
    # If we didn't finish the chain, try to complete it greedily
    while len(chain) < n:
        if time.time() > deadline:
            break
        
        i = len(chain)
        last_pos = chain[-1]
        valid_moves = [site for site in _neighbours(last_pos) if site not in occupied]
        
        if not valid_moves:
            break
        
        # Pick the move with best energy gain
        best_site = None
        best_gain = -float('inf')
        for site in valid_moves:
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_site = site
        
        if best_site is None:
            best_site = valid_moves[0]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain if len(chain) == n else None

def _local_refine(sequence, chain, dimension, max_iters=50, time_budget=2.0):
    """
    Local refinement: try to improve energy by moving individual residues.
    """
    n = len(chain)
    if n == 0:
        return chain, 0
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    deadline = START_TIME + time_budget
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        occupied = {p: i for i, p in enumerate(best_chain)}
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            pos = best_chain[i]
            
            # Try moving residue i to a neighboring empty site
            for new_pos in _neighbours(pos):
                if new_pos in occupied:
                    continue
                
                # Check if this move maintains connectivity
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
                
                # Tentatively apply the move
                old_pos = best_chain[i]
                best_chain[i] = new_pos
                occupied.pop(old_pos)
                occupied[new_pos] = i
                
                # Check self-avoidance
                if len(occupied) != n:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
                    continue
                
                # Compute new energy
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    improved = True
                    occupied = {p: i for i, p in enumerate(best_chain)}
                else:
                    best_chain[i] = old_pos
                    occupied.pop(new_pos)
                    occupied[old_pos] = i
            
            if time.time() > deadline:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using Monte Carlo chain growth with simulated annealing
    and local refinement, with multiple restarts.
    """
    n = len(sequence)
    
    # Determine time budget for MC folding based on sequence length
    if n <= 48:
        mc_time = 8.0
        num_restarts = 4
    elif n <= 80:
        mc_time = 5.0
        num_restarts = 3
    else:
        mc_time = 3.0
        num_restarts = 2
    
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT * 0.85:
            break
        
        seed = 42 + restart * 1000 + n
        rng = random.Random(seed)
        
        # Allocate time for MC folding
        mc_budget = mc_time * (0.9 ** restart)  # Slightly less time for later restarts
        
        candidate = _mc_fold(sequence, dimension, rng, time_budget=mc_budget)
        
        if candidate is None or len(candidate) != n:
            continue
        
        # Local refinement
        refine_budget = min(2.0, (TIME_LIMIT * 0.85 - (time.time() - START_TIME)) * 0.5)
        if refine_budget < 0.5:
            refine_budget = 0.5
        
        candidate, cand_energy = _local_refine(
            sequence, candidate, dimension, 
            max_iters=30, 
            time_budget=refine_budget
        )
        
        if cand_energy < best_energy:
            best_energy = cand_energy
            best_chain = candidate
    
    if best_chain is None or len(best_chain) != n:
        # Fallback: simple greedy
        rng = random.Random(12345 + n)
        chain = [(0,) * dimension]
        occupied = {(0,) * dimension: 0}
        
        for i in range(1, n):
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
            chosen = options[0][1]
            occupied[chosen] = i
            chain.append(chosen)
        
        if len(chain) == n:
            best_chain, _ = _local_refine(sequence, chain, dimension, max_iters=20, time_budget=1.0)
        else:
            # Last resort: straight line
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
