import time
import random
import math

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate H-H contact energy change when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    x, y, z = q
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _compute_potential(occupied, sequence, i):
    """
    Compute a potential field score for each valid move from the last position.
    The potential represents local hydrophobic density and free space.
    Returns dict: position -> score
    """
    if i < 1:
        return {}
    
    last_pos = None
    # We'll compute this in the caller where we have the chain
    
    return {}

def _greedy_complete(sequence, chain, occupied, start_idx, deadline):
    """Complete a partial chain greedily, preferring moves that create H-H contacts."""
    n = len(sequence)
    
    for j in range(start_idx, n):
        if time.time() > deadline:
            break
            
        last_pos = chain[-1]
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in occupied:
                valid_mvs.append(q)
        
        if not valid_mvs:
            break
        
        best_q = None
        best_score = -1e9
        for q in valid_mvs:
            score = 0.0
            if sequence[j] == 'H':
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r in occupied:
                        idx = occupied[r]
                        if idx != j - 1 and sequence[idx] == 'H':
                            score += 10.0
            
            # Free space bonus to avoid trapping
            free = 0
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx, q[1] + dy, q[2] + dz)
                if r not in occupied:
                    free += 1
            score += free * 1.0
            
            if score > best_score:
                best_score = score
                best_q = q
        
        if best_q is None:
            break
            
        chain.append(best_q)
        occupied[best_q] = j
    
    return chain, occupied, len(chain) == n

def _monte_carlo_fold(sequence, n, deadline, rng):
    """
    Monte Carlo path sampling guided by hydrophobic potential.
    Grows the chain step by step, using a softmax over move scores
    that balance H-H contact formation and free space.
    """
    init_chain = [(0, 0, 0)]
    init_occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last_pos = init_chain[-1]
        
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in init_occupied:
                valid_mvs.append(q)
        
        if not valid_mvs:
            return None
        
        # Compute scores for each valid move
        scores = []
        for q in valid_mvs:
            score = 0.0
            
            # Hydrophobic contact potential
            if sequence[i] == 'H':
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r in init_occupied:
                        j = init_occupied[r]
                        if j != i - 1 and sequence[j] == 'H':
                            score += 5.0
            
            # Free space to avoid trapping
            free = 0
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx, q[1] + dy, q[2] + dz)
                if r not in init_occupied:
                    free += 1
            score += free * 0.5
            
            # Slight preference for compactness (closer to origin)
            dist = abs(q[0]) + abs(q[1]) + abs(q[2])
            score -= dist * 0.01
            
            scores.append(score)
        
        # Softmax sampling with temperature
        temperature = max(0.1, 1.0 - i / (n * 1.5))
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        
        # Sample a move
        r = rng.random()
        cumsum = 0.0
        chosen_idx = 0
        for idx, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                chosen_idx = idx
                break
        
        new_pos = valid_mvs[chosen_idx]
        init_chain.append(new_pos)
        init_occupied[new_pos] = i
    
    return init_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    # Use most of the 240s budget, with safety margin
    deadline = start_time + 220.0
    
    best_chain = None
    best_energy = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple Monte Carlo trials
    max_trials = 5000
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        chain = _monte_carlo_fold(sequence, n, deadline, rng)
        
        if chain is None or len(chain) != n:
            continue
        
        # Calculate energy
        occupied = {}
        for idx, pos in enumerate(chain):
            occupied[pos] = idx
        
        energy = 0
        for i in range(n):
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r in occupied:
                    j = occupied[r]
                    if j > i and j != i + 1:
                        if sequence[j] == 'H':
                            energy -= 1
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain[:]
        
        # If we found a very good solution, we can stop early
        # (heuristic: if energy is very negative relative to chain length)
        if n > 10 and best_energy <= -n * 0.4:
            break
    
    # If no complete chain found, try greedy completion from a random start
    if best_chain is None:
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Start with a short random walk
            chain = [(0, 0, 0)]
            occupied = {(0, 0, 0): 0}
            rng2 = random.Random(42 + attempt)
            
            for i in range(1, min(10, n)):
                last_pos = chain[-1]
                valid_mvs = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                    if q not in occupied:
                        valid_mvs.append(q)
                
                if not valid_mvs:
                    break
                
                q = rng2.choice(valid_mvs)
                chain.append(q)
                occupied[q] = i
            
            if len(chain) == min(10, n):
                chain, occupied, complete = _greedy_complete(sequence, chain, occupied, min(10, n), deadline)
                if complete:
                    # Calculate energy
                    energy = 0
                    for i in range(n):
                        if sequence[i] != 'H':
                            continue
                        x, y, z = chain[i]
                        for dx, dy, dz in _NEIGHBOR_OFFSETS:
                            r = (x + dx, y + dy, z + dz)
                            if r in occupied:
                                j = occupied[r]
                                if j > i and j != i + 1:
                                    if sequence[j] == 'H':
                                        energy -= 1
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = chain[:]
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
