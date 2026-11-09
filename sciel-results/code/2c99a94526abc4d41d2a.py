import time
import random
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 240.0
_DEADLINE_MARGIN = 10.0

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng):
    """
    Greedy growth to create an initial self-avoiding walk.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that close loops or are close to other H's
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Add small random noise to avoid deterministic traps
            score = gain * 10.0 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_optimize(sequence, chain, rng, deadline):
    """
    Local optimization: try single-residue moves and small segment regrowths.
    Returns improved chain and energy.
    """
    N = len(chain)
    if N == 0:
        return chain, 0
    
    best_chain = chain[:]
    best_energy = _energy(sequence, chain)
    where = {p: i for i, p in enumerate(best_chain)}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try a random local move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move: pick a position, try moving it to a neighbor
            if N <= 2:
                break
            i = rng.randint(1, N - 2)
            old_pos = best_chain[i]
            for new_pos in _neighbours(old_pos):
                # Check self-avoidance
                if new_pos in where and where[new_pos] != i:
                    continue
                # Check adjacency constraints
                if i > 0:
                    if abs(new_pos[0] - best_chain[i-1][0]) + abs(new_pos[1] - best_chain[i-1][1]) + abs(new_pos[2] - best_chain[i-1][2]) != 1:
                        continue
                if i < N - 1:
                    if abs(new_pos[0] - best_chain[i+1][0]) + abs(new_pos[1] - best_chain[i+1][1]) + abs(new_pos[2] - best_chain[i+1][2]) != 1:
                        continue
                
                # Apply move
                new_chain = best_chain[:]
                new_chain[i] = new_pos
                new_energy = _energy(sequence, new_chain)
                
                delta = new_energy - best_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    best_chain = new_chain
                    best_energy = new_energy
                    no_improve = 0
                    break
                else:
                    no_improve += 1
        else:
            # Small segment regrowth: pick 3-8 consecutive residues
            seg_len = rng.randint(3, min(8, N - 2))
            if seg_len >= N:
                seg_len = max(3, N - 2)
            start_i = rng.randint(1, N - seg_len - 1)
            
            head = best_chain[:start_i]
            tail = best_chain[start_i + seg_len:]
            
            # Regrow the segment
            occupied = set(head) | set(tail)
            current_pos = head[-1] if head else (0, 0, 0)
            
            new_seg = []
            valid = True
            for k in range(seg_len):
                options = []
                for site in _neighbours(current_pos):
                    if site in occupied:
                        continue
                    score = rng.random()
                    if sequence[start_i + k] == 'H':
                        # Bonus for being near H residues in head/tail
                        for q in _neighbours(site):
                            if q in occupied:
                                # Find index in full chain
                                idx = None
                                for idx_h, pos_h in enumerate(head):
                                    if pos_h == q:
                                        idx = idx_h
                                        break
                                if idx is None:
                                    for idx_t, pos_t in enumerate(tail):
                                        if pos_t == q:
                                            idx = start_i + seg_len + idx_t
                                            break
                                if idx is not None and sequence[idx] == 'H':
                                    score += 5.0
                    options.append((score, site))
                
                if not options:
                    valid = False
                    break
                    
                best_score, best_site = max(options)
                new_seg.append(best_site)
                occupied.add(best_site)
                current_pos = best_site
            
            if valid:
                new_chain = head + new_seg + tail
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - best_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    best_chain = new_chain
                    best_energy = new_energy
                    no_improve = 0
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain, best_energy

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    start_time = time.time()
    # Compute deadline based on global budget
    elapsed = time.time() - _START_TIME
    remaining = _TOTAL_BUDGET - elapsed - _DEADLINE_MARGIN
    if remaining < 1.0:
        remaining = 1.0
    
    # Allocate time: 30% for initialization, 70% for SA + local opt
    init_time = remaining * 0.3
    sa_time = remaining * 0.7
    
    rng = random.Random(42)
    
    # Precompute H positions for faster energy
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Initialize with best of several greedy chains
    best_chain = None
    best_energy = 0
    
    init_deadline = start_time + init_time
    for _ in range(20):
        if time.time() > init_deadline:
            break
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search with Simulated Annealing
    T = 1.5
    T_min = 0.05
    cooling_rate = 0.9998
    sa_deadline = start_time + sa_time * 0.7  # Leave some time for local opt
    
    no_improve = 0
    max_no_improve = 500
    restart_count = 0
    max_restarts = 5
    
    while time.time() < sa_deadline:
        # Pick a cut point with bias toward middle
        # Use a distribution that favors cuts in the middle third
        r = rng.random()
        if r < 0.3:
            i = rng.randint(1, max(1, N // 4))
        elif r < 0.7:
            i = rng.randint(max(1, N // 4), max(2, 3 * N // 4))
        else:
            i = rng.randint(max(1, 3 * N // 4), N - 2)
        
        head = current_chain[:i]
        
        # Prepare data structures for regrowth
        head_map = {p: idx for idx, p in enumerate(head)}
        occupied = set(head_map.keys())
        
        new_tail = []
        tail_map = {}
        current_pos = head[-1]
        
        valid = True
        
        for k in range(i, N):
            options = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[k] == "H":
                    for q in _neighbours(site):
                        # Check if q is in head
                        if q in head_map:
                            idx_h = head_map[q]
                            if sequence[idx_h] == "H":
                                gain += 1
                        # Check if q is in new_tail
                        elif q in tail_map:
                            idx_t = tail_map[q]
                            if sequence[idx_t] == "H":
                                gain += 1
                                
                # Score: maximize gain, add noise
                score = gain * 8.0 + rng.random() * 2.0
                options.append((score, site))
                
            if not options:
                valid = False
                break
                
            best_score, best_site = max(options)
            new_tail.append(best_site)
            tail_map[best_site] = k
            occupied.add(best_site)
            current_pos = best_site
            
        if not valid:
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart with a new greedy chain
                restart_count += 1
                if restart_count > max_restarts:
                    break
                chain = _greedy_grow(sequence, rng)
                if chain:
                    e = _energy(sequence, chain)
                    if e < current_energy:
                        current_chain = chain
                        current_energy = e
                    no_improve = 0
                    T = 1.5
            continue
            
        new_chain = head + new_tail
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                no_improve = 0
            else:
                no_improve += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            no_improve += 1
            
        # Reset no_improve counter periodically
        if no_improve > max_no_improve:
            no_improve = 0
            T = max(T, 0.5)
    
    # Final local optimization pass
    local_deadline = start_time + sa_time
    best_chain, best_energy = _local_optimize(sequence, best_chain, rng, local_deadline)
    
    return [list(p) for p in best_chain]
