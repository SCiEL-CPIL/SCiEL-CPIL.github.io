"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n.
    Returns None if it fails to generate one (traps itself).
    """
    if n <= 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                # Simple heuristic: prefer sites with more free space to reduce trapping
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                options.append((free, site))
        
        if not options:
            return None
        
        # Randomly select, but bias towards free space slightly
        # For pure random, we would just pick random. 
        # Let's pick random among valid to ensure diversity, 
        # but maybe sort by free space and pick from top half? 
        # Actually, simple random walk generation often fails for n>50.
        # A better way: use a greedy random walk that avoids immediate traps.
        
        # Let's just pick random valid option for now to keep it simple and fast
        # If it fails, we retry.
        # To improve success rate, we can use a "greedy random" approach:
        # score = free + random
        scored_options = [(free + rng.random() * 2.0, site) for free, site in options]
        scored_options.sort(key=lambda x: x[0], reverse=True)
        
        # Pick from the top few to balance exploration and survival
        k = min(3, len(scored_options))
        _, best_site = rng.choice(scored_options[:k])
        
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain


def _apply_pivot(chain, i, j, rng):
    """
    Apply a pivot move: rotate the sub-chain from i to j by 90, 180, or 270 degrees.
    Returns the new chain if valid, otherwise None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    # We pivot around residue i. The sub-chain is chain[i+1:j+1].
    # Actually, standard pivot is: fix chain[0:i], rotate chain[i:j+1] around chain[i].
    # Let's define pivot at index i. The segment to rotate is chain[i+1 ... j].
    # Wait, usually pivot is defined by two indices i < j. The segment chain[i+1...j] is rotated around chain[i].
    
    prefix = chain[:i+1]
    suffix = chain[j+1:]
    segment = chain[i+1:j+1]
    
    if not segment:
        return chain

    center = chain[i]
    
    # Rotation matrices for 90, 180, 270 degrees
    # (x, y) -> (y, -x) for 90 CW? 
    # Let's use:
    # 90 CW: (x, y) -> (y, -x)
    # 180: (x, y) -> (-x, -y)
    # 270 CW: (x, y) -> (-y, x)
    
    # We need to rotate the relative coordinates with respect to center.
    
    rotations = [
        lambda x, y: (y, -x),      # 90 CW
        lambda x, y: (-x, -y),     # 180
        lambda x, y: (-y, x)       # 270 CW
    ]
    
    rot_func = rng.choice(rotations)
    
    new_segment = []
    for p in segment:
        rx = p[0] - center[0]
        ry = p[1] - center[1]
        nx, ny = rot_func(rx, ry)
        new_segment.append((center[0] + nx, center[1] + ny))
    
    new_chain = prefix + new_segment + suffix
    
    # Check self-avoidance
    if len(new_chain) != len(set(new_chain)):
        return None
    
    # Check connectivity (unit steps)
    for k in range(len(new_chain) - 1):
        if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
            return None
            
    return new_chain


def _regrow_suffix(chain, split, sequence, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from index `split`.
    Returns the new full chain if successful, else None.
    """
    n = len(chain)
    if split < 0 or split >= n:
        return None
        
    prefix = chain[:split]
    suffix_seq = sequence[split:]
    
    if not suffix_seq:
        return chain
        
    occupied = {p: i for i, p in enumerate(prefix)}
    current_pos = prefix[-1]
    new_suffix = [current_pos] # Start with the last prefix pos to build from
    
    # We need to generate the rest
    full_new_chain = list(prefix)
    
    for i, res in enumerate(suffix_seq):
        global_idx = split + i
        last = full_new_chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
                
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < global_idx - 1 and sequence[j] == 'H':
                        gain += 1
                        
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
            
        if not options:
            return None
            
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = global_idx
        full_new_chain.append(best_site)
        
    return full_new_chain


def _local_improve_pivot(sequence, chain, rng, max_iters=5000):
    """
    Improve conformation using pivot moves and regrow moves.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 2.0
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break
            
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot move
            # Pick i and j
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_pivot(chain, i, j, rng)
        else:
            # Regrow suffix
            split = rng.randint(3, n - 2)
            new_chain = _regrow_suffix(chain, split, sequence, rng, temperature=temp)
            
        if new_chain is None:
            no_improve += 1
            temp *= 0.995
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= 0.995
        
        if no_improve > 500:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using Global Stochastic Local Search (Pivot/Regrow).
    """
    rng = random.Random(42)
    n = len(sequence)
    
    # Adaptive time allocation
    if n < 30:
        init_time = 20.0
        sa_time = 100.0
    elif n < 60:
        init_time = 30.0
        sa_time = 80.0
    else:
        init_time = 40.0
        sa_time = 60.0
        
    best_chain = None
    best_energy = 0 # Minimize (more negative is better)
    
    # Phase 1: Generate initial solutions
    phase1_end = time.time() + init_time
    attempts = 0
    while time.time() < phase1_end and attempts < 1000:
        attempts += 1
        chain = _generate_random_saw(n, rng)
        if chain is None:
            continue
            
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
    # Fallback if random SAW failed (should be rare for small n, but possible for large)
    if best_chain is None:
        # Simple greedy fallback
        best_chain = [(0,0)]
        occupied = {(0,0): 0}
        for i in range(1, n):
            last = best_chain[-1]
            found = False
            for site in _neighbours(last):
                if site not in occupied:
                    occupied[site] = i
                    best_chain.append(site)
                    found = True
                    break
            if not found:
                # Trapped, just extend linearly (invalid but better than crash? No, must be valid)
                # If trapped, we have to restart or return a straight line if possible?
                # For n>100, straight line is always valid.
                # Let's just return a straight line if we get stuck here, though energy will be 0.
                # But we should try to complete it.
                # If we get stuck, the chain is incomplete. We must return a valid chain of length n.
                # A straight line is always valid.
                # Let's replace the whole chain with a straight line.
                best_chain = [(i, 0) for i in range(n)]
                break
        best_energy = _compute_energy(sequence, best_chain)
        
    # Phase 2: Stochastic Local Search
    if best_chain is not None:
        phase2_end = time.time() + sa_time
        # Run multiple passes with different random seeds/temps
        for pass_idx in range(3):
            if time.time() > phase2_end:
                break
            remaining = phase2_end - time.time()
            if remaining < 10:
                break
                
            # Use a slightly different rng or just continue
            improved_chain, improved_energy = _local_improve_pivot(
                sequence, best_chain, rng, max_iters=10000)
                
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
