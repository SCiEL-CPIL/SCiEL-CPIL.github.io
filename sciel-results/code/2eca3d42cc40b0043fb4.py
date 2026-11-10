"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

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


def _energy_delta_for_move(sequence, chain, new_chain, old_energy):
    """
    Compute the energy of new_chain. 
    For efficiency in local search, we could do incremental updates, 
    but for simplicity and robustness against bugs, we recompute.
    This is O(N) per move, which is acceptable for N<=100 and ~100k moves.
    """
    return _compute_energy(sequence, new_chain)


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain in contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            elif sequence[i] == 'P':
                # P residues don't contribute to energy, but might block? No, just no gain.
                # Actually, P-H or P-P contacts don't count. Only H-H.
                pass
            
            # Calculate free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize gain, then free space, then random noise
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None # Trapped
        
        # Select best option
        options.sort(key=lambda x: x[0], reverse=True)
        # Pick from top few to allow some exploration
        k = min(3, len(options))
        _, best_site = rng.choice(options[:k])
        
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain


def _greedy_grow_from_fixed(sequence, dimension, fixed_prefix, rng, temperature=0.0):
    """
    Grow the rest of the chain from a fixed prefix.
    """
    chain = list(fixed_prefix)
    occupied = {p: i for i, p in enumerate(chain)}
    n = len(sequence)
    start_idx = len(chain)
    
    if start_idx >= n:
        return chain

    for i in range(start_idx, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        k = min(3, len(options))
        _, best_site = rng.choice(options[:k])
        
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain


def _local_improve(sequence, chain, rng, max_iters=20000, time_limit=None):
    """
    Local search using pivot moves and regrow moves.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 2.0
    no_improve = 0
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if time_limit and time.time() > time_limit:
            break
        if temp < 0.05:
            break
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot move
            # Pick pivot index i and segment end j
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Ensure segment length is reasonable for pivot
            if j - i > n // 2:
                continue
                
            prefix = chain[:i+1]
            suffix = chain[j+1:]
            segment = chain[i+1:j+1]
            
            center = chain[i]
            
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
            
            # Check self-avoidance and connectivity
            if len(new_chain) != len(set(new_chain)):
                continue
            for k in range(len(new_chain) - 1):
                if abs(new_chain[k][0] - new_chain[k+1][0]) + abs(new_chain[k][1] - new_chain[k+1][1]) != 1:
                    break
            else:
                # Valid move
                new_energy = _compute_energy(sequence, new_chain)
                delta = new_energy - current_energy
                
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
                continue
        
        else:
            # Regrow suffix
            split = rng.randint(3, n - 2)
            new_chain = _greedy_grow_from_fixed(sequence, dimension, chain[:split], rng, temperature=temp)
            
            if new_chain is None:
                no_improve += 1
                temp *= 0.995
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
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


def _regrow_sa(sequence, chain, rng, time_limit):
    """
    Simulated Annealing focused on regrowing suffixes.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 5.0
    no_improve = 0
    
    while time.time() < time_limit:
        split = rng.randint(5, n - 2)
        new_chain = _greedy_grow_from_fixed(sequence, 2, chain[:split], rng, temperature=temp)
        
        if new_chain is None:
            no_improve += 1
            temp *= 0.98
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
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
            
        temp *= 0.98
        
        if no_improve > 200:
            break
            
    return best_chain, best_energy


def _min_conflicts_search(sequence, dimension, rng, max_iters=50000):
    """
    Min-conflicts style search: Start with a random valid SAW, 
    then move residues to reduce conflicts (overlap/energy).
    Since we need a valid SAW, we can't just place residues independently.
    Instead, this is a heuristic that uses "kick" moves: 
    Pick a segment, remove it, and regrow it to minimize local energy.
    """
    n = len(sequence)
    # Generate initial valid chain
    chain = None
    for _ in range(100):
        chain = _greedy_grow(sequence, dimension, rng, temperature=1.0)
        if chain:
            break
    
    if not chain:
        # Fallback straight line
        chain = [(i, 0) for i in range(n)]
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Pick a segment to "kick"
        i = rng.randint(0, n - 5)
        j = rng.randint(i + 4, n - 1)
        
        prefix = chain[:i+1]
        suffix = chain[j+1:]
        
        # Regrow the segment i+1..j
        # We need to regrow from prefix, but also ensure it connects to suffix?
        # No, standard regrow just extends from prefix. The suffix is disconnected.
        # This is a "cut and regrow" move.
        # To make it a valid move, the new segment must not intersect prefix or suffix.
        
        # Let's use a simpler move: Regrow the whole suffix from a random point.
        # Or use the _regrow_sa logic.
        # For min-conflicts, let's just do random restarts with greedy grow.
        pass
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a multi-phase strategy:
    1. Generate diverse initial solutions via greedy growth with varying temperatures.
    2. Refine using local search (pivot + regrow).
    3. Refine further using simulated annealing on regrow moves.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    # Adaptive time allocation
    if n < 30:
        init_time = 20.0
        sa_time = 100.0
        local_time = 40.0
    elif n < 60:
        init_time = 30.0
        sa_time = 80.0
        local_time = 50.0
    else:
        init_time = 40.0
        sa_time = 60.0
        local_time = 50.0
        
    best_chain = None
    best_energy = 0 # Minimize (more negative is better)
    
    # Phase 1: Generate initial solutions
    phase1_end = time.time() + init_time
    attempts = 0
    while time.time() < phase1_end and attempts < 1000:
        attempts += 1
        # Vary temperature to get diverse solutions
        temp = rng.uniform(0.5, 2.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
            
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
    # Fallback if greedy growth failed
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    # Phase 2: Local Search Improvement
    if best_chain is not None and time.time() < _DEADLINE - 20:
        phase2_end = time.time() + local_time
        if phase2_end > _DEADLINE - 20:
            phase2_end = _DEADLINE - 20
            
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_iters=50000, time_limit=phase2_end)
            
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
            
    # Phase 3: Simulated Annealing on Regrow
    if best_chain is not None and time.time() < _DEADLINE - 10:
        phase3_end = time.time() + sa_time
        if phase3_end > _DEADLINE - 10:
            phase3_end = _DEADLINE - 10
            
        improved_chain, improved_energy = _regrow_sa(
            sequence, best_chain, rng, time_limit=phase3_end)
            
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
