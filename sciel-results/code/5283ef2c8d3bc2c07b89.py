"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

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

def _generate_greedy_chain(sequence, rng, temperature=0.0):
    """
    Generate a conformation using a greedy growth strategy.
    This creates a compact, contact-rich initial state, unlike random SAWs.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Gain calculation: prefer placing H next to existing H
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic: prefer sites with more empty neighbors to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: gain is dominant, free space is secondary, noise for diversity
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            
            options.append((score, site))
        
        if not options:
            # Stuck: return straight line fallback (should be rare for small n)
            return [(i, 0) for i in range(n)]
        
        # Pick best
        best_score, best_site = max(options, key=lambda x: x[0])
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _regrow_suffix(chain, sequence, start_idx, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from start_idx.
    Keeps chain[0:start_idx] fixed.
    Returns new suffix or None if stuck.
    """
    prefix = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    new_suffix = []
    last = prefix[-1] if prefix else (0, 0)
    
    for i in range(start_idx, len(chain)):
        res = sequence[i]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Gain calculation
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        new_suffix.append(best_site)
        last = best_site
        
    return new_suffix

def _local_improve(sequence, chain, rng, max_iters=5000):
    """
    Local improvement using Regrow moves with Simulated Annealing.
    """
    n = len(chain)
    if n < 5:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 2.0
    cool_rate = 0.995
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        if temp < 0.05:
            break
            
        # Choose a split point for regrow
        # Bias towards shorter suffixes for fine-tuning, longer for global moves
        # Uniform distribution is a good start
        min_split = 3
        max_split = max(min_split, n - 2)
        if max_split <= min_split:
            temp *= cool_rate
            continue
            
        split = rng.randint(min_split, max_split)
        
        new_suffix = _regrow_suffix(chain, sequence, split, rng, temperature=temp)
        if new_suffix is None:
            temp *= cool_rate
            continue
        
        new_chain = chain[:split] + new_suffix
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                # Restart cooling slightly
                temp = min(temp * 1.05, 3.0)
            else:
                temp *= cool_rate
        else:
            temp *= cool_rate

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using a strategy that combines:
    1. Greedy initial growth (to create compact structures).
    2. Multiple independent SA runs starting from these greedy states.
    3. Keeping the best result.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # We have ~170s.
    # Let's allocate time to multiple independent SA runs.
    
    # For short sequences, we can afford more runs.
    # For long sequences, fewer runs but longer.
    # Let's aim for ~10-20 runs depending on n.
    
    num_runs = 15
    time_per_run = 10.0 # 150s total
    
    for run_idx in range(num_runs):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate a greedy initial conformation with some noise
        # Temperature increases with run index to explore more diverse greedy paths
        temp = 0.5 + run_idx * 0.2
        init_chain = _generate_greedy_chain(sequence, rng, temperature=temp)
        
        # Run SA on this initial chain
        sa_end = time.time() + time_per_run
        # Adjust max_iters based on time
        max_iters = 20000 # High, will be cut by time check
        
        sa_chain, sa_energy = _local_improve(sequence, init_chain, rng, max_iters=max_iters)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Fallback if no valid chain found (should not happen)
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
