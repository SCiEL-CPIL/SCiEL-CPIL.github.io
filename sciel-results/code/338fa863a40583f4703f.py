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

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n.
    Uses a simple random walk with backtracking if stuck.
    Returns list of (x, y) tuples.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Simple recursive generation with backtracking
    # For n up to 100, this is fast enough
    def gen(length, current, occupied):
        if length == 0:
            return [current]
        
        # Shuffle directions to avoid bias
        dirs = list(range(4))
        rng.shuffle(dirs)
        
        for d in dirs:
            nx, ny = current[0] + DIRS[d][0], current[1] + DIRS[d][1]
            new_pos = (nx, ny)
            if new_pos not in occupied:
                occupied.add(new_pos)
                suffix = gen(length - 1, new_pos, occupied)
                if suffix:
                    return [current] + suffix
                occupied.remove(new_pos)
        
        return None

    # Start at (0,0)
    occupied = {(0, 0)}
    result = gen(n - 1, (0, 0), occupied)
    if result is None:
        # Fallback: straight line (should rarely happen for n<100)
        return [(i, 0) for i in range(n)]
    return result

def _regrow_suffix(chain, sequence, start_idx, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from start_idx.
    Keeps chain[0:start_idx] fixed.
    Returns new suffix or None if stuck.
    """
    prefix = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    new_suffix = []
    last = prefix[-1]
    
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

def _local_improve_pivot(sequence, chain, rng, max_iters=5000):
    """
    Local improvement using Regrow moves.
    This is a stochastic local search.
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
            
        # Choose move type
        # 50% Regrow, 50% Pivot (if implemented)
        # For now, let's use a mix of Regrow and "Slide" (local shift)
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Regrow move
            # Pick a random split point
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
                    # Restart cooling slightly?
                    temp = min(temp * 1.05, 3.0)
                else:
                    temp *= cool_rate
            else:
                temp *= cool_rate
                
        else:
            # Local Slide Move: Try to move a single residue to an adjacent empty spot
            # This helps fine-tune
            i = rng.randint(1, n - 2)
            p = chain[i]
            moved = False
            for np in _neighbours(p):
                if np in {chain[i-1], chain[i+1]}:
                    continue
                if any(np == q for q in chain):
                    continue
                
                # Check connectivity
                # Moving p to np changes bonds (i-1, i) and (i, i+1)
                # We need to check if np is adjacent to chain[i-1] and chain[i+1]?
                # No, that's a "kink" flip or specific move.
                # A general slide is hard to maintain self-avoidance easily without checking all.
                # Let's stick to Regrow as the primary global move.
                pass
            
            # If we don't implement a good local move, just do another Regrow
            # Let's do a short Regrow (last 5-10 residues) for fine tuning
            split = max(3, n - rng.randint(5, 10))
            new_suffix = _regrow_suffix(chain, sequence, split, rng, temperature=temp * 0.5)
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
                    temp = min(temp * 1.05, 3.0)
                else:
                    temp *= cool_rate
            else:
                temp *= cool_rate

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Global Stochastic Local Search.
    1. Generate diverse initial random SAWs.
    2. Optimize each using Regrow-SA (Pivot-like global moves).
    3. Keep the best.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # We have ~170s.
    # Let's allocate time to multiple independent SA runs starting from different random SAWs.
    
    num_runs = 10
    time_per_run = 15.0 # 150s total
    
    for run_idx in range(num_runs):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate a random initial conformation
        # For short sequences, random SAWs are easy.
        # For long sequences, we might need to use greedy to get a valid start,
        # but the strategy is to avoid greedy bias.
        # Let's try random SAW first. If it fails (returns None or straight line), use greedy.
        
        init_chain = _generate_random_saw(n, rng)
        
        # Check if init_chain is valid (it should be by construction)
        # If it's the fallback straight line, it's a valid SAW but high energy.
        
        # Run SA on this initial chain
        sa_end = time.time() + time_per_run
        # Adjust max_iters based on time
        max_iters = 20000 # High, will be cut by time check
        
        sa_chain, sa_energy = _local_improve_pivot(sequence, init_chain, rng, max_iters=max_iters)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
            # If we found a very good energy, we can reduce time for subsequent runs?
            # Or just keep going to try to beat it.
            
    # Fallback if no valid chain found (should not happen)
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
