import random
import time
import math

# Global deadline guard
_DEADLINE = 230.0
_start_time = None

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

def _neighbours(point):
    """Yield 6 neighbours in 3D lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, temp=0.5):
    """
    Generate a self-avoiding walk using greedy growth with randomness.
    Returns None if stuck.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Score: prioritize H-H contacts, then free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Boltzmann-like selection weight
            score = gain * 5.0 + free * 0.1
            # Add noise
            noise = rng.gauss(0, temp)
            options.append((score + noise, site))
        
        if not options:
            return None
        
        # Pick best
        best_score, site = max(options)
        occupied[site] = i
        chain.append(site)
    
    return chain

def _sa_optimize(sequence, chain, dimension, rng, time_budget):
    """
    Simulated Annealing optimization on a given chain.
    Returns the best chain and energy found during the run.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature parameters
    T = 1.0
    alpha = 0.9995  # Cooling rate
    min_T = 0.01
    
    start_time = time.time()
    
    # Precompute H indices for faster checks? 
    # Actually, _energy is O(N * 6). For N=100, that's 600 ops. Fast enough.
    
    while time.time() - start_time < time_budget:
        if _get_remaining_time() < 1.0:
            break
            
        # Pick a move
        # 80% Reversal, 20% Rotation
        if rng.random() < 0.8:
            # Reversal move
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            if b - a < 2: # Too short, skip
                continue
                
            # Apply reversal
            new_chain = current_chain[:a] + current_chain[a:b+1][::-1] + current_chain[b+1:]
            
            # Validate
            valid = True
            seen = set()
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
                
            if valid:
                # Check connectivity at junctions
                if a > 0:
                    d1 = sum(abs(new_chain[a-1][d] - new_chain[a][d]) for d in range(dimension))
                    if d1 != 1:
                        valid = False
                if valid and b < n - 1:
                    d2 = sum(abs(new_chain[b][d] - new_chain[b+1][d]) for d in range(dimension))
                    if d2 != 1:
                        valid = False
            
            if not valid:
                continue
                
            # Calculate energy delta
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
        else:
            # Rotation move: Move a single residue i to an adjacent free site
            # This is harder to validate quickly without full check
            # Let's try moving a segment of length 1 (single residue)
            # Actually, single residue move requires the new position to be adjacent to both neighbors.
            # This is rare. Let's stick to reversals for simplicity and speed, 
            # or implement a "crank" move: move residue i to a site adjacent to i-1 and i+1.
            
            # Try a crank move on a random residue
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            # Neighbors of old_pos
            for new_pos in _neighbours(old_pos):
                if new_pos == current_chain[i-1] or new_pos == current_chain[i+1]:
                    continue
                
                # Check if new_pos is free
                # We need a fast way to check if new_pos is in current_chain.
                # Since we don't have a dict for current_chain in SA loop (updating it is expensive),
                # we can use a set. But building a set every move is O(N).
                # For N=100, O(N) is 100 ops. _energy is O(N*6)=600 ops.
                # So set check is cheaper.
                
                # To avoid O(N) set construction every time, we can maintain a dict of pos->index.
                # But maintaining it through reversals is complex.
                # Let's just do a linear search for small N or use a set if N is large.
                # For N<=136, linear search is fine.
                
                is_free = True
                for p in current_chain:
                    if p == new_pos:
                        is_free = False
                        break
                if not is_free:
                    continue
                
                # Check connectivity
                d1 = sum(abs(current_chain[i-1][d] - new_pos[d]) for d in range(dimension))
                d2 = sum(abs(new_pos[d] - current_chain[i+1][d]) for d in range(dimension))
                if d1 != 1 or d2 != 1:
                    continue
                
                # Apply move
                new_chain = current_chain[:i] + [new_pos] + current_chain[i+1:]
                
                # Full validation (self-avoidance)
                valid = True
                seen = set()
                for p in new_chain:
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                
                if not valid:
                    continue
                    
                new_energy = _energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                    current_chain = new_chain
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break # Only one crank per iteration
        
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T
            
    return best_chain, best_energy

def fold(sequence, dimension):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0 # 0 is no contacts, energy is negative, so we want to minimize (more negative)
    
    # Strategy: Multi-start SA
    # Allocate time per run based on total remaining time
    # We will run runs of ~2-5 seconds each
    
    while _get_remaining_time() > 5.0:
        # Generate initial chain
        # Use different temperatures for diversity
        temp = rng.uniform(0.1, 2.0)
        chain = _greedy_grow(sequence, dimension, rng, temp)
        
        if chain is None:
            # If greedy fails, try a random walk
            chain = _random_walk(sequence, dimension, rng)
            if chain is None:
                continue
        
        # Run SA
        # Time budget for this SA run
        # Shorter chains can afford more runs, longer chains need longer runs
        if n <= 50:
            time_budget = 1.0
        elif n <= 100:
            time_budget = 2.0
        else:
            time_budget = 3.0
            
        # Ensure we don't overshoot deadline
        if _get_remaining_time() < time_budget + 1.0:
            time_budget = max(0.5, _get_remaining_time() - 1.0)
            
        sa_chain, sa_energy = _sa_optimize(sequence, chain, dimension, rng, time_budget)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
            
    # Fallback
    if best_chain is None:
        # Straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

def _random_walk(sequence, dimension, rng):
    """
    Generate a self-avoiding walk by random choices.
    Returns None if stuck.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
                
        if not options:
            return None
            
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
        
    return chain
