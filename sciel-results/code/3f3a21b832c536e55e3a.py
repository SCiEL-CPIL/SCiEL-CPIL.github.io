"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _rotate_around(p, center, angle):
    """Rotate point p around center by angle (90, 180, 270) degrees."""
    x, y = p
    cx, cy = center
    rx, ry = x - cx, y - cy
    if angle == 90:
        return (cx - ry, cy + rx)
    elif angle == 180:
        return (cx - rx, cy - ry)
    elif angle == 270:
        return (cx + ry, cy - rx)
    return p


def _energy(sequence, chain):
    """Calculate the energy of the conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _init_saw(n):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Try random growth with backtracking
    # To be efficient, we use a simple random walk that backtracks when stuck
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        while True:
            last = chain[-1]
            neighbors = _neighbours(last)
            random.shuffle(neighbors)
            moved = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    moved = True
                    break
            if moved:
                break
            # If no neighbor, backtrack
            # This simple backtracking might be slow for long chains if not careful
            # But for n<=100, it's usually fine.
            # However, to avoid infinite loops or very slow backtracking, 
            # we can use a simpler initialization: a straight line or spiral.
            # Let's use a spiral/serpentine pattern as it's always valid and compact.
            pass
            
    # Actually, random growth with backtracking can get stuck in a "dead end" 
    # and take a long time to backtrack. 
    # A safer initialization is a "serpentine" or "spiral" pattern.
    # Let's implement a simple serpentine pattern.
    chain = []
    for i in range(n):
        # Zigzag pattern
        if i % 2 == 0:
            row = i // 2
            col = row
        else:
            row = i // 2
            col = row + 1
        # This doesn't quite work for a proper lattice walk.
        
    # Let's just use a straight line and let SA fix it? No, SA needs a good start.
    # Let's use a simple "spiral" algorithm.
    # Or better: Random growth with a limit on backtracking steps.
    
    # Robust Initialization: Random Walk with Backtracking
    chain = [(0, 0)]
    occupied = {(0, 0)}
    max_backtracks = 1000
    backtracks = 0
    
    for i in range(1, n):
        while True:
            if backtracks > max_backtracks:
                # Fallback to a simple valid walk (e.g., straight line)
                # This shouldn't happen often
                return [(i, 0) for i in range(n)]
            
            last = chain[-1]
            neighbors = [nb for nb in _neighbours(last) if nb not in occupied]
            if neighbors:
                nb = random.choice(neighbors)
                chain.append(nb)
                occupied.add(nb)
                break
            else:
                # Backtrack
                if len(chain) <= 1:
                    # Reset
                    chain = [(0, 0)]
                    occupied = {(0, 0)}
                    break
                last = chain.pop()
                occupied.remove(last)
                backtracks += 1
                
    return chain


def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Pivot Moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Time budget: ~10 seconds per sequence (100s total for 10 sequences, leaving 20s buffer)
    # But the contract says 120s for ALL sequences. So 12s each.
    # Let's use 10s to be safe.
    start_time = time.time()
    time_limit = 10.0
    
    # Initialize
    chain = _init_saw(n)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = random.Random(42)
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Try a pivot move
        # Choose pivot index k (1 to n-2)
        k = rng.randint(1, n - 2)
        # Choose rotation angle
        angle = rng.choice([90, 180, 270])
        
        pivot_point = chain[k - 1]
        fixed_set = set(chain[:k - 1])
        
        # Rotate the segment chain[k:]
        moving_segment = chain[k:]
        new_moving_segment = [_rotate_around(p, pivot_point, angle) for p in moving_segment]
        
        # Check collision with fixed part
        valid = True
        for p in new_moving_segment:
            if p in fixed_set:
                valid = False
                break
        
        if valid:
            new_chain = chain[:k] + new_moving_segment
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            # Metropolis criterion
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
        
        T *= cooling_rate
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
