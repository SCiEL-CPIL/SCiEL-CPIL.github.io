"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

# Global state for time management
_start_time = None
_TOTAL_BUDGET = 240.0  # Use full budget, evaluator kills at 330s

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return max(0.0, _TOTAL_BUDGET - (time.time() - _start_time))

def _generate_initial_chain(n):
    """
    Generate a compact, self-avoiding initial chain.
    Uses a simple 'spiral' or 'snake' pattern to ensure high density and validity.
    """
    if n <= 1:
        return np.array([(0, 0, 0)] * n, dtype=np.int32)
    
    chain = np.zeros((n, 3), dtype=np.int32)
    # Simple snake pattern in x-y plane
    x, y, z = 0, 0, 0
    chain[0] = [x, y, z]
    
    # Define directions: Right, Up, Left, Down (in x, y)
    # We want to fill a 2D grid to be compact.
    # For simplicity and speed, a straight line is valid but bad energy.
    # A simple "boustrophedon" (snake) in a bounding box is better.
    
    # Let's just do a straight line first? No, that's terrible for contacts.
    # Let's do a simple 2D spiral/snake.
    # Box size: ceil(sqrt(n)) x ceil(sqrt(n))
    w = int(math.ceil(math.sqrt(n)))
    
    # Fill a w x w grid
    # This is O(N) and guarantees self-avoidance if we track visited.
    # But for N=136, a simple random walk with restarts is fast enough for initialization.
    # However, the snake is deterministic and compact.
    
    # Let's try a simple 1D line but folded?
    # Actually, the fastest valid initialization that is compact is a "helix" or "snake".
    # Let's stick to a simple random walk with rejection for init, but optimized.
    # Or better: a simple "L" shape or "U" shape?
    # Let's use a simple 2D snake.
    
    visited = set()
    visited.add((0,0,0))
    chain[0] = [0,0,0]
    
    # Directions: +x, +y, -x, -y
    dirs = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]
    
    # Simple DFS-like growth? No, just random walk with restarts is fine for init.
    # But we want it compact.
    # Let's just use a straight line for now? No, energy 0.
    # Let's use a simple "snake" in the x-y plane.
    
    # Reset chain
    chain = np.zeros((n, 3), dtype=np.int32)
    visited = set()
    
    # Start at 0,0,0
    cur = (0,0,0)
    visited.add(cur)
    chain[0] = cur
    
    # Try to grow
    # For simplicity and robustness, let's just do a straight line if N is large?
    # No, the score depends on contacts. A compact init is crucial.
    
    # Let's try a simple "spiral" in 2D.
    # This is getting complicated. Let's just use a random walk with backtracking?
    # Actually, for N=136, a simple random walk in 3D often succeeds quickly.
    # But it's not compact.
    
    # Let's use a "snake" pattern in a box of size w x w.
    w = int(math.ceil(math.sqrt(n)))
    # If n > w*w, we might need 3D or a larger box.
    # For N=136, w=12, 12*12=144. So a 12x12x1 box is enough.
    
    # Generate a list of all points in a w x w x 1 box
    points = []
    for x in range(w):
        for y in range(w):
            points.append((x, y, 0))
            
    # Shuffle? No, we need a path.
    # A simple "snake" path:
    # Row 0: 0,0 -> w-1,0
    # Row 1: w-1,1 -> 0,1
    # Row 2: 0,2 -> w-1,2
    # ...
    
    path = []
    for y in range(w):
        if y % 2 == 0:
            for x in range(w):
                path.append((x, y, 0))
        else:
            for x in range(w-1, -1, -1):
                path.append((x, y, 0))
                
    # Take first N points
    for i in range(n):
        chain[i] = path[i]
        
    return chain

def _calculate_energy(sequence, chain):
    """Calculate energy of a given chain."""
    if len(chain) < 2:
        return 0
    is_h = np.array([res == 'H' for res in sequence], dtype=bool)
    
    # Vectorized contact counting
    # For each H residue, check neighbors.
    # This is O(N) with small constant.
    energy = 0
    # Build a set of positions for O(1) lookup? 
    # For N=136, a set is fine.
    pos_to_idx = {tuple(p): i for i, p in enumerate(chain)}
    
    for i in range(len(chain)):
        if not is_h[i]:
            continue
        p = chain[i]
        # Check 6 neighbors
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nb = (p[0]+dx, p[1]+dy, p[2]+dz)
            j = pos_to_idx.get(nb)
            if j is not None and j > i + 1 and is_h[j]:
                energy += 1
    return -energy

def _monte_carlo_fold(sequence, n, time_limit):
    """
    Fast Metropolis Monte Carlo using NumPy and incremental energy updates.
    """
    if n < 2:
        return _generate_initial_chain(n)
    
    # Precompute H/P mask
    is_h = np.array([res == 'H' for res in sequence], dtype=bool)
    
    # Initialize chain
    chain = _generate_initial_chain(n)
    current_energy = _calculate_energy(sequence, chain)
    
    best_energy = current_energy
    best_chain = chain.copy()
    
    # Temperature
    T = 1.0  # Start with T=1, maybe anneal later?
    # Let's use a fixed T=1.0 for simplicity, or a simple cooling schedule.
    # Given the time budget, a fixed T=1.0 with many moves is often good.
    # Or T = 0.8?
    
    start_time = time.time()
    rng = np.random.default_rng(12345 + n)
    
    # Precompute neighbor offsets
    offsets = np.array([(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)], dtype=np.int32)
    
    # For incremental energy, we need to know which pairs are contacts.
    # We can maintain a set of contacts? 
    # Or just recalculate energy delta locally.
    # A pivot move at index i moves the tail i..n-1.
    # The only contacts that change are those involving indices in [i-1, n-1] that cross the boundary i-1/i or are within the tail (unchanged) or between tail and head (changed).
    # Actually, contacts within the tail are preserved (rigid body).
    # Contacts within the head are preserved.
    # Contacts between head and tail change.
    # The bond i-1 to i changes.
    
    # Efficient energy delta calculation:
    # 1. Remove all contacts involving indices i-1, i, or any j in tail that was a neighbor of any h in head?
    # This is complex. 
    # Given N=136, O(N) energy calc is ~136*6 = 816 ops.
    # If we do 100k moves/sec, that's 80M ops/sec. Python is slow.
    # NumPy vectorization helps, but the loop over pivots is Python.
    
    # Let's stick to a simple approach:
    # 1. Generate pivot.
    # 2. Check collision (fast with NumPy broadcasting or set).
    # 3. If valid, calculate new energy (full recalc is O(N), but maybe we can optimize).
    # 4. Metropolis.
    
    # To speed up collision check:
    # Head set: set(chain[:i])
    # Tail new positions: chain[i:] + shift
    # Check if any new tail pos in head set.
    
    # This is O(N) per move in Python.
    # For 240s, if we do 1000 moves/sec, that's 240k moves.
    # For N=136, 240k moves is decent.
    
    # Let's try to optimize the collision check using NumPy.
    # head = chain[:i]
    # tail_new = chain[i:] + shift
    # np.any(np.isin(tail_new, head, axis=0)) -> This is slow.
    # Using a set is faster in Python for small N.
    
    while time.time() - start_time < time_limit:
        # Pick pivot
        i = rng.integers(1, n - 1)
        
        # Get pivot point and previous point
        p_curr = chain[i]
        p_prev = chain[i-1]
        
        # Generate random shift
        # Pick a random axis and step
        axis = rng.integers(0, 3)
        step = rng.choice([-1, 1])
        
        # New position for p_curr
        new_p_curr = p_curr.copy()
        new_p_curr[axis] += step
        
        # If new_p_curr is same as p_prev, skip (no move)
        if np.array_equal(new_p_curr, p_prev):
            continue
            
        # If new_p_curr collides with p_prev_prev (if i>1)
        if i > 1:
            p_prev_prev = chain[i-2]
            if np.array_equal(new_p_curr, p_prev_prev):
                continue
        
        # Calculate shift
        shift = new_p_curr - p_curr
        
        # Check collisions
        # Head: chain[:i]
        # Tail: chain[i:]
        # New tail: chain[i:] + shift
        
        # Convert head to set of tuples for fast lookup
        # This is O(i) per move.
        head_set = {tuple(p) for p in chain[:i]}
        
        collision = False
        for j in range(i, n):
            new_p = chain[j] + shift
            if tuple(new_p) in head_set:
                collision = True
                break
                
        if collision:
            continue
            
        # Apply move
        old_tail = chain[i:].copy()
        chain[i:] = old_tail + shift
        
        # Calculate new energy
        # Full recalculation
        new_energy = _calculate_energy(sequence, chain)
        
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain.copy()
        else:
            # Revert
            chain[i:] = old_tail
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Fast Metropolis Monte Carlo."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    remaining = _get_remaining_time()
    
    if remaining <= 0:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time
    # We don't know how many calls we have left.
    # Assume we have ~10-20 calls left.
    # Use a fixed chunk of time, e.g., 5 seconds per call.
    # 26 * 5 = 130s. Safe.
    time_limit = min(remaining, 5.0)
    
    best_chain = _monte_carlo_fold(sequence, n, time_limit)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
