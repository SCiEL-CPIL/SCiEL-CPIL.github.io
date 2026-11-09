"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline for the entire script execution
# The evaluator kills after 210s. We use 205s to be safe.
_GLOBAL_START_TIME = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START_TIME + 205.0

def _neighbours(point):
    """Yield the 4 neighbours of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of the chain. Energy = - (number of H-H contacts)."""
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbours
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (x + dx, y + dy)
                j = pos_to_idx.get(neighbor)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_chain(sequence):
    """
    Generate a simple initial self-avoiding walk using a spiral heuristic.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
        
    chain = [(0, 0)]
    # Directions: Right, Down, Left, Up
    dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    dir_idx = 0
    steps = 1
    turns = 0
    occupied = {chain[0]}
    
    for i in range(1, n):
        moved = False
        while not moved:
            dx, dy = dirs[dir_idx]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                moved = True
                steps -= 1
                if steps == 0:
                    dir_idx = (dir_idx + 1) % 4
                    turns += 1
                    if turns % 2 == 0:
                        steps += 1
            else:
                # Blocked, turn
                dir_idx = (dir_idx + 1) % 4
                steps = 1
                if turns > n * 2: # Safety break
                    # Fallback: simple line
                    chain = [(i, 0) for i in range(n)]
                    return chain
    return chain

def _pivot_move(chain, pivot_idx, rot):
    """
    Attempt a pivot move.
    Returns the new chain if valid, else None.
    rot: 1 (90 deg CW), 2 (180 deg), 3 (270 deg CW)
    """
    if pivot_idx <= 0 or pivot_idx >= len(chain) - 1:
        return None
        
    # The segment to rotate is chain[pivot_idx:]
    # We rotate around chain[pivot_idx-1]
    pivot_point = chain[pivot_idx - 1]
    px, py = pivot_point
    
    new_chain = list(chain[:pivot_idx])
    
    # Rotation logic
    # 1: 90 CW: (x,y) -> (y, -x) relative to pivot
    # 2: 180:   (x,y) -> (-x, -y)
    # 3: 270 CW: (x,y) -> (-y, x)
    
    occupied = set(chain[:pivot_idx])
    
    for i in range(pivot_idx, len(chain)):
        x, y = chain[i]
        # Relative coordinates
        rx, ry = x - px, y - py
        
        if rot == 1:
            nx, ny = ry, -rx
        elif rot == 2:
            nx, ny = -rx, -ry
        elif rot == 3:
            nx, ny = -ry, rx
        else:
            return None
            
        new_pos = (px + nx, py + ny)
        
        if new_pos in occupied:
            return None
        occupied.add(new_pos)
        new_chain.append(new_pos)
        
    return new_chain

def _local_move(chain, idx, direction_idx):
    """
    Attempt a local move of a single residue.
    Returns the new chain if valid, else None.
    direction_idx: 0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)
    """
    if idx <= 0 or idx >= len(chain) - 1:
        return None
        
    x, y = chain[idx]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dx, dy = dirs[direction_idx]
    new_pos = (x + dx, y + dy)
    
    # Check collision
    occupied = set(chain)
    occupied.discard(chain[idx])
    
    if new_pos in occupied:
        return None
        
    new_chain = list(chain)
    new_chain[idx] = new_pos
    return new_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if len(sequence) == 0:
        return []
    if len(sequence) == 1:
        return [[0, 0]]
        
    # Time budget for this specific sequence
    # We have ~205s total. 10 sequences. ~20s per sequence.
    # Let's be conservative: 18s per sequence.
    seq_deadline = time.time() + 18.0
    
    # 1. Initial Conformation
    chain = _initial_chain(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Simulated Annealing
    T = 5.0
    T_min = 0.1
    # Estimated steps: 18s / 50us ~ 360,000
    # (0.1/5)^(1/360000) ~ 0.999985
    cooling_rate = 0.999985
    
    rng = random.Random(42)
    
    while time.time() < seq_deadline and T > T_min:
        # Choose move type
        if rng.random() < 0.5:
            # Local Move
            idx = rng.randint(1, len(chain) - 2)
            dir_idx = rng.randint(0, 3)
            new_chain = _local_move(chain, idx, dir_idx)
        else:
            # Pivot Move
            pivot_idx = rng.randint(1, len(chain) - 1)
            rot = rng.randint(1, 3)
            new_chain = _pivot_move(chain, pivot_idx, rot)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        T *= cooling_rate
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
