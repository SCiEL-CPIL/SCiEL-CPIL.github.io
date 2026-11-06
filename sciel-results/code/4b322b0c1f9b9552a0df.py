"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-start Simulated Annealing with Kink and Pivot moves.
Initialization via fast Greedy Growth.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 10.0  # 10s safety margin

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    if n == 0:
        return 0
    # Map position to index for O(1) lookup
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_growth(sequence, dimension, beam_width=5):
    """Fast greedy growth with a small beam to initialize SA."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
    
    start = (0,) * dimension
    initial_chain = [start]
    initial_occupied = {start: 0}
    
    beam = [(0, initial_chain, initial_occupied)]
    
    for i in range(1, n):
        new_beam = []
        for score, chain, occupied in beam:
            last = chain[-1]
            residue = sequence[i]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Small penalty for trapping? Or just maximize gain.
                # We keep the top beam_width states.
                new_score = score + gain
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beam.append((new_score, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        if _time_remaining() < 5.0:
            break
    
    if not beam:
        return None
    
    best_entry = max(beam, key=lambda x: x[0])
    return best_entry[1]

def _kink_move(chain, rng):
    """Kink move: reverse a sub-chain."""
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)
    segment = chain[i + 1:j + 1]
    new_segment = segment[::-1]
    
    occupied = set(chain)
    for p in segment:
        occupied.discard(p)
    
    valid = True
    for p in new_segment:
        if p in occupied:
            valid = False
            break
    if not valid:
        return None
    
    # Check connectivity
    if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
        return None
    if j + 1 < n:
        if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
            return None
    
    new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
    return new_chain

def _pivot_move(chain, rng):
    """Pivot move: rotate a sub-chain around a bond."""
    n = len(chain)
    if n < 4:
        return None
    
    # Pick a bond k (0 to n-2). The sub-chain is k+1 to n-1.
    # We pick k such that the sub-chain has at least 2 residues? 
    # Actually, rotating a single residue is just a local move.
    # Let's pick k from 1 to n-3 to ensure the segment is not too small or too large?
    # Random k from 0 to n-2.
    k = rng.randint(0, n - 2)
    
    # Axis is chain[k] to chain[k+1]
    p0 = chain[k]
    p1 = chain[k+1]
    axis_vec = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    
    # Determine the axis direction
    # axis_vec is one of (1,0,0), (-1,0,0), etc.
    
    # We will rotate the segment chain[k+1:] around the axis.
    # The point chain[k+1] is on the axis, so it stays fixed?
    # No, the bond is between k and k+1. The rotation is around the line passing through k and k+1.
    # So chain[k] and chain[k+1] are fixed. The rest moves.
    
    # Wait, if we rotate around the bond k-k+1, the point k+1 is on the axis, so it doesn't move.
    # The point k+2 moves.
    
    # Let's identify the axis component.
    if axis_vec[0] != 0:
        axis = 0
    elif axis_vec[1] != 0:
        axis = 1
    else:
        axis = 2
        
    # Generate 3 possible rotations (90, 180, 270 degrees)
    # For a lattice point, rotating 90 degrees around an axis swaps the other two coordinates.
    # Let's define a helper to rotate a point relative to p0 around the axis.
    
    def rotate_point(p, axis, p0, angle):
        # Translate
        x = p[0] - p0[0]
        y = p[1] - p0[1]
        z = p[2] - p0[2]
        
        if axis == 0: # x-axis
            if angle == 1: # 90
                nx, ny, nz = x, z, -y
            elif angle == 2: # 180
                nx, ny, nz = x, -y, -z
            else: # 270
                nx, ny, nz = x, -z, y
        elif axis == 1: # y-axis
            if angle == 1: # 90
                nx, ny, nz = -z, y, x
            elif angle == 2: # 180
                nx, ny, nz = -x, y, -z
            else: # 270
                nx, ny, nz = z, y, -x
        else: # z-axis
            if angle == 1: # 90
                nx, ny, nz = y, -x, z
            elif angle == 2: # 180
                nx, ny, nz = -x, -y, z
            else: # 270
                nx, ny, nz = -y, x, z
        
        return (nx + p0[0], ny + p0[1], nz + p0[2])

    # Try 3 angles
    for angle in [1, 2, 3]:
        new_chain = list(chain)
        # The segment to move is chain[k+1:]
        # But chain[k+1] is on the axis, so it should remain unchanged?
        # Let's check: rotate_point(chain[k+1], axis, p0, angle)
        # If chain[k+1] is on the axis, x,y,z relative to p0 will have 2 zeros.
        # So it should map to itself.
        
        valid = True
        occupied = set(chain)
        # Remove the segment we are moving
        for i in range(k + 1, n):
            occupied.discard(chain[i])
            
        # Check connectivity at k+1
        # chain[k] is fixed. chain[k+1] is fixed (on axis).
        # We need to check if the new chain[k+2] is a neighbor of chain[k+1].
        
        for i in range(k + 1, n):
            new_p = rotate_point(chain[i], axis, p0, angle)
            new_chain[i] = new_p
            if new_p in occupied:
                valid = False
                break
            occupied.add(new_p)
            
            # Check connectivity for i > k+1
            if i > k + 1:
                prev_p = new_chain[i-1]
                if abs(new_p[0] - prev_p[0]) + abs(new_p[1] - prev_p[1]) + abs(new_p[2] - prev_p[2]) != 1:
                    valid = False
                    break
        
        if valid:
            # Check connectivity at k+1
            # chain[k+1] is fixed. chain[k+2] is moved.
            if n > k + 2:
                if abs(new_chain[k+2][0] - chain[k+1][0]) + abs(new_chain[k+2][1] - chain[k+1][1]) + abs(new_chain[k+2][2] - chain[k+1][2]) != 1:
                    continue # Try next angle
            
            return new_chain
    
    return None

def _simulated_annealing(sequence, chain, rng, time_budget):
    """Simulated annealing with Kink and Pivot moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    cooling_rate = 0.999 # Slower cooling
    
    start_t = time.time()
    while time.time() - start_t < time_budget and T > T_min:
        # Choose move: 50% Kink, 50% Pivot
        if rng.random() < 0.5:
            new_chain = _kink_move(chain, rng)
        else:
            new_chain = _pivot_move(chain, rng)
            
        if new_chain is None:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence using Multi-start SA with Kink and Pivot moves."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Allocate time based on sequence length
    # We want to finish all 26 sequences within 330s.
    # Assume average time per sequence is 330/26 ~ 12s.
    # Short sequences can be faster.
    
    if n <= 48:
        num_restarts = 15
        sa_budget_per_restart = 0.5
    elif n <= 64:
        num_restarts = 10
        sa_budget_per_restart = 1.0
    elif n <= 96:
        num_restarts = 8
        sa_budget_per_restart = 1.5
    else:
        num_restarts = 5
        sa_budget_per_restart = 2.0
        
    best_chain = None
    best_energy = 0
    
    for restart in range(num_restarts):
        if _time_remaining() < 10.0:
            break
        
        # Greedy growth initialization
        chain = _greedy_growth(sequence, dimension, beam_width=5)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Simulated annealing
        sa_budget = min(sa_budget_per_restart, _time_remaining() / 2.0)
        if sa_budget > 0.1:
            improved_chain, improved_energy = _simulated_annealing(sequence, chain, rng, sa_budget)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
