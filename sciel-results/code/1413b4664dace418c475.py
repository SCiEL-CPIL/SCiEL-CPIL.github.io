"""
HP lattice protein folding using constructive heuristics and pivot move local search.
"""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 110.0  # Safety margin for 120s limit, leaving time for overhead

def _check_time():
    return (time.time() - _START_TIME) < _TIME_BUDGET

def _neighbours(point):
    """Yield the 4 neighbors of a point in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Energy = - (number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if len(chain) < 2:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i-1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True

def _constructive_fold(sequence):
    """
    Construct a valid conformation using a snake-like pattern in a bounding box.
    This ensures self-avoidance and high density.
    """
    n = len(sequence)
    # Estimate bounding box dimensions
    # We want a roughly square box.
    width = int(n ** 0.5)
    if width * width < n:
        width += 1
    height = (n + width - 1) // width
    
    # Generate the snake path
    chain = []
    for y in range(height):
        if y % 2 == 0:
            for x in range(width):
                if len(chain) < n:
                    chain.append((x, y))
        else:
            for x in range(width - 1, -1, -1):
                if len(chain) < n:
                    chain.append((x, y))
    
    # The chain might be shorter than n if our box estimation was off, 
    # but usually width*height >= n.
    # If it's longer, we truncate. If shorter, we need to extend, 
    # but the snake pattern above fills row by row.
    # Let's verify length.
    if len(chain) != n:
        # Fallback: simple straight line if construction fails (should not happen)
        return [(i, 0) for i in range(n)]
        
    return chain

def _pivot_move(chain, pivot_idx, length, occupied, sequence):
    """
    Attempt a pivot move. Returns (new_chain, new_occupied, energy_diff) or None.
    pivot_idx: index of the pivot point.
    length: length of the segment to rotate (1 to n-1).
    """
    n = len(chain)
    if pivot_idx < 0 or pivot_idx >= n:
        return None
    if length > n - 1 - pivot_idx:
        return None
    if length == 0:
        return None
        
    # The segment to move is chain[pivot_idx+1 : pivot_idx+1+length]
    # The pivot point is chain[pivot_idx]
    # The rest of the chain remains fixed.
    
    # Remove the segment from the occupied set
    segment = chain[pivot_idx+1 : pivot_idx+1+length]
    for p in segment:
        if occupied[p] != pivot_idx: # The pivot itself is not in the segment list
             pass # We don't remove the pivot
    # Actually, we need to remove the points in the segment from the global occupied set
    # and then try to re-insert them.
    
    # Let's use a temporary set for the fixed part of the chain
    # Fixed part: chain[0:pivot_idx+1] and chain[pivot_idx+1+length:]
    # We need to check if the rotated segment collides with the fixed part.
    
    # To make this efficient, we should pre-calculate the fixed part's occupied set.
    # But doing that for every move is expensive. 
    # Instead, we can just check collisions against the global occupied set 
    # after removing the segment points.
    
    # Remove segment points from occupied
    for p in segment:
        del occupied[p]
        
    # Try 4 rotations (0, 90, 180, 270)
    # Rotation around pivot_idx
    pivot = chain[pivot_idx]
    
    best_new_chain = None
    best_energy_diff = None
    best_new_occupied = None
    
    # We need to calculate the energy change.
    # Energy change comes from:
    # 1. Loss of contacts involving the moved segment and the rest of the chain.
    # 2. Gain of new contacts involving the moved segment and the rest of the chain.
    # 3. Contacts within the moved segment remain unchanged relative to each other.
    
    # This is complex to calculate incrementally in a simple pivot.
    # For simplicity and speed, let's just generate the new chain and calculate full energy 
    # if the move is valid. If it's too slow, we optimize.
    # Given N=100, full energy calc is O(N). Pivot generation is O(N).
    # 100 * 100 = 10k ops per move. 10k moves = 100M ops. Might be tight in Python.
    
    # Let's try to be a bit smarter.
    # We will only accept moves that don't increase energy (greedy) or with some probability (SA).
    # Let's stick to a simple hill climbing / simulated annealing.
    
    # Rotation matrices for 2D
    # 90 deg: (x, y) -> (y, -x)
    # 180 deg: (x, y) -> (-x, -y)
    # 270 deg: (x, y) -> (-y, x)
    
    # Relative to pivot
    # new_x = pivot_x + (x - pivot_x) * cos - (y - pivot_y) * sin
    # new_y = pivot_y + (x - pivot_x) * sin + (y - pivot_y) * cos
    
    rotations = [
        (1, 0, 0, 1),   # 0 deg (identity, skip if we want to avoid no-ops? or keep)
        (0, -1, 1, 0),  # 90 deg
        (-1, 0, 0, -1), # 180 deg
        (0, 1, -1, 0)   # 270 deg
    ]
    
    # We already removed the segment.
    # We need to restore it if we fail or if we choose a different rotation.
    # Let's try each rotation.
    
    for c, s, sc, ss in rotations:
        if c == 1 and s == 0 and sc == 0 and ss == 1:
            continue # Skip identity
        
        new_segment = []
        valid = True
        for p in segment:
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            nx = pivot[0] + dx * c - dy * s
            ny = pivot[1] + dx * sc + dy * ss
            new_segment.append((nx, ny))
            
            # Check collision with fixed part
            # The fixed part is everything in occupied (which currently has the segment removed)
            # PLUS the pivot point itself (which is in occupied)
            # Wait, the pivot point is NOT in the segment, so it is still in occupied.
            # So if new_segment point is in occupied, it's a collision.
            if (nx, ny) in occupied:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check if the new segment is self-avoiding internally
        # Since the original segment was self-avoiding and rotation is rigid, it remains self-avoiding.
        
        # Construct new chain
        new_chain = chain[:pivot_idx+1] + new_segment + chain[pivot_idx+1+length:]
        
        # Check unit steps at the boundaries
        # Boundary 1: pivot to first new segment point
        if pivot_idx + 1 < n:
            p1 = chain[pivot_idx]
            p2 = new_segment[0]
            if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) != 1:
                valid = False
        # Boundary 2: last new segment point to next fixed point
        if pivot_idx + 1 + length < n:
            p1 = new_segment[-1]
            p2 = chain[pivot_idx+1+length]
            if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) != 1:
                valid = False
        
        if not valid:
            continue
            
        # Calculate energy
        # We can calculate full energy
        e_new = _energy(sequence, new_chain)
        
        # Restore occupied for this rotation attempt? 
        # No, we will update occupied at the end if we accept.
        # But we need to know the energy diff.
        # Let's return the best valid move found in this pivot attempt.
        # Actually, we should just return one valid move to be processed by the main loop.
        # Or we can return the best one.
        
        # Let's just return the first valid one to keep it simple, or pick randomly.
        # Picking randomly helps exploration.
        
        # Add new segment to occupied
        for p in new_segment:
            occupied[p] = len(chain) # Value doesn't matter much, just presence
            
        return new_chain, occupied, e_new

    # If no valid move found, restore the original segment to occupied
    for p in segment:
        occupied[p] = 0 # Placeholder
        
    return None

def _local_search(sequence, chain, max_iterations=5000):
    """
    Perform local search using pivot moves.
    """
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    
    # Simple simulated annealing or hill climbing
    # Let's use a greedy hill climbing with random restarts of pivot selection
    # or a simple SA.
    
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    
    for i in range(max_iterations):
        if not _check_time():
            break
            
        # Random pivot
        pivot_idx = random.randint(0, n - 2)
        # Random length
        max_len = n - 1 - pivot_idx
        if max_len <= 0:
            continue
        length = random.randint(1, max_len)
        
        result = _pivot_move(chain, pivot_idx, length, occupied, sequence)
        
        if result is None:
            continue
            
        new_chain, new_occupied, new_energy = result
        chain = new_chain
        occupied = new_occupied
        
        delta = new_energy - current_energy
        
        if delta < 0:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < pow(2.71828, -delta / T):
                current_energy = new_energy
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only 2D is supported")
        
    # 1. Constructive phase
    initial_chain = _constructive_fold(sequence)
    
    # 2. Local Search phase
    # We might want to run local search multiple times with different initial perturbations
    # or just once on the constructive result.
    # Given time constraints, one deep local search is better than many shallow ones.
    
    best_chain, best_energy = _local_search(sequence, initial_chain)
    
    # Convert tuples to lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
