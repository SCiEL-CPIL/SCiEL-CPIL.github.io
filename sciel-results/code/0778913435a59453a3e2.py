"""HP lattice protein folding using Pivot Monte Carlo and Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 190.0

def _remaining():
    return _DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbours of a lattice point in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid_move(sequence, chain, i, new_pos, pos_map):
    """
    Check if moving residue i to new_pos is valid (self-avoiding, unit steps).
    """
    n = len(chain)
    if new_pos in pos_map:
        # Allowed to be at new_pos only if it's the residue itself (which we are moving)
        # But pos_map contains all residues. We need to check if it's occupied by someone else.
        if pos_map[new_pos] != i:
            return False
    
    # Check unit distance to previous
    if i > 0:
        prev = chain[i-1]
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            return False
            
    # Check unit distance to next
    if i < n - 1:
        nxt = chain[i+1]
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            return False
            
    return True

def _energy_delta(sequence, chain, i, new_pos, pos_map):
    """
    Calculate the change in energy if residue i moves to new_pos.
    Assumes the move is valid.
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
        
    delta = 0
    is_h = sequence[i] == 'H'
    
    if is_h:
        # Remove contacts with old neighbors
        for q in _neighbours(old_pos):
            j = pos_map.get(q)
            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                delta += 1 # Removing a contact increases energy (less negative)
                
        # Add contacts with new neighbors
        for q in _neighbours(new_pos):
            j = pos_map.get(q)
            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                delta -= 1 # Adding a contact decreases energy (more negative)
                
    return delta

def _pivot_move(sequence, chain, pos_map, rng):
    """
    Perform a pivot move: rotate a segment of the chain around a pivot point.
    Returns (new_chain, new_pos_map, delta_energy) or (None, None, 0) if invalid.
    """
    n = len(chain)
    if n < 2:
        return None, None, 0
    
    # Choose pivot index
    # Pivot is between i-1 and i. We rotate the segment i..n-1 around chain[i-1].
    # Or we can rotate a segment in the middle.
    # Let's pick a random split point.
    # Pivot at index p. We rotate the tail p+1..n-1 around chain[p].
    # Or we can rotate a sub-segment.
    # Let's do a standard pivot: pick a random index i (1 to n-1).
    # The segment i..n-1 is rotated around chain[i-1].
    
    i = rng.randint(1, n - 1)
    pivot = chain[i-1]
    
    # The vector from pivot to chain[i]
    v = (chain[i][0] - pivot[0], chain[i][1] - pivot[1])
    
    # Rotation options: 90 deg clockwise, 90 deg counter-clockwise
    # v = (dx, dy)
    # 90 CW: (dy, -dx)
    # 90 CCW: (-dy, dx)
    # 180: (-dx, -dy)
    
    # We need to check if the rotated segment is valid.
    # This is expensive if done naively.
    # Optimization: We can check collisions with the fixed part (0..i-1) and internal collisions.
    
    # Let's try one random rotation direction
    r = rng.random()
    if r < 0.33:
        nv = (v[1], -v[0])
    elif r < 0.66:
        nv = (-v[1], v[0])
    else:
        nv = (-v[0], -v[1])
        
    # If the rotation is 180, the first step is just reversed.
    # If 90, it's perpendicular.
    
    # We need to propagate the rotation.
    # This is tricky because the "rotation" of a path is not a rigid body rotation
    # in the usual sense unless we define it carefully.
    # A pivot move in polymer physics usually means:
    # Pick a pivot point. Rotate the entire tail around that point.
    # However, on a lattice, a rigid rotation of a path might not map to lattice points
    # if the path is not a straight line.
    
    # Actually, standard "Pivot Move" for lattice polymers:
    # 1. Pick a pivot index i.
    # 2. The segment 0..i-1 is fixed.
    # 3. The segment i..n-1 is rotated by 90 or 180 degrees around the point chain[i-1].
    # BUT: A general lattice path does not map to a valid lattice path under rigid rotation
    # unless the rotation is a symmetry of the lattice (90, 180) AND the path segments align.
    # Wait, if you rotate a vector (1,0) by 90, you get (0,1).
    # If the next step in the original path was (0,1), and you rotate the whole tail,
    # the relative geometry is preserved.
    # So, if we take the tail, apply a rigid rotation around the pivot,
    # the new coordinates will be integers if the original were integers and rotation is 90/180.
    # The only issue is self-avoidance and collision with the fixed head.
    
    # Let's implement rigid rotation of the tail.
    new_chain = chain[:i] # Copy head
    
    # We need to rotate points chain[i], ..., chain[n-1] around pivot.
    # Let's define the rotation function.
    def rotate_point(p, center, angle):
        # p and center are tuples
        # angle: 90, -90, 180
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        if angle == 90:
            ndx, ndy = dy, -dx
        elif angle == -90:
            ndx, ndy = -dy, dx
        elif angle == 180:
            ndx, ndy = -dx, -dy
        else:
            return p
        return (center[0] + ndx, center[1] + ndy)
    
    # Choose angle
    angle = rng.choice([90, -90, 180])
    
    # Generate new tail
    new_tail = []
    valid = True
    occupied_head = set(chain[:i]) # Set of fixed points
    
    for k in range(i, n):
        p_new = rotate_point(chain[k], pivot, angle)
        
        # Check collision with head
        if p_new in occupied_head:
            valid = False
            break
            
        # Check collision with already placed new tail
        if p_new in new_tail:
            valid = False
            break
            
        new_tail.append(p_new)
        
    if not valid:
        return None, None, 0
        
    new_chain.extend(new_tail)
    
    # Verify unit steps at the connection
    if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
        return None, None, 0
        
    # Verify internal unit steps (should be preserved by rotation, but good to be safe? 
    # Rotation preserves distances, so if original was unit, new is unit. 
    # But we must ensure no self-intersection within the tail, which we checked above.)
    
    # Calculate energy delta
    # This is O(N) if we do it naively.
    # We can compute full energy of new_chain and subtract old energy?
    # No, we want delta.
    # Let's just compute the energy of the new chain and return that, 
    # and let the caller manage the state.
    # Actually, for SA, we need delta.
    # Computing full energy is O(N). 
    # Since N is small (<=100), O(N) per move is acceptable.
    
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    
    return new_chain, new_energy

def _segment_move(sequence, chain, pos_map, rng):
    """
    Move a short segment (length 2-4) perpendicular to its axis.
    """
    n = len(chain)
    if n < 5:
        return None, None, 0
        
    seg_len = rng.choice([2, 3, 4])
    # Start index i, such that i-1 and i+seg_len exist
    i = rng.randint(1, n - seg_len - 1)
    
    # Direction of the segment entry
    d_in = (chain[i][0] - chain[i-1][0], chain[i][1] - chain[i-1][1])
    
    # Perpendicular directions
    # If d_in is (1,0), perp is (0,1) and (0,-1)
    # If d_in is (0,1), perp is (1,0) and (-1,0)
    if d_in[0] != 0:
        perp_dirs = [(0, 1), (0, -1)]
    else:
        perp_dirs = [(1, 0), (-1, 0)]
        
    px, py = rng.choice(perp_dirs)
    
    new_chain = list(chain)
    # Shift segment i..i+seg_len-1 by (px, py)
    new_seg = []
    for k in range(seg_len):
        orig = chain[i+k]
        new_p = (orig[0] + px, orig[1] + py)
        new_seg.append(new_p)
        
    # Check validity
    # 1. Connection to i-1
    if abs(new_seg[0][0] - chain[i-1][0]) + abs(new_seg[0][1] - chain[i-1][1]) != 1:
        return None, None, 0
    # 2. Connection to i+seg_len
    end_idx = i + seg_len
    if abs(new_seg[-1][0] - chain[end_idx][0]) + abs(new_seg[-1][1] - chain[end_idx][1]) != 1:
        return None, None, 0
        
    # 3. Self-avoidance
    # The segment moves as a block. We need to check if any new position is occupied
    # by a residue outside the segment [i, i+seg_len-1].
    # Note: The segment might collide with itself? No, it's a rigid shift.
    # But it might collide with the head or tail.
    
    occupied = set(chain)
    # Remove old segment positions
    for p in chain[i:i+seg_len]:
        occupied.remove(p)
        
    for p in new_seg:
        if p in occupied:
            return None, None, 0
            
    # Apply move
    for k in range(seg_len):
        new_chain[i+k] = new_seg[k]
        
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    
    return new_chain, new_energy

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses Pivot Monte Carlo and Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    # Initialize with a simple spiral or random walk
    # Let's use a simple greedy initialization to get a valid start
    # Or just a straight line, which is valid but high energy.
    # A spiral is better.
    
    # Generate a spiral
    def _spiral_init(n):
        chain = [(0, 0)]
        if n == 1:
            return chain
        # Directions: Right, Down, Left, Up
        dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        x, y = 0, 0
        d_idx = 0
        step_count = 0
        steps_in_dir = 1
        dir_change_count = 0
        
        while len(chain) < n:
            dx, dy = dirs[d_idx]
            for _ in range(steps_in_dir):
                if len(chain) >= n:
                    break
                x += dx
                y += dy
                chain.append((x, y))
            d_idx = (d_idx + 1) % 4
            dir_change_count += 1
            if dir_change_count % 2 == 0:
                steps_in_dir += 1
                
        return chain
        
    current_chain = _spiral_init(n)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    # Adjust based on sequence length
    if n <= 20:
        T_start = 2.0
        T_end = 0.1
        max_iter = 5000
    elif n <= 40:
        T_start = 3.0
        T_end = 0.1
        max_iter = 3000
    else:
        T_start = 4.0
        T_end = 0.1
        max_iter = 2000
        
    cooling = (T_start / T_end) ** (1.0 / max_iter)
    T = T_start
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
            
        T = T_start / (cooling ** it)
        
        # Choose move type
        # Pivot moves are global and powerful, but can be expensive if they fail often.
        # Segment moves are local and faster.
        # Let's mix them.
        if rng.random() < 0.5:
            new_chain, new_energy = _pivot_move(sequence, current_chain, None, rng)
        else:
            new_chain, new_energy = _segment_move(sequence, current_chain, None, rng)
            
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
