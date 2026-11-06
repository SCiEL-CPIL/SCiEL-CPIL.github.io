"""HP lattice protein folding: Fast SAW init + Delta-Energy Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _generate_random_saw(n, rng):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    # Simple backtracking-free random walk with restarts
    for _ in range(50):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        success = True
        for i in range(1, n):
            x, y = chain[-1]
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    return None


def _energy_full(seq, chain):
    """Calculate energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _try_local_search(sequence, chain, rng, max_iters, start_temp=3.0, end_temp=0.1):
    """
    Local search using Segment, Kink, and Reversal moves with Simulated Annealing.
    Uses delta-energy calculation for speed.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy_full(sequence, chain)

    is_h = [c == 'H' for c in sequence]
    
    # State
    current_chain = list(chain)
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy_full(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Pre-compute neighbor offsets
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    
    def get_delta_energy(moved_indices, new_chain, old_chain, old_pos_to_idx):
        """
        Calculate the change in energy when residues at 'moved_indices' move from old_chain to new_chain.
        We only need to check neighbors of the moved residues.
        """
        # 1. Remove old contacts involving moved residues
        old_energy_contrib = 0
        for i in moved_indices:
            if not is_h[i]:
                continue
            ox, oy = old_chain[i]
            for dx, dy in offsets:
                j = old_pos_to_idx.get((ox + dx, oy + dy))
                if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                    # Count each contact once. We sum over all moved H.
                    # If both i and j are moved, we might double count or miss?
                    # It's safer to count contacts where i is moved and j is ANY H.
                    # But we need to be careful not to double count i-j and j-i.
                    # Standard trick: Count if i < j.
                    if i < j:
                        old_energy_contrib += 1
        
        # 2. Add new contacts involving moved residues
        new_energy_contrib = 0
        new_pos_to_idx = {p: i for i, p in enumerate(new_chain)}
        for i in moved_indices:
            if not is_h[i]:
                continue
            nx, ny = new_chain[i]
            for dx, dy in offsets:
                j = new_pos_to_idx.get((nx + dx, ny + dy))
                if j is not None and j != i and abs(j - i) > 1 and is_h[j]:
                    if i < j:
                        new_energy_contrib += 1
        
        return (new_energy_contrib - old_energy_contrib) * -1 # Energy is -contacts


    def try_segment_move(chain, pos_to_idx):
        """
        Pick a segment [i, j] and try to rotate it around chain[i-1] or similar.
        Actually, standard segment move: Pick pivot i, segment i+1..j.
        Rotate segment around i.
        """
        if n < 4:
            return None, None, None
        
        # Choose segment length: bias towards shorter segments for speed and validity
        max_seg = min(n - 2, 12)
        seg_len = rng.randint(2, max_seg)
        
        # Choose pivot index i (segment is i+1 to i+seg_len)
        # i can be 0 to n-seg_len-1
        i = rng.randint(0, n - seg_len - 1)
        j = i + seg_len
        
        pivot_pos = chain[i]
        
        # The segment to move is chain[i+1 : j+1]
        # The fixed parts are chain[:i+1] and chain[j+1:]
        fixed_head = set(chain[:i+1])
        fixed_tail = set(chain[j+1:]) if j + 1 < n else set()
        
        # The moving segment
        old_seg = chain[i+1 : j+1]
        
        # Try 3 rotations (90, 180, 270) relative to pivot?
        # Or just try to find a new valid path for the segment?
        # Rotating a segment is a specific type of move.
        # Let's try the "Crankshaft" or "Segment Rotation" where we rotate the whole segment around the pivot.
        
        # Rotation around pivot_pos
        # Vector from pivot to point p: v = p - pivot
        # Rotate v, new_p = pivot + R(v)
        
        rotations = []
        for rot in range(1, 4): # 90, 180, 270
            new_seg = []
            valid = True
            for p in old_seg:
                rx, ry = p[0] - pivot_pos[0], p[1] - pivot_pos[1]
                if rot == 1: # 90 CCW: (x,y) -> (-y, x)
                    new_x, new_y = -ry, rx
                elif rot == 2: # 180: (x,y) -> (-x, -y)
                    new_x, new_y = -rx, -ry
                else: # 270 CCW: (x,y) -> (y, -x)
                    new_x, new_y = ry, -rx
                
                new_p = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
                
                # Check if new_p is in fixed parts
                if new_p in fixed_head or new_p in fixed_tail:
                    valid = False
                    break
                # Check if new_p is already in new_seg (self-collision within segment)
                if new_p in new_seg:
                    valid = False
                    break
                
                new_seg.append(new_p)
            
            if valid:
                # Check connectivity at the ends
                # Start of new_seg must be adjacent to pivot (chain[i])
                if abs(new_seg[0][0] - pivot_pos[0]) + abs(new_seg[0][1] - pivot_pos[1]) != 1:
                    continue
                # End of new_seg must be adjacent to chain[j+1] if it exists
                if j + 1 < n:
                    tail_pos = chain[j+1]
                    if abs(new_seg[-1][0] - tail_pos[0]) + abs(new_seg[-1][1] - tail_pos[1]) != 1:
                        continue
                # If j+1 == n, no tail constraint, but wait, segment move usually keeps ends fixed?
                # No, segment move rotates the segment. The end of the segment moves.
                # So the chain is broken? No, the segment is part of the chain.
                # If we rotate the segment, the connection to the tail (chain[j+1]) might break.
                # Standard segment move in SA:
                # 1. Pick i, j.
                # 2. Try to move segment i+1..j to a new position.
                # 3. Must remain connected to i and j+1.
                # So yes, the end must connect to j+1.
                
                # Construct new chain
                new_chain = chain[:i+1] + new_seg + chain[j+1:]
                
                # Check self-avoidance globally (just in case, though fixed_head/tail check covers most)
                # We already checked against fixed_head and fixed_tail.
                # And within new_seg.
                # So it should be valid.
                
                return new_chain, list(range(i+1, j+1)), pos_to_idx
        
        return None, None, None

    def try_kink_move(chain, pos_to_idx):
        """
        Move two consecutive residues i, i+1.
        """
        if n < 4:
            return None, None, None
        
        i = rng.randint(1, n - 3)
        p0 = chain[i-1]
        p1 = chain[i]
        p2 = chain[i+1]
        p3 = chain[i+2]
        
        # Try all possible new positions for p1 and p2
        # p1_new must be neighbor of p0
        # p2_new must be neighbor of p1_new
        # p2_new must be neighbor of p3
        
        occupied = set(chain)
        occupied.discard(p1)
        occupied.discard(p2)
        
        for p1_new in ((p0[0]+1, p0[1]), (p0[0]-1, p0[1]), (p0[0], p0[1]+1), (p0[0], p0[1]-1)):
            if p1_new == p2:
                continue
            if p1_new in occupied:
                continue
            
            for p2_new in ((p1_new[0]+1, p1_new[1]), (p1_new[0]-1, p1_new[1]), (p1_new[0], p1_new[1]+1), (p1_new[0], p1_new[1]-1)):
                if p2_new == p0:
                    continue
                if p2_new in occupied:
                    continue
                if not (abs(p2_new[0] - p3[0]) + abs(p2_new[1] - p3[1]) == 1):
                    continue
                
                new_chain = list(chain)
                new_chain[i] = p1_new
                new_chain[i+1] = p2_new
                
                return new_chain, [i, i+1], pos_to_idx
        
        return None, None, None

    def try_reversal_move(chain, pos_to_idx):
        """
        Reverse segment [i, j].
        """
        if n < 4:
            return None, None, None
        
        max_len = min(n - 2, 8)
        L = rng.randint(2, max_len)
        i = rng.randint(0, n - L - 1)
        j = i + L
        
        new_chain = list(chain)
        new_chain[i:j+1] = chain[i:j+1][::-1]
        
        # Check validity
        # Only need to check connections at i-1/i and j/j+1 and self-avoidance in the reversed part?
        # Reversing a SAW is always a SAW internally.
        # Check connections:
        if i > 0:
            if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
                return None, None, None
        if j < n - 1:
            if abs(new_chain[j][0] - new_chain[j+1][0]) + abs(new_chain[j][1] - new_chain[j+1][1]) != 1:
                return None, None, None
        
        # Check self-avoidance
        # The reversed segment occupies the same set of points, just in reverse order.
        # So no new self-collisions are introduced internally.
        # No collisions with outside because the set of points is the same.
        # So it is always valid if connections hold.
        
        return new_chain, list(range(i, j+1)), pos_to_idx

    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        progress = it / max(1, max_iters)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move
        r = rng.random()
        new_chain = None
        moved_indices = None
        
        if r < 0.4:
            new_chain, moved_indices, old_pos_to_idx = try_segment_move(current_chain, pos_to_idx)
        elif r < 0.7:
            new_chain, moved_indices, old_pos_to_idx = try_kink_move(current_chain, pos_to_idx)
        else:
            new_chain, moved_indices, old_pos_to_idx = try_reversal_move(current_chain, pos_to_idx)
            
        if new_chain is None:
            continue
            
        # Calculate delta energy
        # Note: moved_indices are indices in the chain.
        # The set of indices is the same, only positions change.
        delta = get_delta_energy(moved_indices, new_chain, current_chain, pos_to_idx)
        
        # Acceptance
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_chain = new_chain
            current_energy += delta
            pos_to_idx = {p: i for i, p in enumerate(current_chain)}
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)
    
    # Number of restarts based on sequence length
    # Since init is fast, we can do more restarts
    if n <= 30:
        num_restarts = 60
    elif n <= 50:
        num_restarts = 40
    elif n <= 70:
        num_restarts = 25
    else:
        num_restarts = 15
    
    # Iterations per restart
    # Delta energy is fast, so we can do many more iterations
    iters_per_restart = max(2000, int(2000000 / n))
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _generate_random_saw(n, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Run local search
        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
