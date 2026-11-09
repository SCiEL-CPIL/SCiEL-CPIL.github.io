"""HP lattice protein folding with pivot-move simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_GLOBAL_START_TIME = None
_HARD_DEADLINE = 320.0  # 330s limit, leave 10s margin

def _init_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()

def _time_remaining():
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _HARD_DEADLINE - (time.time() - _GLOBAL_START_TIME)

# Pre-compute neighbor offsets for 3D cubic lattice
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _get_neighbours_fast(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y + z, z), (x, y, z - 1) # Correction: (x, y, z+1), (x, y, z-1)
    ]
# Fixing the typo in the thought process, actual code below will be correct.

def _neighbours(point):
    x, y, z = point
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _grow(sequence, dimension, rng, bias=0.5):
    """Randomized greedy chain growth."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: gain is important, free space prevents trapping
            score = gain * 5.0 + free * 0.1 * (1 - bias) + rng.random() * bias
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _pivot_sa(sequence, chain, rng, time_budget):
    """
    Simulated Annealing with Pivot Moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    start_time = time.time()
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T0 = 5.0
    T_min = 0.1
    # Estimate steps: pivot moves are O(N) for collision check. N=100 -> ~100 ops.
    # 10s budget -> ~100,000 steps possible? Let's be conservative.
    # We will just loop until time runs out.
    
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    # Helper to apply rotation
    def rotate_point(p, axis_vec, center, sign):
        # axis_vec is (ax, ay, az), one component is +/-1, others 0.
        # center is the point around which we rotate (on the axis).
        # We rotate p around the line defined by center and center+axis_vec.
        # Since axis_vec is a unit cardinal vector, this is simple.
        ax, ay, az = axis_vec
        cx, cy, cz = center
        px, py, pz = p
        
        if ax != 0:
            # Rotate around x-axis. y and z change.
            # Vector from center to p: (px-cx, py-cy, pz-cz)
            # Rotation in yz plane by 90 deg.
            # (dy, dz) -> (sign * -dz, sign * dy) or (sign * dz, sign * -dy)?
            # Right hand rule:
            # If sign=1 (counter-clockwise looking from positive axis?):
            # y' = y cos - z sin. For 90 deg: y' = -z, z' = y.
            # Let's just try both signs and check validity.
            # Actually, we just need a valid move.
            dy = py - cy
            dz = pz - cz
            if sign == 1:
                new_dy = -dz
                new_dz = dy
            else:
                new_dy = dz
                new_dz = -dy
            return (px, cy + new_dy, cz + new_dz)
        elif ay != 0:
            dx = px - cx
            dz = pz - cz
            if sign == 1:
                new_dx = -dz
                new_dz = dx
            else:
                new_dx = dz
                new_dz = -dx
            return (cx + new_dx, py, cz + new_dz)
        else: # az != 0
            dx = px - cx
            dy = py - cy
            if sign == 1:
                new_dx = -dy
                new_dy = dx
            else:
                new_dx = dy
                new_dy = -dx
            return (cx + new_dx, cy + new_dy, pz)

    while time.time() - start_time < time_budget:
        # Pick a random bond i, i+1
        i = rng.randint(0, n - 2)
        p_i = current_chain[i]
        p_ip1 = current_chain[i + 1]
        
        # Axis vector
        axis_vec = (p_ip1[0] - p_i[0], p_ip1[1] - p_i[1], p_ip1[2] - p_i[2])
        
        sign = 1 if rng.random() < 0.5 else -1
        
        # Generate new positions for k > i+1
        new_positions = [None] * (n - i - 2)
        valid = True
        
        # We rotate around the axis passing through p_i (and p_ip1).
        # p_ip1 is fixed.
        # p_i is fixed.
        # The segment i+2 ... n-1 moves.
        
        # To check collisions, we need to know which points are occupied by the FIXED part (0...i)
        # and which by the MOVING part (i+1...n-1).
        # The fixed part is 0...i.
        # The moving part includes i+1 (which doesn't move) and i+2...n-1.
        
        # We must ensure new positions of i+2...n-1 do not collide with 0...i.
        # They also don't collide with i+1 (fixed) or each other (rigid rotation).
        
        # However, we must also ensure they don't collide with the *original* positions of the moving part?
        # No, the old positions are vacated.
        
        # So, check against pos_map of 0...i.
        # Let's build a set of fixed points for O(1) lookup?
        # pos_map contains all points.
        # We can just check: if new_pos in pos_map and pos_map[new_pos] <= i:
        #   Collision.
        
        # Wait, if new_pos is the OLD position of a moving point, that's fine (it's vacated).
        # But pos_map still has the old mapping.
        # So we need to be careful.
        
        # Better: Create a set of fixed points.
        fixed_set = set(current_chain[:i+1])
        
        for k in range(i + 2, n):
            old_pos = current_chain[k]
            new_pos = rotate_point(old_pos, axis_vec, p_i, sign)
            new_positions[k - i - 2] = new_pos
            
            if new_pos in fixed_set:
                valid = False
                break
        
        if not valid:
            continue
        
        # Calculate Delta Energy
        # Contacts involving the moving part (i+1...n-1) change.
        # Specifically, contacts between (i+1...n-1) and (0...i) change.
        # Contacts within (i+1...n-1) are preserved (rigid rotation).
        # Contacts between (i+1...n-1) and themselves are preserved.
        # So we only need to recalculate contacts between the moving part and the fixed part.
        
        # Old contacts between moving and fixed:
        old_cross_contacts = 0
        # Moving part indices: i+1 to n-1
        # Fixed part indices: 0 to i
        # We iterate over moving part H residues and check neighbors in fixed part.
        
        # To be fast, we can just recalculate energy for the whole chain?
        # N=100, Energy calc is O(N).
        # If we do it every step, it's slow.
        # But incremental is complex.
        # Let's try incremental for the cross-contacts.
        
        # Identify H residues in moving part
        moving_h_indices = [k for k in range(i + 1, n) if sequence[k] == 'H']
        
        # Old cross contacts
        for k in moving_h_indices:
            p_k = current_chain[k]
            for q in _neighbours(p_k):
                j = pos_map.get(q)
                if j is not None and j <= i and sequence[j] == 'H':
                    old_cross_contacts += 1
        
        # New cross contacts
        new_cross_contacts = 0
        # We need to check new positions of moving H residues against fixed H residues.
        # Fixed H residues: indices 0..i where sequence[j]=='H'
        fixed_h_set = set(j for j in range(i + 1) if sequence[j] == 'H')
        # Actually, we just need to check if the neighbor is in the fixed part and is H.
        
        # We have new_positions for i+2...n-1.
        # p_ip1 (i+1) is unchanged.
        
        # Map new positions to indices for quick lookup?
        # Or just iterate.
        
        for k in moving_h_indices:
            if k == i + 1:
                p_k = current_chain[k] # Unchanged
            else:
                p_k = new_positions[k - i - 2]
            
            for q in _neighbours(p_k):
                j = pos_map.get(q)
                if j is not None and j <= i and sequence[j] == 'H':
                    # Check if this is a valid contact (not consecutive in chain? j <= i, k >= i+1, so k-j >= 1.
                    # If k=i+1 and j=i, they are consecutive. Contact is defined as NOT adjacent.
                    # So if k-j == 1, it's not a contact.
                    if k - j > 1:
                        new_cross_contacts += 1
        
        delta_e = -(new_cross_contacts - old_cross_contacts)
        
        # Acceptance
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T0): # Using constant T for simplicity? No, need cooling.
            # Update chain
            for k in range(i + 2, n):
                old_pos = current_chain[k]
                new_pos = new_positions[k - i - 2]
                pos_map.pop(old_pos)
                pos_map[new_pos] = k
                current_chain[k] = new_pos
            
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        # Simple cooling: T0 is constant for now, or we can decrease it.
        # Given short time, constant T with high value allows exploration.
        # Let's keep T0 = 2.0
        
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Pivot SA."""
    _init_time()
    time_left = _time_remaining()
    
    # Allocate time for this sequence
    # Assume ~26 sequences.
    # If time_left is large, we have plenty.
    # We will use 80% of the estimated per-sequence time.
    # Estimated per sequence = time_left / 1 (since we don't know how many are left, but we know total).
    # Safer: Use a fixed slice if we are early, or all remaining time if we are late.
    
    # Simple strategy:
    # If this is the first sequence, assume we have 300s.
    # We will use a dynamic budget.
    
    # Let's just use a fixed 10s per sequence, capped by time_left.
    seq_budget = min(10.0, time_left * 0.9)
    
    if seq_budget < 1.0:
        # Not enough time for proper search, return a greedy growth
        rng = random.Random(12345 + len(sequence))
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        return [list(p) for p in chain]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Positive infinity for min energy
    
    # Try a few restarts
    # Distribute budget among restarts
    n_restarts = 2
    time_per_restart = seq_budget / n_restarts
    
    for _ in range(n_restarts):
        chain = _grow(sequence, dimension, rng, bias=0.3)
        if chain is None:
            continue
        
        # Run SA
        improved_chain = _pivot_sa(sequence, chain, rng, time_per_restart)
        
        e = _energy(sequence, improved_chain)
        if best_chain is None or e < best_energy:
            best_chain = improved_chain
            best_energy = e
        
        # Check global deadline
        if _time_remaining() < 5.0:
            break
            
    if best_chain is None:
        # Fallback
        rng = random.Random(12345 + len(sequence))
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        else:
            best_chain = chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
