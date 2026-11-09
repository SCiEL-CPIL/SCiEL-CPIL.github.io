"""HP lattice protein folding with Simulated Annealing, Pivot moves, and Regrowth."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
# The evaluator gives 240s for all sequences. We assume a hard kill at 330s.
# We need to be safe. Let's use a global deadline.
# However, `fold` is called per sequence. If the process stays alive, we can track global time.
# If the process is restarted per sequence, we can't.
# The prompt says "fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together".
# This phrasing is slightly ambiguous. "subprocess" (singular) suggests one process for all.
# "240 s CPU budget for all sequences together" confirms shared budget.
# So we can use a global deadline.

# Let's assume the worst case: we don't know when we are called.
# We will track global time and stop if we are close to the limit.
# Also, we will allocate a per-call budget that shrinks as we approach the global limit.

def _get_neighbors(x, y, z):
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]

def _calc_energy(sequence, chain):
    """Compute energy = -number of non-consecutive H-H contacts."""
    if not chain:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _get_neighbors(*p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _pivot_move(chain, pivot_idx, rng):
    """
    Perform a pivot move: rotate the tail of the chain (from pivot_idx to end) 
    around the bond connecting pivot_idx-1 and pivot_idx.
    Returns the new chain if valid, None otherwise.
    """
    n = len(chain)
    if n < 3 or pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # The tail starts at pivot_idx
    tail = chain[pivot_idx:]
    # The axis is the vector from chain[pivot_idx-1] to chain[pivot_idx]
    p0 = chain[pivot_idx - 1]
    p1 = chain[pivot_idx]
    
    # The tail is a list of relative coordinates from p1
    # Actually, it's easier to rotate the absolute coordinates around the axis p0->p1
    # But standard pivot moves rotate the tail around the bond.
    # Let's define the tail relative to p1.
    # The first point of the tail is p1 itself? No, the pivot is at pivot_idx.
    # The bond is between pivot_idx-1 and pivot_idx.
    # The tail consists of residues from pivot_idx to n-1.
    # We rotate the tail around the axis defined by p0 and p1.
    # Wait, the standard definition: rotate the part of the chain after the pivot.
    # If pivot is at index k, we rotate k, k+1, ... around the bond (k-1, k).
    
    # Let's extract the tail coordinates relative to p1 (the pivot point).
    # Actually, it's better to rotate around the bond vector.
    # Let v = p1 - p0.
    # We rotate the vector (p_i - p1) for i >= pivot_idx around v.
    
    vx = p1[0] - p0[0]
    vy = p1[1] - p0[1]
    vz = p1[2] - p0[2]
    
    # Normalize v? Not strictly necessary for rotation if we use cross products correctly,
    # but let's ensure it's a unit vector for standard rotation formulas.
    norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    if norm == 0:
        return None
    vx /= norm
    vy /= norm
    vz /= norm
    
    # Choose a rotation angle. 90 degrees is common.
    # Let's try 90, 180, 270 degrees.
    angles = [math.pi/2, math.pi, 3*math.pi/2]
    # Shuffle angles to randomize
    rng.shuffle(angles)
    
    for theta in angles:
        c = math.cos(theta)
        s = math.sin(theta)
        
        new_tail = []
        valid = True
        
        # We need to check self-avoidance against the fixed part (0 to pivot_idx-1)
        # and the new tail.
        # Fixed part: chain[0:pivot_idx]
        # Note: chain[pivot_idx-1] is in the fixed part. chain[pivot_idx] is the first of the tail.
        
        occupied_fixed = set(chain[:pivot_idx])
        # Wait, the pivot point p1 is part of the tail? 
        # Standard pivot: the bond (k-1, k) is the axis. The tail k...n-1 is rotated.
        # So p1 (index pivot_idx) is the first point of the tail.
        # Its position relative to the axis origin (let's say p0) is v.
        # Rotating v around v gives v. So p1 stays at p1?
        # No, the axis passes through p0 and p1.
        # If we rotate the tail around the axis, p1 is on the axis, so it doesn't move.
        # p2, p3... move.
        
        # Let's verify:
        # p1 is on the axis. Rotating p1 around the axis leaves it unchanged.
        # So new_chain[pivot_idx] = p1.
        
        # For i > pivot_idx:
        # pos_i = chain[i]
        # rel_i = pos_i - p1
        # Rotate rel_i around v by theta.
        
        # Rotation of vector u around unit vector v by theta:
        # u_rot = u * c + (v x u) * s + v * (v . u) * (1 - c)
        
        new_tail.append(p1) # p1 is fixed
        
        # We need to check if the new positions collide with fixed part or each other.
        # Let's build the new tail list and check.
        
        new_positions = [p1]
        new_occupied = set(new_positions)
        
        for i in range(pivot_idx + 1, n):
            px, py, pz = chain[i]
            # rel vector from p1
            rx = px - p1[0]
            ry = py - p1[1]
            rz = pz - p1[2]
            
            # Cross product v x u
            cx = vy * rz - vz * ry
            cy = vz * rx - vx * rz
            cz = vx * ry - vy * rx
            
            # Dot product v . u
            dot = vx * rx + vy * ry + vz * rz
            
            # u_rot
            rx_rot = rx * c + cx * s + vx * dot * (1 - c)
            ry_rot = ry * c + cy * s + vy * dot * (1 - c)
            rz_rot = rz * c + cz * s + vz * dot * (1 - c)
            
            # New absolute position
            new_px = p1[0] + rx_rot
            new_py = p1[1] + ry_rot
            new_pz = p1[2] + rz_rot
            
            # Check if it's an integer (should be for 90, 180, 270)
            # Due to floating point, check closeness to integer
            if not (abs(new_px - round(new_px)) < 1e-9 and 
                    abs(new_py - round(new_py)) < 1e-9 and 
                    abs(new_pz - round(new_pz)) < 1e-9):
                valid = False
                break
                
            new_px = int(round(new_px))
            new_py = int(round(new_py))
            new_pz = int(round(new_pz))
            
            new_pos = (new_px, new_py, new_pz)
            
            # Check collision with fixed part
            if new_pos in occupied_fixed:
                valid = False
                break
            
            # Check collision with already placed new tail
            if new_pos in new_occupied:
                valid = False
                break
                
            new_positions.append(new_pos)
            new_occupied.add(new_pos)
            
        if not valid:
            continue
            
        # Construct new chain
        new_chain = chain[:pivot_idx] + new_positions
        return new_chain
        
    return None

def _regrow_suffix(sequence, prefix, start_idx, rng, temp):
    """
    Regrow the chain from start_idx to the end.
    Returns the full chain if successful, None if stuck.
    """
    n = len(sequence)
    if start_idx >= n:
        return list(prefix)
    
    current_chain = list(prefix)
    occupied = {p: i for i, p in enumerate(current_chain)}
    last_pos = current_chain[-1]
    
    for i in range(start_idx, n):
        neighbors = _get_neighbors(*last_pos)
        valid_moves = []
        
        for move in neighbors:
            if move in occupied:
                continue
            
            # Calculate score for this move
            score = 0.0
            if sequence[i] == 'H':
                # Count H-H contacts with existing chain
                for q in _get_neighbors(*move):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1.0
            
            # Count free space (heuristic to avoid trapping)
            free_space = 0
            for q in _get_neighbors(*move):
                if q not in occupied:
                    free_space += 1
            score += free_space * 0.1
            
            # Add noise
            score += rng.random() * temp
            
            valid_moves.append((score, move))
        
        if not valid_moves:
            return None
        
        # Pick the best move
        best_move = max(valid_moves, key=lambda x: x[0])[1]
        
        current_chain.append(best_move)
        occupied[best_move] = i
        last_pos = best_move
        
    return current_chain

def _initial_chain(sequence, rng):
    """Create an initial chain using greedy regrowth from start."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        neighbors = _get_neighbors(*chain[-1])
        valid_moves = []
        for move in neighbors:
            if move in occupied:
                continue
            valid_moves.append(move)
        
        if not valid_moves:
            return None
        
        # Pick random move for diversity
        move = rng.choice(valid_moves)
        chain.append(move)
        occupied[move] = i
        
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Pivot and Regrowth moves.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension] * n
    
    # Time management
    # Global deadline is 330s. We started at _START_TIME.
    # We want to leave 5s margin.
    global_deadline = _START_TIME + 325.0
    
    # If we are already close to the deadline, return a quick result
    if time.time() > global_deadline - 10.0:
        # Try to generate a valid chain quickly
        rng = random.Random(42)
        c = _initial_chain(sequence, rng)
        if c:
            return [list(p) for p in c]
        return [[i, 0, 0] for i in range(n)]
    
    # Estimate time available for this call
    # We don't know how many calls are left.
    # Assume 26 sequences.
    # If this is the i-th call, we have (325 - elapsed) / remaining_calls time.
    # But we don't know i or remaining_calls.
    # Let's just use a fixed budget per call, say 10s, but capped by global deadline.
    # If we use 10s per call, 26 calls = 260s. Safe.
    # If the process is shared, we should be more conservative.
    # Let's use 8s per call.
    
    start_time = time.time()
    time_limit = min(8.0, global_deadline - time.time() - 5.0)
    
    if time_limit < 1.0:
        # Not enough time, return quick result
        rng = random.Random(42)
        c = _initial_chain(sequence, rng)
        if c:
            return [list(p) for p in c]
        return [[i, 0, 0] for i in range(n)]
        
    rng = random.Random(42)
    
    # Generate multiple initial chains
    initial_chains = []
    for _ in range(5):
        c = _initial_chain(sequence, rng)
        if c:
            initial_chains.append(c)
            
    if not initial_chains:
        return [[i, 0, 0] for i in range(n)]
        
    best_chain = initial_chains[0]
    best_energy = _calc_energy(sequence, best_chain)
    
    # Simulated Annealing for each initial chain
    time_per_chain = time_limit / len(initial_chains)
    
    for chain in initial_chains:
        current_chain = list(chain)
        current_energy = _calc_energy(sequence, current_chain)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() - start_time < start_time + time_limit - 0.5:
            if time.time() > start_time + time_limit - 0.1:
                break
            if time.time() > global_deadline - 5.0:
                break
                
            # Randomly choose between Pivot and Regrowth
            # Pivot is good for local rearrangement, Regrowth for global.
            # Let's try Pivot 50%, Regrowth 50%.
            
            if rng.random() < 0.5:
                # Pivot Move
                if n > 4:
                    # Pivot index between 2 and n-3
                    pivot_idx = rng.randint(2, n - 3)
                    new_chain = _pivot_move(current_chain, pivot_idx, rng)
                else:
                    new_chain = None
            else:
                # Regrowth Move
                if n > 4:
                    k = rng.randint(n // 3, n - 2)
                else:
                    k = n - 2
                prefix = current_chain[:k+1]
                new_chain = _regrow_suffix(sequence, prefix, k + 1, rng, T)
            
            if new_chain is None:
                T *= cooling
                continue
                
            new_energy = _calc_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
            # Check time frequently
            if time.time() - start_time > start_time + time_limit - 1.0:
                break
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
