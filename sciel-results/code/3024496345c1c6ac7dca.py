import time
import random
import math
import sys

# Global time management
START_TIME = time.time()
TIME_LIMIT = 235.0  # Leave a 5s margin for process exit/overhead

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
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

def _grow_random(sequence, rng):
    """
    Generate a random self-avoiding walk.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _grow_spiral(sequence, rng):
    """
    Generate a spiral-like initial structure which is compact.
    """
    N = len(sequence)
    if N == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    dx, dy, dz = 1, 0, 0
    
    # Simple spiral in XY plane, then Z
    # This is a heuristic to get a compact start
    # A better approach: use a known good template if N matches, else random
    # For now, let's try a "box filling" strategy
    # Estimate box size
    L = int(math.ceil(N ** (1/3)))
    
    # Fill a box of size LxLxL
    # This might fail if N is not a perfect cube or if the path gets stuck
    # A safer compact start:
    # Go right L, up 1, left L, up 1...
    
    current = (0, 0, 0)
    for i in range(1, N):
        # Try to stay in a local box
        # Prefer moves that keep x,y,z small
        best_site = None
        min_dist = float('inf')
        options = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            # Heuristic: prefer sites closer to origin to stay compact
            dist = sum(abs(c) for c in site)
            if dist < min_dist:
                min_dist = dist
                best_site = site
            options.append(site)
        
        if not options:
            return None
            
        # Add some randomness to avoid getting stuck in a deterministic trap
        if rng.random() < 0.2 and len(options) > 1:
             best_site = rng.choice(options)
        else:
             if best_site is None:
                 best_site = options[0]
        
        chain.append(best_site)
        occupied[best_site] = i
        current = best_site
        
    return chain

def _tail_regrowth(sequence, chain, cut_idx, rng, T):
    """
    Regrow the tail of the chain from cut_idx.
    """
    N = len(sequence)
    if cut_idx >= N:
        return chain, 0
    
    head = chain[:cut_idx]
    head_map = {p: idx for idx, p in enumerate(head)}
    occupied = set(head_map.keys())
    
    new_tail = []
    tail_map = {}
    current_pos = head[-1]
    
    valid = True
    # We use a greedy-with-noise approach for the regrowth itself to ensure speed,
    # but the acceptance of the *whole* new chain is done via Metropolis in the main loop.
    # Actually, to make it more effective, we can use a local SA for the regrowth?
    # No, that's too slow. Greedy regrowth with random restarts of the cut point is standard.
    
    for k in range(cut_idx, N):
        options = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    if q in head_map:
                        idx_h = head_map[q]
                        if sequence[idx_h] == "H":
                            gain += 1
                    elif q in tail_map:
                        idx_t = tail_map[q]
                        if sequence[idx_t] == "H":
                            gain += 1
                            
            # Score: maximize gain, add noise proportional to T
            noise = rng.gauss(0, T)
            score = gain * 5.0 + noise
            options.append((score, site))
            
        if not options:
            valid = False
            break
            
        # Sort by score descending and pick the best
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        
        new_tail.append(best_site)
        tail_map[best_site] = k
        occupied.add(best_site)
        current_pos = best_site
        
    if not valid:
        return chain, 0
        
    new_chain = head + new_tail
    return new_chain, 0

def _pivot_move(sequence, chain, rng):
    """
    Perform a pivot move: rotate a sub-chain around a bond.
    """
    N = len(sequence)
    if N < 4:
        return chain
    
    # Pick a pivot point i (1 to N-2)
    i = rng.randint(1, N - 2)
    
    # The pivot is at bond (i-1, i)
    # We rotate the tail (i to N-1) around the bond axis.
    # In 3D, this is complex. A simpler approximation is to reflect the tail
    # or rotate it by 90/180/270 degrees around the bond vector.
    
    p0 = chain[i-1]
    p1 = chain[i]
    
    # Vector of the bond
    v = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
    
    # Find a perpendicular vector to define rotation plane
    # This is getting complicated. Let's use a simpler "Punch" move or "Slide".
    # Slide: Move a sub-chain (i to j) to a new position if valid.
    
    # Let's implement a simple "Punch" move:
    # Remove segment [i, j] and reinsert it at [k, k+1]
    
    i = rng.randint(1, N - 2)
    length = rng.randint(1, min(10, N - i - 1))
    j = i + length - 1
    
    # Remove segment
    segment = chain[i:j+1]
    new_chain = chain[:i] + chain[j+1:]
    
    # Try to reinsert at a random location
    k = rng.randint(1, len(new_chain) - 1)
    
    # Check if insertion is valid
    # The segment must connect to new_chain[k-1] and new_chain[k]
    # And not collide with the rest of the chain
    
    # This is a bit expensive to check fully.
    # Let's stick to Tail Regrowth and simple local moves.
    
    # Local Move: Slide a single residue? No, too slow.
    # Let's use "Pivot" by reflecting the tail across the plane defined by the bond and a random vector?
    
    # Alternative: Use the "Pivot" move as defined in HP literature:
    # Pick a cut. Rotate the tail by 90, 180, or 270 degrees around the cut bond.
    
    # Let's implement a robust 90-degree rotation around the bond axis.
    
    # Axis is v.
    # We need to rotate the vector (p_k - p_i) for all k > i.
    
    # This requires vector math.
    # Let's try a simpler move: "Flip" the tail if possible?
    
    # Actually, the most effective simple move after Tail Regrowth is "Punch".
    # Let's implement a fast Punch.
    
    i = rng.randint(1, N - 3)
    L = rng.randint(1, 5) # Small punches
    j = i + L - 1
    
    if j >= N:
        return chain
        
    segment = chain[i:j+1]
    
    # Remove segment
    head = chain[:i]
    tail = chain[j+1:]
    
    # Try to insert segment at a random position in the head+tail
    # We need to find a gap (a, b) in the remaining chain such that:
    # 1. a and b are adjacent in the new chain
    # 2. segment[0] is adjacent to a
    # 3. segment[-1] is adjacent to b
    # 4. No collisions
    
    # This is a search.
    # To keep it fast, we only try a few random insertion points.
    
    new_chain_rest = head + tail
    occupied_rest = {p: idx for idx, p in enumerate(new_chain_rest)}
    
    # Find all gaps in new_chain_rest
    # A gap is between new_chain_rest[k-1] and new_chain_rest[k]
    
    # We will try to insert at a random gap
    if len(new_chain_rest) < 2:
        return chain
        
    k = rng.randint(1, len(new_chain_rest) - 1)
    
    p_a = new_chain_rest[k-1]
    p_b = new_chain_rest[k]
    
    # Check adjacency for segment ends
    # dist(segment[0], p_a) == 1 and dist(segment[-1], p_b) == 1
    
    def dist(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        
    if dist(segment[0], p_a) != 1 or dist(segment[-1], p_b) != 1:
        return chain
        
    # Check collisions
    # The segment occupies L points.
    # We need to ensure none of these points are in occupied_rest
    # Also, the internal bonds of the segment are already valid.
    
    # However, the segment might collide with itself? No, it's a rigid body.
    # It might collide with the rest of the chain.
    
    # Translate the segment so that segment[0] is at a position adjacent to p_a
    # Actually, we just check if the existing segment coordinates collide with occupied_rest.
    # Wait, the segment is being moved. Its coordinates are relative to the chain.
    # We need to *translate* the segment to fit into the gap.
    
    # Vector from segment[0] to p_a?
    # No, we need to find a translation vector t such that:
    # segment[0] + t = p_a' (a neighbor of p_a? No, p_a is the neighbor)
    # Actually, the new chain will be:
    # ... p_a, segment[0]+t, segment[1]+t, ..., segment[-1]+t, p_b ...
    # We require dist(p_a, segment[0]+t) == 1 and dist(segment[-1]+t, p_b) == 1.
    
    # This is a constraint satisfaction problem.
    # Let's simplify: Only allow punches where the segment is moved to a position
    # such that segment[0] is adjacent to p_a.
    
    # Candidates for segment[0]+t: Neighbors of p_a.
    # For each candidate c0:
    #   t = c0 - segment[0]
    #   Check if segment[-1]+t is adjacent to p_b.
    #   Check if all (segment[k]+t) are not in occupied_rest.
    
    for c0 in _neighbours(p_a):
        t = (c0[0]-segment[0][0], c0[1]-segment[0][1], c0[2]-segment[0][2])
        
        # Check end connection
        end_point = (segment[-1][0]+t[0], segment[-1][1]+t[1], segment[-1][2]+t[2])
        if dist(end_point, p_b) != 1:
            continue
            
        # Check collisions
        valid = True
        for s in segment:
            new_s = (s[0]+t[0], s[1]+t[1], s[2]+t[2])
            if new_s in occupied_rest:
                valid = False
                break
            # Also check if new_s collides with the *other* part of the segment?
            # No, the segment is rigid and self-avoiding.
            
        if valid:
            # Construct new chain
            new_chain = new_chain_rest[:k-1] + [p_a]
            for s in segment:
                new_chain.append((s[0]+t[0], s[1]+t[1], s[2]+t[2]))
            new_chain.append(p_b)
            new_chain.extend(new_chain_rest[k+1:])
            
            # Wait, the indices in occupied_rest are based on the *old* chain.
            # The new chain is valid if the self-avoidance holds.
            # The check `new_s in occupied_rest` ensures no collision with the *rest* of the chain.
            # The segment itself is self-avoiding.
            # The only other potential collision is if the segment overlaps with itself? No.
            # So this should be valid.
            
            return new_chain
            
    return chain

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    # Check global time
    if time.time() - START_TIME > TIME_LIMIT:
        # If we are out of time, return a simple valid chain (straight line)
        return [[i, 0, 0] for i in range(N)]
    
    # Determine local time budget
    # We don't know exactly how many calls are left, but we can estimate.
    # If this is the first call, we have ~235s.
    # We will just run until the global limit.
    
    rng = random.Random(42 + N) # Different seed for different lengths? Or just 42
    
    # 1. Initialization
    best_chain = None
    best_energy = 0
    
    # Try a few initializations
    for _ in range(5):
        chain = _grow_spiral(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback
        best_chain = _grow_random(sequence, rng)
        if best_chain is None:
            best_chain = [[i, 0, 0] for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # 2. Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Track acceptance rate to adjust cooling
    accepted = 0
    total_moves = 0
    
    while time.time() - START_TIME < TIME_LIMIT:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Tail Regrowth
            # Bias cut point towards the end? Or middle?
            # Middle allows more restructuring.
            # Let's pick uniformly from 1 to N-1
            i = rng.randint(1, N - 1)
            new_chain, _ = _tail_regrowth(sequence, current_chain, i, rng, T)
        else:
            # Punch Move
            new_chain = _pivot_move(sequence, current_chain, rng)
            
        if new_chain is current_chain:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        total_moves += 1
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            accepted += 1
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            pass
            
        # Adjust cooling rate based on acceptance
        # If acceptance is too high, cool faster. If too low, cool slower (or reheat).
        if total_moves > 100:
            acc_rate = accepted / total_moves
            if acc_rate > 0.5:
                cooling_rate = 0.998
            elif acc_rate < 0.1:
                cooling_rate = 0.9995
                # Occasionally reheat
                if rng.random() < 0.01:
                    T = max(T, 1.0)
            else:
                cooling_rate = 0.999
                
            # Reset counters
            accepted = 0
            total_moves = 0

    return [list(p) for p in best_chain]
