# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _sa_fold(sequence, dimension, time_limit):
    """Simulated Annealing for HP folding using segment moves."""
    n = len(sequence)
    if n <= 2:
        if n == 1:
            return [[0] * dimension]
        else:
            return [[0] * dimension, [1, 0] if dimension == 2 else [1] + [0] * (dimension - 1)]
    
    rng = random.Random(12345 + n)
    
    # 1. Greedy Initialization
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            break
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    if len(chain) < n:
        # Fallback
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}
        
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Simulated Annealing
    start_time = time.time()
    max_time = time_limit
    current_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time - 0.5:
            break
        
        # Pick random segment start
        k = rng.randint(0, n - 1)
        dx, dy = rng.choice(dirs)
        
        # Check validity of moving segment k..n-1 by (dx, dy)
        # We must ensure the new positions don't collide with 0..k-1
        # We also must ensure the segment doesn't collide with itself (it won't, relative pos preserved)
        # BUT, we must remove the segment from 'occupied' before checking new positions
        # to avoid false positives from the segment's own old positions.
        
        # Optimization: Only check the "frontier" of the segment?
        # No, the whole segment moves.
        
        # Remove segment from occupied
        # To speed up, we can just check if (x+dx, y+dy) is in occupied, 
        # but we need to ignore the segment's current positions.
        
        # Let's create a set of the segment's current positions to ignore
        # Or, simpler:
        # 1. Store old positions.
        # 2. Check new positions against occupied.
        #    If new_pos in occupied:
        #       If new_pos is in the segment's old positions, it's OK?
        #       No, if new_pos is in the segment's old positions, it means the segment is overlapping itself?
        #       Wait. If I shift a segment, can it overlap itself?
        #       Yes, if the segment folds back.
        #       Example: L-shape. Shift down.
        #       So we MUST check for self-collision.
        
        # Efficient check:
        # 1. Get set of old positions of segment.
        # 2. For each new position, if it is in `occupied`, 
        #    check if it is in the segment's old positions.
        #    If it is in occupied AND NOT in segment old positions -> Collision.
        #    If it is in occupied AND in segment old positions -> OK (it's just moving within its own footprint? No, it's moving TO a spot that was occupied by another part of the segment).
        #    Wait, if I move the segment, the "old" positions become empty.
        #    So if `new_pos` is in `occupied`, and `new_pos` is one of the `old_positions`, 
        #    it means another part of the segment is moving into a spot vacated by this part?
        #    No, `occupied` contains the CURRENT state.
        #    So if `new_pos` is in `occupied`, it is occupied by SOMETHING.
        #    If it is occupied by a part of the segment that is ALSO moving, we need to be careful.
        #    Actually, if we shift the whole segment rigidly, the set of positions occupied by the segment changes from $S$ to $S'$.
        #    We need $S'$ to be disjoint from $Occupied \setminus S$.
        #    And $S'$ must be self-avoiding (which it is, if $S$ was).
        
        # So:
        # 1. $S = \{ chain[i] \text{ for } i \in [k, n) \}$
        # 2. $S' = \{ p + d \text{ for } p \in S \}$
        # 3. Check if any $p' \in S'$ is in $Occupied \setminus S$.
        
        segment_old = [chain[i] for i in range(k, n)]
        segment_old_set = set(segment_old)
        
        valid = True
        for i in range(k, n):
            new_p = (chain[i][0] + dx, chain[i][1] + dy)
            if new_p in occupied:
                # If it's in the segment's old positions, it's fine?
                # No, `occupied` still has the old positions.
                # If `new_p` is in `segment_old_set`, it means the new position is currently occupied by the segment itself.
                # Since the whole segment is moving, the old position will be vacated.
                # So this is a valid move (internal overlap during move is allowed if it resolves).
                # BUT, we must ensure `new_p` is NOT occupied by the prefix (0..k-1).
                if new_p not in segment_old_set:
                    valid = False
                    break
        
        if valid:
            # Update state
            for i in range(k, n):
                old_p = chain[i]
                new_p = (chain[i][0] + dx, chain[i][1] + dy)
                chain[i] = new_p
                del occupied[old_p]
                occupied[new_p] = i
            
            new_energy = _energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / current_temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                for i in range(k, n):
                    new_p = chain[i]
                    old_p = (new_p[0] - dx, new_p[1] - dy)
                    chain[i] = old_p
                    del occupied[new_p]
                    occupied[old_p] = i
                    
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp

    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Best conformation using Simulated Annealing."""
    # 120s total for 10 sequences -> 12s each.
    # Use 10s to be safe.
    time_limit = 10.0
    return _sa_fold(sequence, dimension, time_limit)
# EVOLVE-BLOCK-END
