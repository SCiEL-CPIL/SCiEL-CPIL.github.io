import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0

def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    # Fast energy calculation
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    # Optimization: only check H residues
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _grow(sequence, dimension, rng, bias=1.0):
    # From inspiration, improved
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site) if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            
            # Centroid bias
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = (site[0] - cx)**2 + (site[1] - cy)**2
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
            
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _polish(sequence, chain, time_limit=2.0):
    # Deterministic hill climbing with pivot and slither moves
    # Returns best chain found
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    n = len(current_chain)
    
    while _time_left() > 10 and (time.time() - start_time) < time_limit:
        improved = False
        best_move = None
        best_delta = 0
        
        # Try Pivot Moves
        # To speed up, we can sample pivots or iterate all. 
        # Iterating all is O(N^3) per step. For N=100, 10^6 ops. 
        # If we do 100 steps, 10^8 ops. This is borderline in Python.
        # Let's sample pivots to keep it fast, or limit iterations.
        
        # Actually, the inspiration iterates all pivots. Let's try to be efficient.
        # We will iterate all pivots but break early if we find a big improvement? No, need best.
        
        # Optimization: Precompute set of fixed points for each pivot? No, too expensive.
        
        # Let's use a randomized subset of pivots if N is large, or just accept the cost.
        # For N=100, 100 pivots * 3 angles * 100 points = 30,000 collision checks per iteration.
        # 1000 iterations = 30M checks. This is fine for 2 seconds.
        
        for pivot_idx in range(1, n - 1):
            if _time_left() < 10 or (time.time() - start_time) > time_limit:
                break
                
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start >= n:
                continue
            
            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])
            
            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in segment:
                    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    else:
                        ndx, ndy = -dx, -dy
                    new_p = (pivot[0] + ndx, pivot[1] + ndy)
                    if new_p in fixed_set:
                        valid = False
                        break
                    rotated.append(new_p)
                
                if not valid:
                    continue
                
                # Check self-avoidance of rotated segment? 
                # Rotation preserves internal distances, so internal collisions are preserved.
                # But wait, the original segment was self-avoiding. 
                # Does rotation preserve self-avoidance? Yes, it's an isometry.
                # So we only need to check collision with fixed_set.
                
                new_chain = current_chain[:segment_start] + rotated
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain
        
        # Try Slither Move (move tail)
        if not best_move or best_delta == 0: # Or always try?
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        if best_move is not None:
            current_chain = best_move
            current_energy += best_delta
            improved = True
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # No improving move found. 
            # In a pure hill climb, we stop. 
            # But to allow the outer loop to perturb, we break here.
            break
            
    return best_chain

def _random_perturb(sequence, chain, rng, num_moves=5):
    """Apply random valid moves to escape local minima."""
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    for _ in range(num_moves):
        # Try a random pivot move
        pivot_idx = rng.randint(1, n - 2)
        pivot = current_chain[pivot_idx]
        segment_start = pivot_idx + 1
        segment = current_chain[segment_start:]
        fixed_set = set(current_chain[:pivot_idx + 1])
        
        angles = [90, -90, 180]
        rng.shuffle(angles)
        
        moved = False
        for angle in angles:
            rotated = []
            valid = True
            for p in segment:
                dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                if angle == 90:
                    ndx, ndy = -dy, dx
                elif angle == -90:
                    ndx, ndy = dy, -dx
                else:
                    ndx, ndy = -dx, -dy
                new_p = (pivot[0] + ndx, pivot[1] + ndy)
                if new_p in fixed_set:
                    valid = False
                    break
                rotated.append(new_p)
            
            if valid:
                # Update chain and occupied
                # Remove old segment from occupied
                for p in segment:
                    del occupied[p]
                # Add new segment
                current_chain[segment_start:] = rotated
                for i, p in enumerate(rotated):
                    occupied[p] = segment_start + i
                moved = True
                break
                
        if not moved:
            # Try a single move
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            for q in _neighbours(p):
                if q in occupied and occupied[q] != i:
                    continue
                # Check neighbors
                prev = current_chain[i-1]
                nxt = current_chain[i+1]
                if sum(abs(q[d] - prev[d]) for d in range(2)) != 1:
                    continue
                if sum(abs(q[d] - nxt[d]) for d in range(2)) != 1:
                    continue
                
                # Valid move
                del occupied[p]
                current_chain[i] = q
                occupied[q] = i
                moved = True
                break
                
        if not moved:
            break # Can't perturb further
            
    return current_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 1 # Worst
    
    # Determine budget
    # Total time 195s. 
    # We want to try many restarts.
    # Each restart: Grow (fast) + Polish (up to 2s) + Perturb/Polish (optional).
    
    # Strategy: 
    # 1. Do a few fast restarts with short polish.
    # 2. Do a few slow restarts with long polish.
    # 3. Iterative Perturbation on the best chain found.
    
    # Let's define a list of (polish_time, perturb_rounds)
    # For N=20, we can do many. For N=100, fewer.
    
    if n <= 30:
        restarts = 50
        polish_time = 1.0
        perturb_rounds = 2
    elif n <= 60:
        restarts = 20
        polish_time = 2.0
        perturb_rounds = 1
    else:
        restarts = 10
        polish_time = 3.0
        perturb_rounds = 1
        
    # Phase 1: Restart and Polish
    for r in range(restarts):
        if _time_left() < 5:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        # Polish
        polished_chain = _polish(sequence, chain, time_limit=polish_time)
        
        if polished_chain:
            e = _energy(sequence, polished_chain)
            if e < best_energy:
                best_energy = e
                best_chain = polished_chain
        
        # Phase 2: Perturb and Polish (if time and if we have a good chain)
        if best_chain is not None and _time_left() > 10:
            # Perturb the current best
            perturbed = _random_perturb(sequence, best_chain, rng, num_moves=5)
            polished_chain = _polish(sequence, perturbed, time_limit=polish_time)
            
            if polished_chain:
                e = _energy(sequence, polished_chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = polished_chain
                    
    # Final Polish with remaining time
    if best_chain is not None and _time_left() > 5:
        # Do one more long polish on the best
        final_chain = _polish(sequence, best_chain, time_limit=_time_left() - 5)
        if final_chain:
            e = _energy(sequence, final_chain)
            if e < best_energy:
                best_energy = e
                best_chain = final_chain
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
