"""HP lattice protein folding using Metropolis Monte Carlo with pivot moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts (non-consecutive)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            j = where.get((nx, ny))
            if j is not None and abs(i - j) > 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _initial_compact_walk(n, rng):
    """Create a compact initial self-avoiding walk using a spiral-like pattern."""
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Directions: Right, Down, Left, Up
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    dir_idx = 0
    step_len = 1
    
    while len(chain) < n:
        dx, dy = directions[dir_idx]
        # Try to move step_len times in current direction
        for _ in range(step_len):
            if len(chain) >= n:
                break
            nx, ny = chain[-1][0] + dx, chain[-1][1] + dy
            if (nx, ny) not in occupied:
                chain.append((nx, ny))
                occupied.add((nx, ny))
            else:
                # If blocked, try next direction immediately (break inner loop logic slightly)
                # Actually, standard spiral logic: if blocked, change direction
                break
        
        if len(chain) >= n:
            break
            
        dir_idx = (dir_idx + 1) % 4
        if dir_idx % 2 == 0:
            step_len += 1
            
    # If spiral got stuck, fill randomly
    while len(chain) < n:
        options = []
        x, y = chain[-1]
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                options.append((nx, ny))
        if not options:
            break
        next_pt = rng.choice(options)
        chain.append(next_pt)
        occupied.add(next_pt)
        
    return chain


def _mc_fold(sequence, n, rng, time_budget):
    """Metropolis Monte Carlo with pivot moves."""
    start_time = time.time()
    
    # Initial conformation
    chain = _initial_compact_walk(n, rng)
    current_energy = _compute_energy(sequence, chain)
    best_chain = [list(p) for p in chain]
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.5
    # Estimate steps based on time budget. ~1ms per step? 
    # Let's just run until time runs out.
    
    T = T_start
    cooling_rate = 0.999
    
    # Pre-compute H indices for faster energy update? 
    # For now, full re-calc is O(N). Pivot move is O(N) for collision check.
    # Optimization: Only check collisions for moved part.
    
    occupied = set(tuple(p) for p in chain)
    
    while T > T_end and (time.time() - start_time) < time_budget:
        # Choose move: Pivot or End Rotation
        if n < 4:
            # Small chains, just try random valid moves if possible, but pivot is hard
            break
            
        # 1. Pivot Move: Rotate sub-chain [k..n-1] around chain[k-1]
        # k is in [1, n-1]
        # Probability of picking k: uniform? Or biased towards smaller k (larger tails)?
        # Uniform is fine.
        k = rng.randint(1, n - 1)
        
        # Anchor is chain[k-1]
        ax, ay = chain[k-1]
        
        # The vector from anchor to chain[k] is v = (vx, vy)
        vx, vy = chain[k][0] - ax, chain[k][1] - ay
        
        # Possible rotations of v by 90, 180, 270 degrees
        # (vx, vy) -> (-vy, vx) (90 deg CCW)
        # (vx, vy) -> (-vx, -vy) (180 deg)
        # (vx, vy) -> (vy, -vx) (270 deg CCW / 90 deg CW)
        
        new_vx, new_vy = None, None
        roll = rng.randint(1, 3)
        if roll == 1:
            new_vx, new_vy = -vy, vx
        elif roll == 2:
            new_vx, new_vy = -vx, -vy
        else:
            new_vx, new_vy = vy, -vx
            
        # New position for chain[k]
        nk_x, nk_y = ax + new_vx, ay + new_vy
        
        # Check if new position for chain[k] is free
        if (nk_x, nk_y) in occupied:
            continue
            
        # Now we need to rotate the rest of the chain [k+1..n-1] relative to chain[k]
        # We must verify that the entire rotated tail does not collide with the fixed head [0..k-1]
        # and does not self-collide.
        
        # Store old positions of tail
        old_tail = [tuple(p) for p in chain[k:]]
        
        # Calculate new positions
        new_tail = []
        new_tail.append((nk_x, nk_y))
        
        # The relative vector from chain[k] to chain[k+1] is r1 = chain[k+1] - chain[k]
        # We rotate r1 by the same angle.
        # It's easier to rotate the entire vector from anchor to each point?
        # No, rotation is around chain[k-1].
        # So for any point P in the tail, the new point P' is Anchor + R(P - Anchor).
        
        # Let's apply rotation to all points in tail [k..n-1]
        valid = True
        for i in range(k, n):
            px, py = chain[i]
            # Vector from anchor
            dx, dy = px - ax, py - ay
            # Rotate
            if roll == 1:
                ndx, ndy = -dy, dx
            elif roll == 2:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dy, -dx
            nx, ny = ax + ndx, ay + ndy
            new_tail.append((nx, ny))
            
            # Check collision with occupied set (which includes head and old tail)
            # We must exclude the old tail positions from the collision check for the new tail
            # But we are building new_tail, so we need to check against:
            # 1. Head [0..k-1]
            # 2. Other new_tail points (self-collision)
            # 3. Old tail positions that are NOT being moved? No, all tail is moved.
            
            # So we need a set of "fixed" positions: Head [0..k-1]
            # And we check new_tail against Fixed and previous new_tail points.
            
            # Optimization: Pre-compute head set? 
            # For now, just check against 'occupied' but we need to be careful.
            # 'occupied' contains Head + Old Tail.
            # We remove Old Tail from consideration?
            # Let's just create a set of fixed points: head_set = set(chain[:k])
            # And check new points against head_set and already added new_tail points.
            
            # This is O(N) per step if we do it naively.
            # Let's refine the check.
            
            if i == k:
                # Already checked (nk_x, nk_y) against occupied. 
                # But occupied includes old tail. So it's free relative to old config.
                # It might collide with head? Yes, (nk_x, nk_y) could be in head.
                # Wait, if (nk_x, nk_y) is in head, it's in occupied. So the first check `if (nk_x, nk_y) in occupied` handles it.
                pass
            else:
                # Check against head
                # We don't have head_set readily available without O(N) scan or pre-calc.
                # Let's just check against 'occupied' but we must ignore the old tail positions.
                # This is tricky.
                
                # Alternative: 
                # 1. Remove old tail from occupied.
                # 2. Check new tail points against occupied (which is now just head).
                # 3. Check new tail points against each other.
                
                # Let's do that.
                pass
            
        # Let's restart the logic for clarity and correctness, even if slightly slower, to ensure validity.
        # 1. Identify Head [0..k-1] and Tail [k..n-1].
        # 2. Create a set of fixed points: head_set = set(tuple(p) for p in chain[:k])
        # 3. Generate new_tail.
        # 4. Check each new point in new_tail against head_set.
        # 5. Check each new point in new_tail against other new points in new_tail.
        
        head_set = set(tuple(p) for p in chain[:k])
        
        new_tail = []
        valid = True
        seen_in_new_tail = set()
        
        for i in range(k, n):
            px, py = chain[i]
            dx, dy = px - ax, py - ay
            if roll == 1:
                ndx, ndy = -dy, dx
            elif roll == 2:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dy, -dx
            nx, ny = ax + ndx, ay + ndy
            pt = (nx, ny)
            
            # Check collision with head
            if pt in head_set:
                valid = False
                break
            # Check collision with other new tail points
            if pt in seen_in_new_tail:
                valid = False
                break
            
            seen_in_new_tail.add(pt)
            new_tail.append(pt)
            
        if not valid:
            continue
            
        # If valid, accept/reject based on energy
        # Update chain
        # We need to update occupied set for next iterations?
        # It's better to maintain 'occupied' efficiently or rebuild it.
        # Rebuilding is O(N). We are doing O(N) work for the move anyway.
        
        new_chain = chain[:k] + [list(p) for p in new_tail]
        
        # Compute new energy
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            # Update occupied set
            occupied = set(tuple(p) for p in chain)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in chain]
                
        T *= cooling_rate
        
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using Metropolis Monte Carlo with multiple restarts."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    total_budget = 10.0  # seconds per sequence
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    # Multi-start
    n_restarts = 5
    time_per_restart = total_budget / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 0.5:
            break
        
        rng = random.Random(42 + len(sequence) * 100 + restart * 7)
        time_budget = min(time_per_restart, remaining)
        
        chain, energy = _mc_fold(sequence, n, rng, time_budget)
        
        if chain is not None:
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
