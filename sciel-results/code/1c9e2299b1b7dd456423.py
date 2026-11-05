"""HP lattice protein folding on the square lattice using pivot local search."""
# EVOLVE-BLOCK-START
import random
import time

def _energy(sequence, chain):
    """Calculate the energy of the conformation.
    Energy is -1 for each H-H contact (non-adjacent in sequence, adjacent in space).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                j = pos_to_idx.get(neighbor)
                if j is not None:
                    # Must not be adjacent in sequence
                    if abs(i - j) > 1 and sequence[j] == 'H':
                        # Count each contact once. 
                        # Since we iterate all i, we will count (i,j) and (j,i).
                        # We can just count if i < j to avoid double counting, 
                        # or divide by 2 at the end.
                        # Let's count if i < j
                        if i < j:
                            contacts += 1
    return -contacts

def _is_valid_chain(chain):
    """Check if chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dx = abs(p[0] - prev[0])
            dy = abs(p[1] - prev[1])
            if dx + dy != 1:
                return False
    return True

def _rotate_around_center(point, center, angle):
    """Rotate point around center by angle (90, 180, 270 degrees)."""
    x, y = point
    cx, cy = center
    dx = x - cx
    dy = y - cy
    
    if angle == 90:
        return (cx - dy, cy + dx)
    elif angle == 180:
        return (cx - dx, cy - dy)
    elif angle == 270:
        return (cx + dy, cy - dx)
    else:
        return (x, y)

def _generate_initial(sequence, rng):
    """Generate an initial self-avoiding walk using greedy/randomized approach."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        # Possible moves: Up, Down, Left, Right
        # Avoid immediate backtracking if possible? 
        # Greedy: prefer moves that create H-H contacts
        candidates = []
        
        # Order moves: try to go in a spiral or random
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rng.shuffle(moves)
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if pos in occupied:
                continue
            
            # Score this move
            score = 0
            if sequence[i] == 'H':
                # Check if this move creates new H-H contacts with previous H's
                # Neighbors of pos
                for px, py in [(nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)]:
                    p = (px, py)
                    if p in occupied:
                        idx = occupied[p]
                        if idx < i - 1 and sequence[idx] == 'H':
                            score += 1
            # Add small random noise
            score += rng.random() * 0.1
            candidates.append((score, pos))
            
        if not candidates:
            # Dead end. This simple greedy might fail for long sequences.
            # For robustness, we might need a more complex growth or just accept failure and retry.
            return None
            
        # Pick best candidate
        best_score, best_pos = max(candidates)
        chain.append(best_pos)
        occupied[best_pos] = i
        
    return chain

def _local_search(sequence, initial_chain, time_limit, rng):
    """Improve the conformation using pivot moves."""
    if not initial_chain:
        return None, 0
        
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    n = len(current_chain)
    if n < 3:
        return best_chain, best_energy
    
    start_time = time.time()
    stall_count = 0
    max_stall = 5000 # If no improvement for this many moves, try a kick or restart
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random pivot index
        # Avoid pivots at the very end (no tail) or very beginning (small tail)
        # Pivot i means tail is i+1 to n-1
        # Tail length = n - 1 - i
        # We want tail length >= 2 to make rotation meaningful
        # So i <= n - 3
        if n <= 3:
            break
            
        pivot_idx = rng.randint(1, n - 3)
        angle = rng.choice([90, 180, 270])
        
        # Perform pivot
        center = current_chain[pivot_idx]
        head = current_chain[:pivot_idx + 1]
        tail = current_chain[pivot_idx + 1:]
        
        new_tail = [_rotate_around_center(p, center, angle) for p in tail]
        new_chain = head + new_tail
        
        # Check validity
        # 1. Self-avoidance
        # 2. Unit steps (guaranteed by rotation if original was valid, 
        #    except potentially at the pivot junction? 
        #    The step from head[-1] to new_tail[0] must be unit.
        #    Rotation preserves distances, so dist(center, tail[0]) is preserved.
        #    So unit step at junction is guaranteed.)
        
        # Check self-avoidance
        # We can do this efficiently by checking if new_tail intersects head or itself.
        # Since N is small, simple set check is fine.
        # But we can be smarter:
        # head_set = set(head)
        # if any(p in head_set for p in new_tail): invalid
        # if len(set(new_tail)) != len(new_tail): invalid
        
        # Let's just use the set check on the whole chain for simplicity and safety, 
        # as N <= 100.
        if len(set(new_chain)) != n:
            continue
            
        # Calculate new energy
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            stall_count = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Try a "kick": accept a worse move to escape local minimum
                # Or just break and rely on restarts
                # Let's accept a worse move with small probability
                if rng.random() < 0.01:
                    current_chain = new_chain
                    current_energy = new_energy
                    stall_count = 0
                else:
                    # Reset stall count to avoid infinite loop on same local min
                    # or just break
                    stall_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Time budget for this sequence
    # Total budget 120s for 10 sequences -> 12s each
    # We allocate 10s to be safe.
    time_limit = 10.0
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Try a few initial conformations
    num_initials = 5
    
    for k in range(num_initials):
        # Reduce time limit for subsequent trials
        remaining_time = time_limit / (num_initials - k + 1)
        
        # Generate initial
        # Use different seeds or strategies
        init_rng = random.Random(12345 + n + k)
        initial = _generate_initial(sequence, init_rng)
        
        if initial is None:
            # If greedy failed, try a straight line (always valid)
            initial = [(i, 0) for i in range(n)]
            
        # Refine
        refined_chain, refined_energy = _local_search(sequence, initial, remaining_time, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
        # Check global time
        # We don't have a global timer here, but the caller (evaluator) kills us.
        # We should probably check if we've spent too much time in total.
        # Since fold is called once per sequence, and we don't know when it's called,
        # we assume the 120s is total.
        # We can't easily track total time across calls if state isn't preserved.
        # However, the prompt says "fold is called once per benchmark sequence".
        # We can use a global variable to track time.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        
    # Convert to list of lists
    return [list(p) for p in best_chain]

# Global time tracking
_start_time = time.time()
_TOTAL_BUDGET = 110.0 # Leave 10s margin

def _check_time():
    return time.time() - _start_time > _TOTAL_BUDGET

# Modify fold to respect global budget
def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    if _check_time():
        # Out of time, return straight line
        return [[i, 0] for i in range(n)]
    
    # Allocate time based on remaining budget
    # Simple heuristic: if it's the first sequence, take more time.
    # But we don't know the index.
    # Let's just use a fixed per-call limit that sums to < 120s.
    # If we call fold 10 times, and each takes 11s, total 110s.
    time_limit = 11.0
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Try a few initial conformations
    num_initials = 3 # Reduce to allow more time per refinement
    
    for k in range(num_initials):
        if _check_time():
            break
            
        remaining_time = time_limit / (num_initials - k + 1)
        
        # Generate initial
        init_rng = random.Random(12345 + n + k)
        initial = _generate_initial(sequence, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine
        refined_chain, refined_energy = _local_search(sequence, initial, remaining_time, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
