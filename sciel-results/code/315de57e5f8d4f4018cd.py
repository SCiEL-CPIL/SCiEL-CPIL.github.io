import time
import random
import math

# Global deadline guard
_T0 = time.time()
_DEADLINE = 195.0  # 210s limit, 15s safety margin

def _remaining_time():
    return _DEADLINE - (time.time() - _T0)

def _energy(seq, chain):
    """Calculate energy of a conformation."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    # Only check H residues
    for i, c in enumerate(seq):
        if c == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                j = pos_to_idx.get((nx, ny))
                if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_chain(seq, rng):
    """
    Generate a self-avoiding walk using a greedy strategy with randomness.
    Prefers moves that create H-H contacts.
    """
    n = len(seq)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        
        # 4 neighbors
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]
        
        for pos in neighbors:
            if pos in occupied:
                continue
            
            # Calculate score
            score = 0.0
            if seq[i] == 'H':
                # Count potential new H-H contacts
                px, py = pos
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    j = occupied.get((nx, ny))
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 1.0
            
            # Small bonus for keeping options open (avoiding corners)
            free_neighbors = 0
            px, py = pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (px + dx, py + dy) not in occupied:
                    free_neighbors += 1
            score += 0.1 * free_neighbors
            
            # Add randomness
            score += rng.random() * 0.5
            
            options.append((score, pos))
        
        if not options:
            return None # Dead end
        
        # Select best option
        best_score, best_pos = max(options)
        chain.append(best_pos)
        occupied[best_pos] = i
        
    return chain

def _try_pivot_move(chain, i, rng):
    """
    Attempt a pivot move at index i.
    Rotates the segment chain[i:] by 90 degrees.
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i >= n - 1:
        return None
    
    # The pivot point is chain[i]
    px, py = chain[i]
    
    # Segment to rotate: chain[i+1:]
    # We rotate around chain[i]
    # Old relative coords: (x - px, y - py)
    # New relative coords: (-(y - py), x - px) for 90 deg CCW
    
    new_segment = []
    # We need to check if the new positions collide with the prefix chain[:i]
    # The prefix is chain[0:i]
    prefix_set = set(chain[:i])
    
    # Also need to ensure the new segment doesn't collide with itself?
    # Rotation is an isometry, so internal self-avoidance is preserved.
    # We only need to check against the prefix.
    
    # However, the connection between chain[i-1] and chain[i] is fixed.
    # The connection between chain[i] and chain[i+1] changes.
    # We must ensure the new chain[i+1] is adjacent to chain[i].
    # By definition of rotation around chain[i], the distance is preserved,
    # but we need to ensure it's a valid lattice step (Manhattan distance 1).
    # Original chain[i+1] was at distance 1 from chain[i].
    # Rotated position is also at distance 1.
    # So connectivity is preserved.
    
    # Check collisions
    for k in range(i + 1, n):
        x, y = chain[k]
        # New pos
        nx = px - (y - py)
        ny = py + (x - px)
        
        if (nx, ny) in prefix_set:
            return None
        
        new_segment.append((nx, ny))
    
    # Construct new chain
    # chain[:i+1] + new_segment
    # Note: chain[i] is the pivot, it stays in place.
    # So new chain is chain[0:i+1] + new_segment
    # Wait, chain[i] is included in the prefix for collision check?
    # prefix_set = set(chain[:i]) -> indices 0 to i-1.
    # The pivot is chain[i].
    # The new segment starts at i+1.
    # So we check new_segment against chain[0:i].
    # What about chain[i]?
    # The new first element of new_segment is at distance 1 from chain[i].
    # It cannot be chain[i] itself.
    # So checking against chain[0:i] is sufficient for external collisions.
    
    return chain[:i+1] + new_segment

def _local_search(seq, chain, rng, time_limit):
    """
    Refine a chain using local moves (Pivots).
    Uses a simple hill-climbing with occasional random kicks.
    """
    current_chain = chain
    current_energy = _energy(seq, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    n = len(seq)
    # Simple temperature schedule
    # We will just do greedy moves with some randomness to escape local minima
    
    steps = 0
    max_steps = 50000 # Cap to prevent infinite loops if time check is slow
    start_time = time.time()
    
    while _remaining_time() > 0.1 and (time.time() - start_time) < time_limit:
        # Random pivot index
        i = rng.randint(1, n - 2)
        
        new_chain = _try_pivot_move(current_chain, i, rng)
        if new_chain is None:
            # Try a different move or random walk step?
            # For now, just continue
            continue
        
        new_energy = _energy(seq, new_chain)
        
        # Accept if better
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = current_chain
                best_energy = current_energy
        else:
            # Accept with small probability to escape local minima
            # Probability = exp(-delta / T)
            # T decreases over time
            # Simple heuristic: accept 10% of worse moves
            if rng.random() < 0.1:
                current_chain = new_chain
                current_energy = new_energy
        
        steps += 1
        if steps > max_steps:
            break

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Main entry point.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Allocate time based on sequence length
    # Total available time
    total_time = _remaining_time()
    if total_time <= 0:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # We have 10 sequences. We don't know which one is being called,
    # but we assume the evaluator calls them sequentially.
    # We should use a fraction of the *total* remaining time?
    # No, we should use a fraction of the *allocated* time for this sequence.
    # Since we don't know the order, we assume a fair split of the 195s budget.
    # But if we are called for the first sequence, we have 195s.
    # If we are called for the last, we have less.
    # The prompt says "fold is called once per benchmark sequence".
    # It does not say we can coordinate between calls.
    # So we must assume we have to finish within a reasonable fraction of the total budget
    # OR we can use all remaining time if we are the last one?
    # "120 s CPU budget for all sequences together" (from task description)
    # "evaluator KILLS the program after 210s wall-clock"
    # If we use 100s on the first sequence, we have 115s left for 9 others.
    # To be safe, we should limit each sequence to ~15-20s.
    # Let's aim for 15s per sequence.
    
    time_budget = 15.0
    if _remaining_time() < time_budget:
        time_budget = _remaining_time() * 0.9 # Use most of what's left if tight
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0 # Maximize energy (minimize -energy)
    
    # Phase 1: Greedy Growth
    # Number of restarts depends on time budget
    # Estimate time per restart: ~1ms for growth + refinement
    # Let's do a mix of fast restarts and deep refinements.
    
    num_fast_restarts = 50
    num_deep_restarts = 5
    
    # Fast restarts
    for _ in range(num_fast_restarts):
        if _remaining_time() - time_budget < 0.5:
            break
        chain = _grow_chain(sequence, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Deep refinements
    for _ in range(num_deep_restarts):
        if _remaining_time() - time_budget < 1.0:
            break
        chain = _grow_chain(sequence, rng)
        if chain is None:
            continue
        # Refine this specific chain
        # Allocate 2s per deep refinement
        refined_chain, refined_energy = _local_search(sequence, chain, rng, 2.0)
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
            
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
