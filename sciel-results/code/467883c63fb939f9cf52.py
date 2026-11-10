"""Rewritten HP lattice protein folding using simulated annealing with chain-reversal moves."""
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -1 per H-H contact on neighbouring sites (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    # Count each contact only once (i < j)
                    if i < j:
                        contacts += 1
    return -contacts


def _is_valid_walk(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            diff = sum(abs(a - b) for a, b in zip(chain[i], chain[i - 1]))
            if diff != 1:
                return False
    return True


def _initial_spiral(sequence, dimension):
    """Generate an initial self-avoiding walk using a spiral-like pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    # Start with a simple path that grows in a pattern
    # Use a Hilbert-curve-like or simple back-and-forth pattern
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Directions: right, down, left, up, then repeat with larger steps
    # For simplicity, use a snake pattern
    for i in range(1, n):
        last = chain[-1]
        # Try to continue in a pattern: go right, then down, then left, then up, etc.
        # Simple approach: alternate between going right and going down in a block pattern
        block = i // 2
        pos_in_block = i % 2
        
        if block % 2 == 0:
            # Even blocks: go right
            candidate = (last[0] + 1, last[1])
        else:
            # Odd blocks: go down  
            candidate = (last[0], last[1] + 1)
        
        # If candidate is occupied, try other directions
        if candidate in occupied:
            found = False
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                cand = (last[0] + dx, last[1] + dy)
                if cand not in occupied:
                    candidate = cand
                    found = True
                    break
            if not found:
                # Fallback: try to find any free neighbour
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    cand = (last[0] + dx, last[1] + dy)
                    if cand not in occupied:
                        candidate = cand
                        found = True
                        break
                if not found:
                    # Can't continue, return what we have (but this shouldn't happen for small n)
                    return chain
        else:
            found = True
        
        if found:
            occupied[candidate] = i
            chain.append(candidate)
    
    return chain


def _chain_reversal_move(chain, rng):
    """
    Perform a chain-reversal move: pick two positions i < j, reverse the subchain from i to j.
    This is a valid move if the resulting walk is self-avoiding.
    Returns (new_chain, delta_energy) or (None, 0) if move is invalid.
    """
    n = len(chain)
    if n < 3:
        return None, 0
    
    # Pick i and j such that the reversal is valid
    # For efficiency, pick i and j such that j - i >= 2 (at least 3 residues in the segment)
    # Also, the endpoints must not create conflicts
    
    # Try a few random reversals
    for _ in range(10):
        i = rng.randint(0, n - 3)
        j = rng.randint(i + 2, n - 1)
        
        # Check if the reversal is valid
        # The segment chain[i:j+1] is reversed
        # We need to check:
        # 1. The new connection at position i-1 to chain[j] (if i > 0)
        # 2. The new connection at position j+1 to chain[i] (if j < n-1)
        # 3. No self-intersections in the reversed segment
        
        # Quick check: the endpoints of the segment
        if i > 0:
            # chain[i-1] must be adjacent to chain[j]
            diff = sum(abs(a - b) for a, b in zip(chain[i - 1], chain[j]))
            if diff != 1:
                continue
            # chain[i-1] must not be in the reversed segment (except at the endpoints)
            # Actually, chain[i-1] is not in chain[i:j+1], so we just need to check
            # that chain[i-1] != chain[j] (already checked by diff) and that
            # chain[i-1] is not in chain[i+1:j] (the interior of the reversed segment)
            # This is expensive, so we'll do a full check below
        
        if j < n - 1:
            # chain[j+1] must be adjacent to chain[i]
            diff = sum(abs(a - b) for a, b in zip(chain[j + 1], chain[i]))
            if diff != 1:
                continue
        
        # Perform the reversal
        new_chain = chain[:i] + chain[i:j + 1][::-1] + chain[j + 1:]
        
        # Check self-avoidance
        if not _is_valid_walk(new_chain):
            continue
        
        return new_chain, 0  # Energy delta computed separately
    
    return None, 0


def _single_step_move(chain, rng):
    """
    Try to move one residue (not the first or last) to an adjacent empty site.
    Returns (new_chain, delta_energy) or (None, 0) if no valid move found.
    """
    n = len(chain)
    if n < 3:
        return None, 0
    
    # Pick a random interior position
    i = rng.randint(1, n - 2)
    
    # Try each of the 4 neighbours of chain[i]
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        candidate = (chain[i][0] + dx, chain[i][1] + dy)
        
        # Check if candidate is occupied (by any other residue)
        if candidate in set(chain[:i]) | set(chain[i + 1:]):
            continue
        
        # Check if the move maintains unit steps
        # chain[i-1] to candidate must be unit step
        diff1 = sum(abs(a - b) for a, b in zip(chain[i - 1], candidate))
        if diff1 != 1:
            continue
        # candidate to chain[i+1] must be unit step
        diff2 = sum(abs(a - b) for a, b in zip(candidate, chain[i + 1]))
        if diff2 != 1:
            continue
        
        new_chain = chain[:i] + [candidate] + chain[i + 1:]
        
        # Check self-avoidance (should be fine since we checked occupancy)
        if not _is_valid_walk(new_chain):
            continue
        
        return new_chain, 0
    
    return None, 0


def _simulated_annealing(sequence, dimension, initial_chain, time_budget):
    """
    Run simulated annealing starting from initial_chain.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 1:
        return initial_chain
    
    current_chain = list(initial_chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Temperature schedule
    T0 = 1.0
    T_end = 0.01
    # We'll use a geometric cooling schedule
    # Number of steps: we'll run until time budget is exhausted
    # Let's estimate steps per second and set total steps accordingly
    
    rng = random.Random(42 + len(sequence))
    
    step = 0
    T = T0
    
    while time.time() < deadline:
        step += 1
        
        # Cool down
        # Use a simple linear cooling for now
        # We'll adjust based on time
        if step % 100 == 0:
            # Reduce temperature
            T = max(T_end, T * 0.99)
        
        # Choose a move
        if rng.random() < 0.5:
            new_chain, _ = _chain_reversal_move(current_chain, rng)
        else:
            new_chain, _ = _single_step_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence.
    Uses simulated annealing with chain-reversal and single-step moves.
    """
    n = len(sequence)
    if n <= 1:
        return [list(p) for p in [(0,) * dimension]]
    
    # Check for deadline
    total_deadline = 210.0  # Wall-clock limit
    start_time = time.time()
    
    # Generate multiple initial conformations and run SA on each
    # Allocate time budget across runs
    num_runs = min(5, max(1, n // 10))  # 1-5 runs depending on sequence length
    
    # Time per run, with safety margin
    time_per_run = (total_deadline - (time.time() - start_time) - 5.0) / num_runs
    time_per_run = max(1.0, time_per_run)
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is "worst" for comparison
    
    for run in range(num_runs):
        # Check if we're running out of time
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 10.0:
            break
        
        # Generate initial conformation
        if run == 0:
            initial = _initial_spiral(sequence, dimension)
        else:
            # Use a different random seed for variety
            rng_init = random.Random(12345 + run * 7919 + len(sequence))
            # Generate a random self-avoiding walk
            initial = [(0, 0)]
            occupied = {(0, 0): 0}
            for i in range(1, n):
                last = initial[-1]
                options = []
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    cand = (last[0] + dx, last[1] + dy)
                    if cand not in occupied:
                        options.append(cand)
                if not options:
                    break
                choice = rng_init.choice(options)
                occupied[choice] = i
                initial.append(choice)
        
        if len(initial) < n:
            # Fallback to spiral
            initial = _initial_spiral(sequence, dimension)
        
        # Run simulated annealing
        chain = _simulated_annealing(sequence, dimension, initial, time_per_run)
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Final fallback
    if best_chain is None or len(best_chain) != n:
        # Use a simple straight line
        best_chain = [(i, 0) for i in range(n)]
    
    # Ensure the chain is valid
    if not _is_valid_walk(best_chain):
        # Last resort: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
