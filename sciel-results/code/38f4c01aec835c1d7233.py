"""
HP lattice protein folding using a Hamiltonian Path Skeleton + Pivot Search strategy.

Strategy:
1. Generate a compact, space-filling self-avoiding walk (skeleton) on the square lattice.
   This ensures the chain occupies a dense region, maximizing the geometric potential for contacts.
   We use a deterministic "Boustrophedon" (snake-like) path in a grid that is as close to square as possible.
2. Use Pivot Search (Monte Carlo local search) to optimize the energy.
   - A pivot move selects a random pivot point in the chain and rotates the segment beyond it by 90, 180, or 270 degrees.
   - We accept the move if it is self-avoiding and does not increase energy (or with some probability if it does, to escape local minima, though for short time budgets, greedy or near-greedy is often sufficient).
   - Since we start from a valid self-avoiding walk, we only need to check for collisions during the rotation.
3. Run multiple restarts with different random seeds for the pivot search and potentially different skeleton orientations/sizes to find the best conformation.

Why this is better:
- The current "grow" strategy is myopic and often creates unrepairable traps.
- Pivot search on a compact skeleton allows global rearrangements that fix local sub-optimality in the H-H contact placement.
- It is a standard, proven method for HP folding that scales well for N=100.
"""

import random
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Generate the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Compute HP contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _generate_skeleton(n):
    """
    Generate a compact self-avoiding walk of length n on the square lattice.
    Uses a Boustrophedon (snake) pattern in a grid.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Find a grid size close to square that can hold n points
    # We want width W and height H such that W*H >= n
    # To keep it compact, we want W and H to be close.
    # Start with H = ceil(sqrt(n)), then adjust W.
    
    # Try to find a grid that fits exactly or is very close
    # We'll iterate over possible heights
    best_w, best_h = 1, n
    min_diff = float('inf')
    
    for h in range(1, int(n**0.5) + 2):
        w = (n + h - 1) // h  # ceil(n/h)
        # Check if w*h >= n
        if w * h >= n:
            diff = abs(w - h)
            if diff < min_diff:
                min_diff = diff
                best_w, best_h = w, h
            
    W, H = best_w, best_h
    
    # Generate the snake path
    chain = []
    # We need to fill W*H points, but we only need n.
    # The snake path visits all cells in the grid.
    # We will generate the full grid path and truncate to n.
    
    for y in range(H):
        if y % 2 == 0:
            # Left to Right
            for x in range(W):
                chain.append((x, y))
        else:
            # Right to Left
            for x in range(W-1, -1, -1):
                chain.append((x, y))
        
        if len(chain) >= n:
            break
    
    # Truncate to n
    chain = chain[:n]
    
    return chain

def _pivot_search(sequence, chain, time_limit=5.0, rng=None):
    """
    Perform pivot search to minimize energy.
    """
    if rng is None:
        rng = random.Random()
        
    n = len(chain)
    if n < 2:
        return chain, _energy(sequence, chain)
    
    best_chain = chain
    best_energy = _energy(sequence, chain)
    
    # Precompute H positions for faster energy delta calculation?
    # For now, full re-evaluation is O(N) which is fine for N=100 and few iterations.
    # But we can optimize by only checking the moved segment.
    
    # To speed up, we can maintain a set of occupied positions
    # However, pivot moves change many positions.
    # Given N=100, O(N) energy calc is ~100 operations.
    # We can do ~10000 pivots in 5 seconds if overhead is low.
    
    # Optimized energy check for pivot:
    # Only the segment [pivot+1, n-1] moves.
    # We need to check:
    # 1. Self-avoidance: new positions of the moved segment must not collide with fixed segment or themselves.
    # 2. Energy change: 
    #    - Remove contacts involving the moved segment.
    #    - Add new contacts involving the moved segment.
    
    # Let's implement a simpler version first and optimize if needed.
    # For N=100, full re-evaluation might be too slow if we do many iterations.
    # Let's try to estimate: 10000 iterations * 100 ops/iter = 1M ops. Python does ~10M ops/sec.
    # So 1M ops is 0.1s. This is feasible.
    
    # However, the collision check for the moved segment is also O(M) where M is segment length.
    # Average M is N/2 = 50.
    # So total ops per pivot ~ 150.
    # 10000 pivots ~ 1.5M ops. ~0.15s.
    # We can do more iterations.
    
    start_time = time.time()
    
    # Directions for rotation: 90, 180, 270
    # Rotation 90 deg CCW: (x, y) -> (-y, x)
    # Rotation 180 deg: (x, y) -> (-x, -y)
    # Rotation 270 deg CCW: (x, y) -> (y, -x)
    
    while time.time() - start_time < time_limit:
        # Choose a random pivot point
        # Pivot point i: the segment from i+1 to n-1 rotates around i
        # i can be 0 to n-2
        i = rng.randint(0, n - 2)
        
        # Choose a random rotation
        # 0: 90 deg, 1: 180 deg, 2: 270 deg
        rot = rng.randint(0, 2)
        
        # Calculate new positions for the segment
        pivot_x, pivot_y = chain[i]
        
        new_chain = chain[:i+1]  # Copy the fixed part
        
        # We need to be careful with memory allocation.
        # Let's create a list of new positions for the moving part.
        moving_part = chain[i+1:]
        
        # Calculate new coordinates
        new_moving_part = []
        for p in moving_part:
            dx = p[0] - pivot_x
            dy = p[1] - pivot_y
            if rot == 0: # 90 deg CCW
                new_dx, new_dy = -dy, dx
            elif rot == 1: # 180 deg
                new_dx, new_dy = -dx, -dy
            else: # 270 deg CCW
                new_dx, new_dy = dy, -dx
            
            new_x = pivot_x + new_dx
            new_y = pivot_y + new_dy
            new_moving_part.append((new_x, new_y))
        
        new_chain.extend(new_moving_part)
        
        # Check self-avoidance
        # Create a set of all positions
        # If any duplicate, invalid.
        pos_set = set()
        valid = True
        for p in new_chain:
            if p in pos_set:
                valid = False
                break
            pos_set.add(p)
        
        if not valid:
            continue
            
        # Calculate energy
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            # Restart timer? No, just keep going.
            # Maybe add a small probability of accepting worse moves to escape local minima?
            # For now, let's stick to greedy to ensure progress.
            # If we get stuck, the multiple restarts in fold() will help.
            
        # Optional: Simulated Annealing style acceptance
        # if new_energy > best_energy:
        #     delta = new_energy - best_energy
        #     if rng.random() < exp(-delta / temperature):
        #         best_chain = new_chain
        #         best_energy = new_energy
        #         temperature *= 0.99
        # But we don't have a global temperature variable here easily without passing it.
        # Let's keep it simple: greedy.
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using a Hamiltonian Path Skeleton + Pivot Search.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Time budget management
    # Total budget is 120s for 10 sequences, so ~12s per sequence on average.
    # But the evaluator kills at 210s wall clock.
    # We should be conservative.
    start_time = time.time()
    # Let's aim to use up to 10 seconds per sequence to be safe, 
    # but check total time if possible. Since we don't know which sequence we are on,
    # we assume a fixed budget per call.
    # However, if we are called 10 times, 10*10=100s is safe.
    time_limit_per_seq = 10.0
    
    best_chain = None
    best_energy = 0  # 0 is no contacts (worst)
    
    # Number of restarts
    # We can do more restarts if they are fast.
    # Skeleton generation is O(N). Pivot search is the main cost.
    num_restarts = 5
    
    # To diversify, we can try different skeleton orientations or slight perturbations.
    # But the Boustrophedon skeleton is quite rigid.
    # We can rotate the entire skeleton before starting pivot search.
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit_per_seq:
            break
        
        # Generate skeleton
        chain = _generate_skeleton(n)
        
        # Apply a random rotation/reflection to the skeleton to break symmetry?
        # Pivot search will handle local rearrangements, but starting from a rotated skeleton
        # might lead to different local minima.
        if restart > 0:
            # Rotate the whole chain around origin
            rot = restart % 4
            new_chain = []
            for p in chain:
                x, y = p
                if rot == 0:
                    pass
                elif rot == 1:
                    x, y = -y, x
                elif rot == 2:
                    x, y = -x, -y
                else:
                    x, y = y, -x
                new_chain.append((x, y))
            chain = new_chain
        
        # Run pivot search
        rng = random.Random(12345 + restart * 101 + n)
        # Allocate time for this restart
        # If we have 5 restarts and 10s total, we give 2s each.
        # But we can be smarter: give more time to earlier restarts?
        # Or just split evenly.
        time_for_this = (time_limit_per_seq - (time.time() - start_time)) / (num_restarts - restart)
        time_for_this = max(0.1, min(5.0, time_for_this)) # Cap at 5s per restart to avoid one taking too long
        
        chain, energy = _pivot_search(sequence, chain, time_limit=time_for_this, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no valid chain was found (should be rare), return a simple linear chain
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
