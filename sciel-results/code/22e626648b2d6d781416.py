"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
# We set a slightly tighter internal deadline to ensure we return a result safely.
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Return the 4 unit-step neighbours of a 2D point (tuple) as a list."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            # Right
            if (x + 1, y) in where:
                j = where[(x + 1, y)]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
            # Left
            if (x - 1, y) in where:
                j = where[(x - 1, y)]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
            # Up
            if (x, y + 1) in where:
                j = where[(x, y + 1)]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
            # Down
            if (x, y - 1) in where:
                j = where[(x, y - 1)]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _hilbert_curve(order):
    """
    Generate a Hilbert curve of the given order on a 2^order x 2^order grid.
    Returns a list of (x, y) coordinates in the order of the path.
    """
    size = 1 << order
    curve = []
    
    def hilbert(n, x, y, dx, dy):
        if n == 0:
            curve.append((x, y))
            return
        nn = n - 1
        # Rotate/reflect
        if dx * dy < 0:
            x = size - 1 - x
            y = size - 1 - y
        dx, dy = dy, -dx
        # Recurse
        hilbert(nn, x, y, dx, dy)
        hilbert(nn, x + dx, y + dy, dx, dy)
        hilbert(nn, x + dx * 2, y + dy * 2, dx, dy)
        hilbert(nn, x + dx * 2, y + dy, dx, dy)
        
    # Standard Hilbert curve starts at (0,0) with dx=1, dy=1? 
    # Actually, standard recursive definition varies. 
    # Let's use a simpler iterative approach or a known correct recursive one.
    # The above recursive one is a bit tricky with the rotation logic.
    # Let's use a simpler "Peano" or just a snake pattern if Hilbert is too complex to get right quickly.
    # A snake pattern (boustrophedon) is also a Hamiltonian path and much easier to implement correctly.
    # Snake pattern:
    # Row 0: (0,0), (1,0), ..., (size-1, 0)
    # Row 1: (size-1, 1), ..., (0, 1)
    # Row 2: (0, 2), ..., (size-1, 2)
    # This is a valid self-avoiding walk.
    
    curve = []
    for y in range(size):
        if y % 2 == 0:
            for x in range(size):
                curve.append((x, y))
        else:
            for x in range(size - 1, -1, -1):
                curve.append((x, y))
    return curve


def _generate_initial_chain(n, rng):
    """
    Generate an initial self-avoiding chain of length n using a space-filling curve (Snake).
    We pick a random contiguous segment of the snake curve.
    """
    # Determine grid size
    # We need a square grid large enough to hold n residues.
    side = int(math.ceil(math.sqrt(n)))
    # Add some padding to ensure we can fit a path of length n easily
    # The snake curve on a side x side grid has side*side points.
    # If side*side < n, increase side.
    while side * side < n:
        side += 1
        
    curve = _hilbert_curve(order=0) # dummy, we use the snake logic inside _hilbert_curve? 
    # Wait, my _hilbert_curve function above actually implements a Snake pattern in the comment block 
    # but the code is recursive Hilbert. Let's replace it with a simple snake generator.
    
    # Re-implementing simple snake generation for clarity and correctness
    snake_curve = []
    for y in range(side):
        if y % 2 == 0:
            for x in range(side):
                snake_curve.append((x, y))
        else:
            for x in range(side - 1, -1, -1):
                snake_curve.append((x, y))
            
    # Pick a random start index
    max_start = len(snake_curve) - n
    if max_start < 0:
        # If n is larger than the grid (shouldn't happen due to while loop), expand grid
        # But side*side >= n is guaranteed.
        pass
        
    start_idx = rng.randint(0, max_start)
    chain = snake_curve[start_idx : start_idx + n]
    
    # Translate to start at (0,0) for consistency? Not strictly necessary, but good.
    min_x = min(p[0] for p in chain)
    min_y = min(p[1] for p in chain)
    chain = [(p[0] - min_x, p[1] - min_y) for p in chain]
    
    return chain


def _pivot_move(chain, n, rng):
    """
    Perform a pivot move on the chain.
    Returns the new chain if valid, otherwise None.
    """
    # Pick a pivot point i (1 <= i < n-1)
    # We usually pick i close to the middle or randomly.
    # To allow large rearrangements, pick i randomly from [1, n-1]
    i = rng.randint(1, n - 1)
    
    # The segment from i to n-1 is rotated 180 degrees around point i-1?
    # No, a pivot move rotates the suffix (i..n-1) around the bond (i-1, i).
    # The new position of i is the reflection of the old position of i across point i-1.
    # Let p_prev = chain[i-1], p_curr = chain[i].
    # new_p_curr = 2 * p_prev - p_curr.
    # The rest of the suffix is translated by the same vector.
    
    p_prev = chain[i - 1]
    p_curr = chain[i]
    
    # Vector from p_curr to p_prev
    dx = p_prev[0] - p_curr[0]
    dy = p_prev[1] - p_curr[1]
    
    # The new chain will have chain[0..i-1] unchanged.
    # The new chain[i] will be (p_curr[0] + dx, p_curr[1] + dy)
    # The new chain[j] for j > i will be (old_chain[j][0] + dx, old_chain[j][1] + dy)
    
    new_chain = chain[:i]
    # Calculate new positions for the suffix
    # We need to check if any of the new positions collide with the fixed prefix (0..i-1)
    # or with each other (they won't, as it's a rigid body translation/rotation of a valid path).
    # Actually, a 180 degree rotation around the bond keeps the relative distances in the suffix intact.
    # So we only need to check collisions with the prefix.
    
    occupied_prefix = {p: idx for idx, p in enumerate(chain[:i])}
    
    # Construct the new suffix
    # new_pos[0] corresponds to index i
    # new_pos[k] corresponds to index i+k
    # new_pos[0] = p_curr + (dx, dy)
    # new_pos[k] = p_curr + (dx, dy) + (chain[i+k] - p_curr) = chain[i+k] + (dx, dy)
    
    # Wait, the rotation is 180 degrees.
    # If we rotate the vector (p_curr - p_prev) by 180, it becomes (p_prev - p_curr).
    # So the new vector from p_prev to new_p_curr is (p_prev - p_curr).
    # Thus new_p_curr = p_prev + (p_prev - p_curr) = 2*p_prev - p_curr.
    # The rest of the chain is translated by the same shift vector: shift = new_p_curr - p_curr = 2*p_prev - 2*p_curr.
    
    shift_x = 2 * p_prev[0] - 2 * p_curr[0]
    shift_y = 2 * p_prev[1] - 2 * p_curr[1]
    
    # Check collisions
    for k in range(i, n):
        new_x = chain[k][0] + shift_x
        new_y = chain[k][1] + shift_y
        new_p = (new_x, new_y)
        if new_p in occupied_prefix:
            return None
            
    # If no collisions, construct new chain
    for k in range(i, n):
        new_x = chain[k][0] + shift_x
        new_y = chain[k][1] + shift_y
        new_chain.append((new_x, new_y))
        
    return new_chain


def _simulated_annealing(sequence, initial_chain, rng, time_budget):
    """
    Perform Simulated Annealing using pivot moves.
    """
    n = len(initial_chain)
    current_chain = initial_chain
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = current_chain
    best_energy = current_energy
    
    # Cooling schedule
    # Start high, end low.
    # We don't know the exact number of steps, so we use time.
    # Let's estimate steps per second.
    # A pivot move is O(n). _compute_energy is O(n).
    # For n=100, this is fast. We can do thousands of steps per second.
    
    # Let's define a fixed number of steps and adjust temperature based on progress.
    # Or just use a fixed schedule of temperatures.
    
    # Heuristic: 
    # For n=20, we can do many moves.
    # For n=100, fewer moves.
    # Let's aim for ~100,000 to 500,000 moves total depending on n.
    max_steps = max(50000, 2000000 // n)
    
    # Time check inside loop
    start_time = time.time()
    
    # Temperature schedule: T = T0 * (T_end/T0)^(step/max_steps)
    T0 = 5.0
    T_end = 0.01
    
    for step in range(max_steps):
        if time.time() > time_budget:
            break
            
        # Calculate current temperature
        if max_steps > 0:
            progress = step / max_steps
            T = T0 * (T_end / T0) ** progress
        else:
            T = T0
            
        # Generate new chain
        new_chain = _pivot_move(current_chain, n, rng)
        if new_chain is None:
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        else:
            # Reject
            pass
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using an initial space-filling curve and Simulated Annealing.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
        
    best_chain_overall = None
    best_energy_overall = 0 # We want to minimize (more negative is better)
    
    # We will run SA from multiple different initial snake segments to diversify the search.
    # How many restarts?
    # Each SA run takes some time.
    # Let's allocate time per run.
    # Total budget: ~175s.
    # Let's try 10-20 restarts.
    
    num_restarts = 15
    
    # Estimate time per restart
    # We don't know exactly, so we'll just run them sequentially and stop if time runs out.
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate initial chain
        # Vary the random seed for the initial chain selection
        init_rng = random.Random(42 + r)
        init_chain = _generate_initial_chain(n, init_rng)
        
        # Run SA
        # Allocate time budget for this run
        # If we have 175s total and 15 restarts, that's ~11s per restart.
        # But later restarts might be faster if we stop early? No, SA runs fixed steps or time.
        # Let's give each run a fixed time slice, reducing if time is running out.
        remaining_time = _DEADLINE - time.time()
        if remaining_time < 5:
            break
            
        # Allocate up to 10% of remaining time to this run, min 1s
        run_time_budget = max(1.0, remaining_time * 0.1)
        
        chain, energy = _simulated_annealing(sequence, init_chain, rng, time.time() + run_time_budget)
        
        if chain is None:
            continue
            
        if best_chain_overall is None or energy < best_energy_overall:
            best_chain_overall = chain
            best_energy_overall = energy
            
    # Fallback if SA failed (shouldn't happen, but just in case)
    if best_chain_overall is None:
        best_chain_overall = _generate_initial_chain(n, rng)
        
    return [list(p) for p in best_chain_overall]
# EVOLVE-BLOCK-END
