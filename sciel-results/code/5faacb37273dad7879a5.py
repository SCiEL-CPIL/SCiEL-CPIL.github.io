"""Rewritten program for HP lattice protein folding with optimized simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Pre-compute neighbor offsets for 3D cubic lattice
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _neighbours(point):
    """Generate 6 neighbours of a 3D lattice point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y + 1, z) # Wait, duplicate in manual list? No, standard is 6.
    # Let's use the pre-computed offsets for speed in tight loops if needed, 
    # but for clarity and current structure, yielding is fine.
    # Actually, let's just use the offsets directly in the hot loop.
    pass

def _get_neighbours_fast(x, y, z):
    """Fast neighbor generation using pre-computed offsets."""
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth with improved heuristic."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        last_pos = chain[-1]
        x, y, z = last_pos
        
        # Generate candidate positions
        candidates = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        for site in candidates:
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                # Check neighbors of site for existing H residues
                # Neighbors: (sx±1, sy, sz), (sx, sy±1, sz), (sx, sy, sz±1)
                for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,sz+1), (sx,sy,sz-1)]:
                    j = occupied.get((nx, ny, nz))
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Calculate free space: number of unoccupied neighbors
            free = 0
            sx, sy, sz = site
            for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,sz+1), (sx,sy,sz-1)]:
                if (nx, ny, nz) not in occupied:
                    free += 1
            
            # Score: Weighted gain + free space + random noise
            # Tune weights: Gain is critical. Free space prevents trapping.
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Select best option
        max_score = -1
        best_site = None
        for score, site in options:
            if score > max_score:
                max_score = score
                best_site = site
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _energy(sequence, chain):
    """Calculate energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            # Check 6 neighbors
            for nx, ny, nz in [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]:
                j = where.get((nx, ny, nz))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _delta_energy_flip(sequence, chain, i, new_pos):
    """
    Calculate energy change when moving residue i from chain[i] to new_pos.
    Assumes the move is valid (self-avoiding, unit steps).
    Only contacts involving residue i change.
    """
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == "H"
    if not is_h:
        return 0
    
    # We need to count H-H contacts involving i at old_pos and new_pos.
    # A contact exists if neighbor j is H, j != i, and |j - i| > 1.
    
    # To do this fast, we need to know which residues are at which positions.
    # We don't have a global map here to avoid rebuilding it.
    # However, we can pass the map or rebuild it? Rebuilding is O(N).
    # For N=136, O(N) is ~100 ops. 
    # If we do this for every step, it's slow.
    # We should maintain a position-to-index map in the annealing loop.
    
    # This function signature is tricky if we don't have the map.
    # Let's assume the caller has the map or we compute it locally.
    # For efficiency, the annealing loop should maintain `pos_to_idx`.
    
    # Since this is a helper, let's make it take the map.
    # But the previous interface didn't. 
    # I will modify the annealing loop to maintain the map and calculate delta inline or with a helper that takes the map.
    
    # For now, let's return 0 and handle in loop? No, that breaks correctness.
    # I will rewrite the annealing loop to maintain the map and calculate delta inline.
    pass

def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with fast flip moves and maintained position map."""
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Maintain a map from position to index for O(1) neighbor lookups
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    # Temperature schedule
    # Start high enough to accept bad moves, end low to refine.
    # Energy scale is roughly -N/2 to 0.
    T0 = 2.0 
    T_min = 0.01
    
    # Estimate steps
    # We want to run as many steps as possible before deadline.
    # Each step involves:
    # 1. Pick random i (O(1))
    # 2. Check neighbors of chain[i] (6 checks)
    # 3. Check validity (set lookup O(1))
    # 4. Calculate delta (check 12 neighbors max, map lookup O(1))
    # 5. Accept/Reject
    # Roughly 50-100 Python ops per step.
    # 100 ops * 10^6 steps = 10^8 ops ~ 10s in Python? Maybe 1-2s for 10^6 steps.
    # Let's target 1-5 million steps per run.
    
    time_left = deadline - time.time()
    if time_left <= 0.1:
        return best_chain
    
    # Adaptive step count: aim to use 80% of available time for this anneal
    # We will run in a loop checking time every N steps.
    
    T = T0
    # We need a cooling factor. 
    # Let's use a fixed number of steps for this run, determined by time budget.
    # Or just run until time runs out, cooling exponentially.
    
    # Let's use a simple geometric cooling.
    # If we don't know total steps, we can just cool based on time or step count.
    # Let's assume we run for `max_steps` steps.
    # Estimate max_steps based on time.
    # Let's try 1,000,000 steps if time allows.
    
    # To be safe and effective, we will run multiple shorter anneals in `fold`.
    # This anneal function will run for a fixed duration or step count.
    # Let's run for `time_budget` seconds.
    
    start_time = time.time()
    step = 0
    last_check_time = start_time
    
    # Cooling: T = T0 * (T_min/T0)^(step / total_estimated_steps)
    # Since we don't know total steps, let's use a fixed cooling rate per step.
    # Or, simpler: reduce T by a small factor every 1000 steps?
    # Better: T = T0 * exp(-step / tau).
    # Let's set tau such that T drops to T_min in ~100,000 steps.
    tau = 100000.0 / math.log(T0 / T_min)
    
    while True:
        # Check time every 1000 steps
        if step % 1000 == 0:
            current_time = time.time()
            if current_time >= deadline - 0.1:
                break
            # Update T
            T = T0 * math.exp(-step / tau)
            if T < T_min:
                T = T_min
        
        step += 1
        
        # Pick a random internal residue to move
        # Avoid endpoints (0 and n-1) to maintain unit steps easily?
        # Actually, moving endpoints is harder to validate. 
        # Moving internal residues i (1 to n-2) is standard.
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try to move i to a random valid neighbor
        # Get neighbors of old_pos
        x, y, z = old_pos
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        # Shuffle to randomize
        rng.shuffle(neighbors)
        
        valid_move_found = False
        for new_pos in neighbors:
            # Check if new_pos is occupied by another residue
            if new_pos in pos_to_idx:
                # If it's occupied by i itself, skip (shouldn't happen as we check old_pos != new_pos)
                # If occupied by j != i, invalid
                continue
            
            # Check unit steps with neighbors i-1 and i+1
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            # Manhattan distance must be 1
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) + abs(new_pos[2] - prev_pos[2]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) + abs(new_pos[2] - next_pos[2]) != 1:
                continue
            
            # Valid move!
            # Calculate delta energy
            delta = 0
            if sequence[i] == "H":
                # Remove old contacts
                # Check neighbors of old_pos
                ox, oy, oz = old_pos
                for nx, ny, nz in [(ox+1,oy,oz), (ox-1,oy,oz), (ox,oy+1,oz), (ox,oy-1,oz), (ox,oy,oz+1), (ox,oy,oz-1)]:
                    j = pos_to_idx.get((nx, ny, nz))
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                        delta += 1 # Removing a contact increases energy (less negative)
                
                # Add new contacts
                nx, ny, nz = new_pos
                for nxx, nyy, nzz in [(nx+1,ny,nz), (nx-1,ny,nz), (nx,ny+1,nz), (nx,ny-1,nz), (nx,ny,nz+1), (nx,ny,nz-1)]:
                    j = pos_to_idx.get((nxx, nyy, nzz))
                    if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                        delta -= 1 # Adding a contact decreases energy (more negative)
            
            # Acceptance criterion
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Apply move
                pos_to_idx[old_pos] = -1 # Mark as free temporarily
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                
                valid_move_found = True
                break
        
        if not valid_move_found:
            continue
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multi-start growth + simulated annealing."""
    # Total budget is 240s for all sequences, but we are called per sequence.
    # The prompt says "240 s CPU budget for all sequences together".
    # However, the function is called once per sequence.
    # We must manage our own time usage to ensure we don't exceed the total budget 
    # if we were responsible for all, but since we are just one instance, 
    # we should use as much time as we can for this sequence?
    # No, the evaluator calls fold() for each sequence. 
    # If there are 26 sequences, and total budget is 240s, we have ~9s per sequence.
    # We should set a deadline based on the time remaining for this call?
    # We don't know when other calls happen. 
    # Safest: Assume we have ~9s per sequence. 
    # Or, use a fixed time limit per sequence, e.g., 8s, to be safe.
    
    # Let's use a dynamic deadline. 
    # If we assume the evaluator runs them sequentially, we can't know the global time.
    # But we can measure how long we've been running? No, process start.
    # Let's just use a fixed time budget per sequence, e.g., 8.5 seconds.
    # This leaves 0.5s margin for overhead.
    
    start_time = time.time()
    deadline = start_time + 8.5
    
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0
    
    n = len(sequence)
    
    # Phase 1: Multiple growth starts
    # Number of growths depends on time.
    # Growth is fast. Let's do 10-20.
    num_growths = 15
    for g in range(num_growths):
        if time.time() >= deadline - 2.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Annealing
    # Use remaining time for annealing.
    # We will perform multiple annealing runs if time permits, or one long one.
    # One long one is better for deep search.
    
    if time.time() < deadline - 0.5:
        annealed = _anneal(sequence, best, dimension, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy
        
        # If time remains, try a new growth and anneal again
        if time.time() < deadline - 1.0:
            chain2 = _grow(sequence, dimension, rng)
            if chain2 is not None:
                e2 = _energy(sequence, chain2)
                if e2 < best_energy:
                    best, best_energy = chain2, e2
                annealed2 = _anneal(sequence, best, dimension, rng, deadline)
                e2a = _energy(sequence, annealed2)
                if e2a < best_energy:
                    best, best_energy = annealed2, e2a
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
