"""HP lattice protein folding using Simulated Annealing with pivot moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Safety margin for 120s CPU / 180s wall clock

def _check_deadline():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _rotate_point(p, pivot, angle):
    """Rotate point p around pivot by 90 degrees * angle times in 2D."""
    x, y = p[0] - pivot[0], p[1] - pivot[1]
    for _ in range(angle):
        x, y = -y, x
    return (pivot[0] + x, pivot[1] + y)

def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk using DFS."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    def dfs(current_idx, current_pos):
        if current_idx == n - 1:
            return True
        # Shuffle neighbors for randomness
        neighbors = list(_neighbours(current_pos))
        rng.shuffle(neighbors)
        for next_pos in neighbors:
            if next_pos not in occupied:
                chain.append(next_pos)
                occupied.add(next_pos)
                if dfs(current_idx + 1, next_pos):
                    return True
                chain.pop()
                occupied.remove(next_pos)
        return False
    
    if not dfs(0, (0, 0)):
        # Fallback to straight line if DFS fails (unlikely for small N, but safe)
        chain = [(i, 0) for i in range(n)]
        occupied = set(chain)
        
    return chain

def _calculate_energy_delta(chain, sequence, start_idx, end_idx, occupied_before, occupied_after):
    """
    Calculate energy difference when segment [start_idx+1 ... end_idx] moves.
    occupied_before: set of all occupied points before move.
    occupied_after: set of all occupied points after move.
    Note: We only need to consider contacts involving the moved segment or its neighbors.
    However, for simplicity and correctness in a pivot move, we can just recalculate energy
    for the moved points and their neighbors.
    """
    # This helper is tricky to get right without full recalculation.
    # Given the small size (<=100 residues), full recalculation is O(N^2) which is ~10^4 ops.
    # 10^4 ops * 10^5 steps = 10^9 ops. This might be too slow in Python.
    # Let's try to be smarter.
    
    # Actually, let's stick to a simpler but slightly slower delta calculation if possible,
    # or just recalculate if N is small.
    # For N=100, O(N^2) is 10,000. 
    # If we do 10,000 steps per second, that's 10^8 ops/sec. 
    # Python does ~10^7 simple ops/sec. So 10,000 ops takes ~1ms.
    # 1000 steps/sec * 120s = 120,000 steps.
    # 120,000 * 1ms = 120s. This is borderline.
    
    # We need a faster delta.
    # The energy is -count(H-H contacts).
    # A move affects contacts:
    # 1. Between moved segment and fixed segment.
    # 2. Within moved segment (internal structure preserved, so internal contacts don't change? 
    #    NO, pivot rotation changes relative positions? No, rigid rotation preserves internal distances.
    #    So internal H-H contacts within the moved segment remain the same.
    # 3. Between fixed segment and moved segment.
    
    # So we only need to check contacts between the moved segment and the rest of the chain.
    moved_indices = range(start_idx + 1, end_idx + 1)
    
    # Old contacts involving moved indices
    old_contacts = 0
    # New contacts involving moved indices
    
    # We need to know which H's are in the moved segment.
    moved_h_indices = [i for i in moved_indices if sequence[i] == 'H']
    
    # For each H in moved segment, check neighbors in the FIXED part of the chain.
    # Fixed part is indices [0...start_idx] U [end_idx+1...N-1]
    # But wait, the moved segment changes position.
    
    # Let's identify the set of points occupied by the fixed chain.
    # fixed_points = {chain[i] for i in range(N) if i not in moved_indices}
    
    # Old Energy Contribution from contacts involving moved Hs:
    # For each i in moved_h_indices:
    #   for each neighbor q of chain[i]:
    #     if q is occupied by a fixed H (index j not in moved_indices, j != i):
    #        old_contacts += 1 (if we count each pair once, we need to be careful)
        
    # To avoid double counting, let's just count all H-H pairs where at least one index is in moved_indices.
    # Since internal pairs (both in moved) don't change, we can subtract them or ignore them if we calculate delta correctly.
    
    # Let's calculate Total Old Energy and Total New Energy? No, too slow.
    
    # Let's calculate Delta E.
    # Delta E = New Contacts - Old Contacts.
    # Contacts are pairs (i, j) such that |i-j| > 1, dist(chain[i], chain[j]) == 1, seq[i]=='H', seq[j]=='H'.
    
    # The set of pairs that can change is:
    # Pairs where i is in moved_indices and j is NOT in moved_indices.
    # Pairs where i is NOT in moved_indices and j IS in moved_indices. (Same set)
    
    # So we just need to check neighbors of the moved points against the fixed points.
    
    # Build a map from point to index for the FIXED chain only?
    # Or just use the global map but filter indices.
    
    where = {p: i for i, p in enumerate(chain)}
    
    old_delta = 0
    new_delta = 0
    
    # We need the new positions of the moved points to check new contacts.
    # The caller should provide the new chain or we compute it here?
    # Let's assume this function is called with the state BEFORE the move, 
    # and we pass the new positions of the moved segment.
    
    # Actually, it's safer to just recalculate the full energy if N < 50, 
    # and use delta for larger N.
    # Given the benchmark sequences are 20-100, and Python is slow, 
    # let's try to implement the delta properly.
    
    # Input: 
    # chain: current chain
    # sequence: seq
    # start_idx, end_idx: range of moved segment (inclusive)
    # new_positions: list of new positions for indices start_idx+1...end_idx
    
    # 1. Remove old contacts involving moved segment
    # 2. Add new contacts involving moved segment
    
    # To do this efficiently, we need to know which points are occupied by the FIXED chain.
    fixed_occupied_h = set()
    for i in range(len(chain)):
        if i < start_idx + 1 or i > end_idx:
            if sequence[i] == 'H':
                fixed_occupied_h.add(chain[i])
                
    # Old contacts:
    for i in range(start_idx + 1, end_idx + 1):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                if q in fixed_occupied_h:
                    old_delta += 1
                    
    # New contacts:
    # The new positions of the moved segment are in new_positions
    # The fixed chain remains the same.
    for i, new_p in zip(range(start_idx + 1, end_idx + 1), new_positions):
        if sequence[i] == 'H':
            for q in _neighbours(new_p):
                if q in fixed_occupied_h:
                    new_delta += 1
                    
    return new_delta - old_delta

def _simulate_annealing(sequence, dimension, rng, initial_chain, time_budget):
    """Run SA on the given initial chain until time budget expires."""
    n = len(sequence)
    if n < 2:
        return initial_chain
    
    chain = list(initial_chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Cooling schedule
    T = 1.0
    alpha = 0.9995  # Cooling rate
    # Min T
    T_min = 0.01
    
    start_time = time.time()
    
    while True:
        if _check_deadline():
            break
        if (time.time() - start_time) > time_budget:
            break
        if T < T_min:
            # Restart or break? Let's break this run.
            break
            
        # Choose a random pivot point
        # Pivot at i means we rotate segment i+1...N-1 around i
        # i can be 0 to N-2
        pivot_idx = rng.randint(0, n - 2)
        
        # Choose rotation: 1, 2, or 3 times 90 degrees
        rotation = rng.choice([1, 2, 3])
        
        # Calculate new positions for segment pivot_idx+1 ... n-1
        pivot_point = chain[pivot_idx]
        
        # Check for collisions and calculate new positions
        new_segment = []
        valid = True
        # We need to check against the fixed part of the chain (indices 0...pivot_idx)
        # And against the new segment itself (self-collision)
        
        # Fixed occupied set (excluding the segment to be moved)
        fixed_occupied = set(chain[:pivot_idx+1])
        
        # Note: The segment being moved is chain[pivot_idx+1:]
        
        for i in range(pivot_idx + 1, n):
            old_pos = chain[i]
            new_pos = _rotate_point(old_pos, pivot_point, rotation)
            
            # Check collision with fixed part
            if new_pos in fixed_occupied:
                valid = False
                break
            
            # Check collision with previously placed new positions
            if new_pos in new_segment:
                valid = False
                break
                
            new_segment.append(new_pos)
            
        if not valid:
            continue
            
        # Calculate energy delta
        # We need to pass the new positions to the delta calculator
        # But _calculate_energy_delta expects the old chain and new positions.
        
        # Let's optimize: we already have fixed_occupied_h? No, we need H's specifically.
        # Let's just call the delta function.
        
        # Wait, the delta function I wrote above assumes `chain` is the old chain.
        # It constructs `fixed_occupied_h` from `chain` excluding the moved part.
        # This is correct.
        
        delta_e = _calculate_energy_delta(chain, sequence, pivot_idx, n - 1, None, new_segment)
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Apply move
            for i, new_pos in zip(range(pivot_idx + 1, n), new_segment):
                chain[i] = new_pos
                
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        # Cool
        T *= alpha
        
    return best_chain

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

def fold(sequence, dimension):
    """Best conformation using multiple SA restarts."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
        
    # Determine number of restarts based on time
    # Let's say each restart gets 5 seconds max, or until T_min.
    # We will loop until deadline.
    
    global_best_chain = None
    global_best_energy = 1 # Max possible energy is 0 (unfolded), but we want min (most negative)
    
    # Use a fixed seed for reproducibility in evaluation? 
    # The task says "randomized", but deterministic is safer for debugging.
    # However, different runs might find different local minima.
    # Let's use a master RNG.
    master_rng = random.Random(42)
    
    while not _check_deadline():
        # 1. Generate initial chain
        init_chain = _initial_chain(sequence, dimension, master_rng)
        
        # 2. Run SA for a short duration
        # Estimate time per restart. 
        # If N=100, maybe 1-2 seconds per restart is good.
        # Let's allow 2 seconds per restart.
        restart_budget = 2.0
        
        sa_chain = _simulate_annealing(sequence, dimension, master_rng, init_chain, restart_budget)
        
        sa_energy = _energy(sequence, sa_chain)
        
        if global_best_chain is None or sa_energy < global_best_energy:
            global_best_energy = sa_energy
            global_best_chain = sa_chain
            
        # Check if we have time for another restart
        if _check_deadline():
            break
            
    if global_best_chain is None:
        # Fallback
        global_best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in global_best_chain]
# EVOLVE-BLOCK-END
