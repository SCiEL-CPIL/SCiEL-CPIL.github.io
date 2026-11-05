"""HP lattice protein folding on the square lattice using model-based optimization with crankshaft and pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _neighbours(point):
    """Yield the 4 lattice neighbours of a 2D point."""
    x, y = point
    yield (x+1, y)
    yield (x-1, y)
    yield (x, y+1)
    yield (x, y-1)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, s in enumerate(sequence) if s == "H"]
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            j = where.get((nx, ny))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
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
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_crankshaft(sequence, chain, k1, k2, rot_choice):
    """
    Attempt a crankshaft move: rotate the segment chain[k1+1..k2] around the pivot at chain[k1].
    rot_choice: 0=90deg, 1=180deg, 2=270deg
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if k2 - k1 < 2:
        return None, None
    
    px, py = chain[k1]
    
    if rot_choice == 0:  # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:  # 180 deg
        def rot(x, y): return (-x, -y)
    else:  # 270 deg
        def rot(x, y): return (y, -x)
        
    # Build prefix and suffix
    prefix = chain[:k1+1]
    suffix = chain[k2+1:]
    
    # The moving segment is chain[k1+1..k2]
    seg = chain[k1+1:k2+1]
    
    # Check that the pivot point is not in suffix (it shouldn't be, but just in case)
    # Also need to check against all non-moving parts
    
    # Build set of occupied points in prefix and suffix
    fixed_occupied = set(prefix) | set(suffix)
    
    new_seg = []
    for j in range(len(seg)):
        ox = seg[j][0] - px
        oy = seg[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_seg.append(new_pos)
    
    # Check for collisions
    for pos in new_seg:
        if pos in fixed_occupied:
            return None, None
    # Check for self-collisions within new_seg
    if len(set(new_seg)) != len(new_seg):
        return None, None
    
    new_chain = prefix + new_seg + suffix
    
    # Verify self-avoidance and unit steps
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _try_pivot_move(sequence, chain, k, rot_choice):
    """
    Attempt a pivot move at index k with a specific rotation.
    rot_choice: 0=90deg, 1=180deg, 2=270deg
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if k >= n - 1:
        return None, None
    
    px, py = chain[k]
    
    if rot_choice == 0:  # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:  # 180 deg
        def rot(x, y): return (-x, -y)
    else:  # 270 deg
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k+1])
    
    new_chain = list(chain[:k+1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
            
        new_chain.append(new_pos)
        
    # Verify self-avoidance and unit steps
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _try_two_point_swap(sequence, chain, i, j):
    """
    Attempt a two-point swap: swap positions of residues at indices i and j.
    Only valid if the resulting chain is still a valid self-avoiding walk with unit steps.
    This is a rare but powerful move that can escape deep local minima.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if i >= j:
        i, j = j, i
    if j - i < 2:
        return None, None
    
    # For a two-point swap to preserve unit steps, we need:
    # The segment between i and j must be able to "flip" in place
    # This is complex; a simpler approach: try swapping and check validity
    
    new_chain = list(chain)
    new_chain[i], new_chain[j] = new_chain[j], new_chain[i]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None, None
        seen.add(p)
    
    # Check unit steps
    for k in range(1, n):
        dx = abs(new_chain[k][0] - new_chain[k-1][0])
        dy = abs(new_chain[k][1] - new_chain[k-1][1])
        if dx + dy != 1:
            return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot, crankshaft, and two-point swap moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters - tuned for better exploration
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type: pivot (50%), crankshaft (35%), two-point swap (15%)
        move_type = rng.random()
        
        if move_type < 0.35 and n > 6:
            # Crankshaft: pick k1, k2 with k2-k1 >= 2
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        elif move_type < 0.50 and n > 4:
            # Two-point swap: rare but powerful
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            new_chain, new_energy = _try_two_point_swap(sequence, current_chain, i, j)
        else:
            # Pivot move
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_rounds=200):
    """
    Greedy polish: repeatedly scan all possible pivot, crankshaft, and two-point swap moves
    and accept any that improve energy. Stop when no move improves energy or max_rounds reached.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        
        # Try pivot moves
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        # Try crankshaft moves
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Try two-point swap moves (limited to avoid O(n^2) cost)
        if n > 4 and not improved:
            # Sample a subset of swap pairs to keep cost manageable
            num_samples = min(20, n * n // 4)
            for _ in range(num_samples):
                i = random.randint(0, n - 3)
                j = random.randint(i + 2, n - 1)
                new_chain, new_energy = _try_two_point_swap(sequence, current_chain, i, j)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    # Allocate time. 
    # Reserve 15% for polish and overhead.
    sa_time = remaining * 0.80
    polish_time = remaining * 0.10
    
    if sa_time < 1.0:
        # Not enough time for SA, just greedy
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Dynamic number of restarts based on time and sequence length
    # More restarts for shorter sequences
    if n <= 30:
        num_restarts = 5
    elif n <= 50:
        num_restarts = 3
    else:
        num_restarts = 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 10.0:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine with SA
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    # Final Polish with more rounds
    if time.time() - _GLOBAL_START < _TOTAL_BUDGET - 5.0:  # Leave 5s margin
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=300)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
