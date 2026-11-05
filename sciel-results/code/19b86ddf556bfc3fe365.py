"""HP lattice protein folding on the square lattice using model-based optimization with crankshaft, pivot, and two-point moves."""
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
    h_indices = set(i for i, s in enumerate(sequence) if s == "H")
    contacts = 0
    for i in h_indices:
        p = chain[i]
        x, y = p
        # Check 4 neighbours
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_indices:
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
            # Add a small random perturbation to avoid deterministic traps
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

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
    
    # Build set of occupied points in prefix and suffix
    fixed_occupied = set(prefix) | set(suffix)
    
    new_seg = []
    for j in range(len(seg)):
        ox = seg[j][0] - px
        oy = seg[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_seg.append(new_pos)
    
    # Check for collisions with fixed parts
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

def _try_two_point_move(sequence, chain, i, j, rot1, rot2):
    """
    Attempt a two-point move:
    1. Rotate segment [i+1..j] around chain[i] by rot1.
    2. Rotate segment [j+1..n-1] around the new position of j by rot2.
    rot1, rot2: 0=90deg, 1=180deg, 2=270deg
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if i >= j or j >= n - 1:
        return None, None
        
    # Define rotation functions
    def get_rot(r):
        if r == 0:
            return lambda x, y: (-y, x)
        elif r == 1:
            return lambda x, y: (-x, -y)
        else:
            return lambda x, y: (y, -x)
            
    rot1_fn = get_rot(rot1)
    rot2_fn = get_rot(rot2)
    
    # Part 1: Rotate middle segment [i+1..j] around chain[i]
    px1, py1 = chain[i]
    mid_seg = chain[i+1:j+1]
    
    new_mid = []
    for p in mid_seg:
        ox = p[0] - px1
        oy = p[1] - py1
        nx, ny = rot1_fn(ox, oy)
        new_mid.append((px1 + nx, py1 + ny))
        
    # Check mid_seg collisions with prefix [0..i]
    prefix_occupied = set(chain[:i+1])
    for p in new_mid:
        if p in prefix_occupied:
            return None, None
            
    # Check mid_seg self-collisions
    if len(set(new_mid)) != len(new_mid):
        return None, None
        
    # The new position of residue j
    new_j_pos = new_mid[-1]
    
    # Part 2: Rotate tail segment [j+1..n-1] around new_j_pos
    tail_seg = chain[j+1:]
    new_tail = []
    for p in tail_seg:
        ox = p[0] - new_j_pos[0]
        oy = p[1] - new_j_pos[1]
        nx, ny = rot2_fn(ox, oy)
        new_tail.append((new_j_pos[0] + nx, new_j_pos[1] + ny))
        
    # Check tail collisions with prefix and new_mid
    fixed_occupied = prefix_occupied | set(new_mid)
    for p in new_tail:
        if p in fixed_occupied:
            return None, None
            
    # Check tail self-collisions
    if len(set(new_tail)) != len(new_tail):
        return None, None
        
    # Construct new chain
    new_chain = chain[:i+1] + new_mid + new_tail
    
    # Final verification of unit steps and self-avoidance (safety check)
    seen = set()
    for idx, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if idx > 0:
            dx = abs(new_chain[idx][0] - new_chain[idx-1][0])
            dy = abs(new_chain[idx][1] - new_chain[idx-1][1])
            if dx + dy != 1:
                return None, None
                
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot, crankshaft, and two-point moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T_start = 2.5
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        r = rng.random()
        
        if r < 0.2:
            # Two-point move
            # Pick i, j such that i < j < n-1
            # To ensure valid segments, i in [0, n-3], j in [i+1, n-2]
            if n > 5:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 1, n - 2)
                rot1 = rng.randint(0, 2)
                rot2 = rng.randint(0, 2)
                new_chain, new_energy = _try_two_point_move(sequence, current_chain, i, j, rot1, rot2)
            else:
                new_chain, new_energy = None, None
        elif r < 0.5:
            # Crankshaft
            if n > 6:
                k1 = rng.randint(0, n - 3)
                k2 = rng.randint(k1 + 2, n - 1)
                rot_choice = rng.randint(0, 2)
                new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
            else:
                new_chain, new_energy = None, None
        else:
            # Pivot
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

def _greedy_polish(sequence, chain, max_rounds=100, deadline=None):
    """
    Greedy polish: repeatedly scan all possible pivot and crankshaft moves and accept any that improve energy.
    Stop when no move improves energy, max_rounds reached, or deadline exceeded.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
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
    # Reserve 10% for polish and overhead.
    sa_time = remaining * 0.85
    polish_time = remaining * 0.05
    
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
        num_restarts = 4
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
        
    # Final Polish: high-precision pass with more rounds and a tighter deadline guard
    remaining_for_polish = _TOTAL_BUDGET - 3.0 - (time.time() - _GLOBAL_START)
    if remaining_for_polish > 1.0:
        # Use more rounds for the final polish to escape shallow local minima
        polish_rounds = 300 if n <= 40 else 200
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 3.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
