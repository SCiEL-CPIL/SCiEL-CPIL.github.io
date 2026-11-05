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
    h_indices = set(i for i, s in enumerate(sequence) if s == "H")
    contacts = 0
    for i in h_indices:
        p = chain[i]
        x, y = p
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_indices:
                contacts += 1
    return -contacts

def _try_pinch_move(sequence, chain, i, target_j):
    """
    Attempt to move residue i to a position adjacent to residue target_j,
    specifically trying to form a new H-H contact.
    Only works if both are H residues and they're not already adjacent in the chain.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    if sequence[i] != "H" or sequence[target_j] != "H":
        return None, None
    if abs(i - target_j) <= 1:
        return None, None
    
    n = len(chain)
    pi = chain[i]
    ptj = chain[target_j]
    
    tx, ty = ptj
    candidates = [(tx+1, ty), (tx-1, ty), (tx, ty+1), (tx, ty-1)]
    
    occupied = set(chain)
    occupied.discard(pi)
    
    for new_pos in candidates:
        if new_pos in occupied or new_pos == pi:
            continue
        
        new_chain = list(chain)
        new_chain[i] = new_pos
        
        if len(set(new_chain)) != n:
            continue
        
        valid = True
        for idx in range(n - 1):
            if abs(new_chain[idx][0] - new_chain[idx+1][0]) + abs(new_chain[idx][1] - new_chain[idx+1][1]) != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        return new_chain, new_energy
    
    return None, None

def _try_two_point_swap(sequence, chain, i, j):
    """
    Attempt to swap residues i and j in the chain.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    if abs(i - j) <= 1:
        return None, None
    
    n = len(chain)
    pi = chain[i]
    pj = chain[j]
    
    # Check if swapping maintains unit steps
    # For i: neighbors are chain[i-1] and chain[i+1]
    # For j: neighbors are chain[j-1] and chain[j+1]
    
    # After swap, chain[i] = pj, chain[j] = pi
    # Need to verify:
    # 1. chain[i-1] -> pj is unit step (if i > 0)
    # 2. pj -> chain[i+1] is unit step (if i < n-1)
    # 3. chain[j-1] -> pi is unit step (if j > 0)
    # 4. pi -> chain[j+1] is unit step (if j < n-1)
    
    if i > 0:
        dx = abs(pj[0] - chain[i-1][0]) + abs(pj[1] - chain[i-1][1])
        if dx != 1:
            return None, None
    if i < n - 1:
        dx = abs(chain[i+1][0] - pj[0]) + abs(chain[i+1][1] - pj[1])
        if dx != 1:
            return None, None
    if j > 0:
        dx = abs(pi[0] - chain[j-1][0]) + abs(pi[1] - chain[j-1][1])
        if dx != 1:
            return None, None
    if j < n - 1:
        dx = abs(chain[j+1][0] - pi[0]) + abs(chain[j+1][1] - pi[1])
        if dx != 1:
            return None, None
    
    # Check self-avoidance after swap
    new_chain = list(chain)
    new_chain[i], new_chain[j] = pj, pi
    
    if len(set(new_chain)) != n:
        return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy

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

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot, crankshaft, pinch, and two-point swap moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute H indices for pinch moves
    h_indices = [i for i, s in enumerate(sequence) if s == "H"]
    
    # Simulated Annealing parameters
    T_start = 2.0
    T_end = 0.003
    start_time = time.time()
    
    max_iters = 250000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type: 40% pivot, 20% crankshaft, 20% pinch, 20% two-point swap
        move_type = rng.random()
        
        if move_type < 0.4:
            # Pivot move
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        elif move_type < 0.6 and n > 6:
            # Crankshaft move
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        elif move_type < 0.8:
            # Pinch move: try to form H-H contact
            if len(h_indices) >= 2:
                i = h_indices[rng.randint(0, len(h_indices) - 1)]
                target_j = h_indices[rng.randint(0, len(h_indices) - 1)]
                new_chain, new_energy = _try_pinch_move(sequence, current_chain, i, target_j)
            else:
                # Fallback to pivot if not enough H residues
                k = rng.randint(0, n - 2)
                rot_choice = rng.randint(0, 2)
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        else:
            # Two-point swap move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if abs(i - j) > 1:
                new_chain, new_energy = _try_two_point_swap(sequence, current_chain, i, j)
            else:
                new_chain, new_energy = None, None
        
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
    Greedy polish: repeatedly scan all possible pivot, crankshaft, pinch, and two-point swap moves and accept any that improve energy.
    Stop when no move improves energy, max_rounds reached, or deadline exceeded.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    h_indices = [i for i, s in enumerate(sequence) if s == "H"]
    
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
        if not improved and n > 6:
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
        
        # Try pinch moves
        if not improved and len(h_indices) >= 2:
            for i in h_indices:
                for j in h_indices:
                    if abs(i - j) > 1:
                        new_chain, new_energy = _try_pinch_move(sequence, current_chain, i, j)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                if improved:
                    break
        
        # Try two-point swap moves
        if not improved:
            for i in range(n):
                for j in range(i + 2, n):
                    new_chain, new_energy = _try_two_point_swap(sequence, current_chain, i, j)
                    if new_chain is not None and new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
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
        num_restarts = 5
    elif n <= 50:
        num_restarts = 4
    else:
        num_restarts = 3
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
        polish_rounds = 600 if n <= 40 else 400
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 3.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
        
        # Second polish pass with even more rounds if time remains
        remaining_for_polish2 = _TOTAL_BUDGET - 3.0 - (time.time() - _GLOBAL_START)
        if remaining_for_polish2 > 1.0:
            polish_rounds2 = 500 if n <= 40 else 350
            polished_chain2, polished_energy2 = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds2, deadline=_TOTAL_BUDGET - 3.0)
            if polished_energy2 < best_energy:
                best_chain = polished_chain2
                best_energy = polished_energy2
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
