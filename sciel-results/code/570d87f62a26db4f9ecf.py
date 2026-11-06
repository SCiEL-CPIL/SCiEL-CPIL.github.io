"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Heuristic: Strongly prefer gain, slightly prefer compactness
            score = bias * gain * 2.0 + 0.1 * free - 0.05 * dist + rng.random() * 0.2
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _get_fixed_set(chain, pivot_idx):
    return set(chain[:pivot_idx + 1])


def _rotate_segment(segment, pivot, angle):
    rotated = []
    for p in segment:
        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
        if angle == 90:
            ndx, ndy = -dy, dx
        elif angle == -90:
            ndx, ndy = dy, -dx
        else:  # 180
            ndx, ndy = -dx, -dy
        rotated.append((pivot[0] + ndx, pivot[1] + ndy))
    return rotated


def _check_collision_and_delta(sequence, chain, pivot_idx, angle, current_energy):
    """
    Check if a pivot move is valid and calculate the change in energy.
    Returns (new_chain, new_energy) if valid, else (None, None).
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None, None
        
    pivot = chain[pivot_idx]
    segment = chain[pivot_idx + 1:]
    fixed_set = _get_fixed_set(chain, pivot_idx)
    
    # Rotate segment
    rotated_segment = _rotate_segment(segment, pivot, angle)
    
    # Check collision with fixed set
    for p in rotated_segment:
        if p in fixed_set:
            return None, None
            
    # The rotated segment is internally self-avoiding because it's a rotation of a self-avoiding path.
    # We only need to check collisions with the fixed part.
    
    # Calculate energy delta
    # Old cross contacts: contacts between segment and fixed_set
    # New cross contacts: contacts between rotated_segment and fixed_set
    
    # Helper to count H-H contacts between a set of points (segment) and fixed_set
    def count_cross_contacts(seg_pts, fix_pts, seq, start_idx_seg):
        # start_idx_seg is the index of the first point in seg_pts in the full chain
        # We need to check neighbors of seg_pts that are in fix_pts
        count = 0
        # Map fixed points to their sequence indices? 
        # It's faster to just check if the neighbor is in fix_pts and then look up its index in chain?
        # Actually, we can precompute a map of fixed points to indices.
        return count # Placeholder, will implement properly below

    # Let's do it efficiently
    # Build a set of H positions in the fixed part? No, we need indices to check adjacency in chain?
    # No, contact is defined by lattice proximity, not chain adjacency (except j > i+1).
    # Since fixed part and segment are disjoint in chain indices, any lattice contact between them is valid 
    # provided they are not chain-adjacent.
    # The only chain-adjacent pair between fixed and segment is (pivot, segment[0]).
    # So we must exclude the contact between pivot and segment[0] (or rotated[0]) if it exists.
    
    # Let's count all H-H contacts between segment points and fixed points.
    # Then subtract 1 if the specific pair (pivot, seg[0]) or (pivot, rot[0]) is an H-H contact.
    
    # Precompute H indices in fixed part for quick lookup?
    # Let's just iterate over segment points and check their 4 neighbors in fixed_set.
    
    old_cross = 0
    new_cross = 0
    
    # We need to know which indices in the fixed part are H.
    # Let's create a set of H positions in the fixed part.
    fixed_h_set = set()
    for i in range(pivot_idx + 1):
        if sequence[i] == 'H':
            fixed_h_set.add(chain[i])
            
    # Iterate through segment
    for k, p in enumerate(segment):
        # p is at index pivot_idx + 1 + k
        if sequence[pivot_idx + 1 + k] == 'H':
            for q in _neighbours(p):
                if q in fixed_h_set:
                    # Check if this is the chain-adjacent pair (pivot, segment[0])
                    # This only happens if k==0 and q == pivot
                    if k == 0 and q == pivot:
                        continue
                    old_cross += 1
                    
    for k, p in enumerate(rotated_segment):
        if sequence[pivot_idx + 1 + k] == 'H':
            for q in _neighbours(p):
                if q in fixed_h_set:
                    if k == 0 and q == pivot:
                        continue
                    new_cross += 1
                    
    delta_contacts = new_cross - old_cross
    new_energy = current_energy - delta_contacts
    
    new_chain = chain[:pivot_idx + 1] + rotated_segment
    return new_chain, new_energy


def _sa_search(sequence, initial_chain, time_limit, rng):
    """Simulated Annealing with fast energy delta."""
    if initial_chain is None:
        return None, 0

    current_chain = list(initial_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    
    # Temperature schedule
    T = 2.0
    cooling = 0.999
    min_T = 0.05
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # If no improvement for this many steps, perturb
    
    while time.time() - start_time < time_limit and _time_left() > 10:
        # Pick a pivot index
        # Bias towards shorter segments? Or uniform?
        # Uniform is standard.
        pivot_idx = rng.randint(1, n - 2)
        angle = rng.choice([90, -90, 180])
        
        new_chain, new_energy = _check_collision_and_delta(sequence, current_chain, pivot_idx, angle, current_energy)
        
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.1)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            T *= cooling
            if T < min_T:
                T = min_T
        else:
            no_improve_count += 1
            
        # Iterated Local Search: if stuck, perturb
        if no_improve_count > max_no_improve:
            # Apply a kick: a few random valid pivots
            for _ in range(3):
                kick_pivot = rng.randint(1, n - 2)
                kick_angle = rng.choice([90, -90, 180])
                kick_chain, kick_energy = _check_collision_and_delta(sequence, current_chain, kick_pivot, kick_angle, current_energy)
                if kick_chain is not None:
                    # Accept kick regardless of energy? Or only if not too bad?
                    # Let's accept it to escape local minima
                    current_chain = kick_chain
                    current_energy = kick_energy
                    T = 1.0 # Reheat
                    no_improve_count = 0
                    break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + SA with delta energy."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 500
        sa_time = 15.0
    elif seq_len <= 50:
        num_grow = 300
        sa_time = 12.0
    elif seq_len <= 70:
        num_grow = 200
        sa_time = 10.0
    else:
        num_grow = 100
        sa_time = 8.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth
    # Use a range of biases to explore different compactness levels
    # We store the top few initial chains to potentially restart SA from different points if time permits
    initial_chains = []
    
    for i in range(num_grow):
        if _time_left() < 30:
            break

        # Bias between 0.5 and 3.0
        bias = 0.5 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
            
        # Keep top 5 initial chains for diversity in SA
        if len(initial_chains) < 5:
            initial_chains.append((chain, value))
        else:
            # Replace worst if better
            if value < initial_chains[-1][1]:
                initial_chains[-1] = (chain, value)
                # Sort to keep best at end? Or just keep a heap?
                # For simplicity, just keep a list and sort if we use them
                pass

    # Pass 2: SA on the best chain
    if best_chain is not None and _time_left() > 40:
        # Run SA on the best initial chain
        improved, improved_energy = _sa_search(sequence, best_chain, sa_time, rng)
        if improved is not None:
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # If time permits, run SA on the 2nd best initial chain
        if len(initial_chains) > 1 and _time_left() > 30:
            second_chain, second_energy = initial_chains[1]
            # Only run if the second chain is not too bad (within 2 contacts of best)
            if second_energy - best_energy <= 2:
                improved2, improved2_energy = _sa_search(sequence, second_chain, min(5.0, _time_left() - 20), rng)
                if improved2 is not None and improved2_energy < best_energy:
                    best_energy = improved2_energy
                    best_chain = improved2

    # Pass 3: Final Polish
    if best_chain is not None and _time_left() > 20:
        # Low temperature SA polish
        polish_time = min(5.0, _time_left() - 15)
        if polish_time > 1.0:
            # Use a new RNG for polish to be independent
            rng_polish = random.Random(rng.random() * 1000000)
            polished, polished_energy = _sa_search(sequence, best_chain, polish_time, rng_polish)
            if polished is not None and polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
