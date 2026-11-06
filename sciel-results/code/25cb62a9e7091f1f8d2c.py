"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 190.0  # leave 20s buffer for overhead/sandbox


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 (2D) or 6 (3D) neighbours of a lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 1  # invalid
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta_move(sequence, chain, i, new_pos):
    """
    Compute energy change when moving residue i to new_pos.
    Returns (delta_energy, valid) where valid indicates no overlap.
    """
    n = len(chain)
    old_pos = chain[i]
    
    # Check if new_pos is occupied by any residue other than i-1, i, i+1
    # (adjacent residues are allowed to be at new_pos only if they're the neighbors)
    # Actually, new_pos must not be occupied by ANY residue except potentially i-1 or i+1
    # But if i-1 or i+1 is at new_pos, that would create a zero-length edge which is invalid
    # So new_pos must not be in the chain at all (except we're removing i)
    
    # Build set of occupied positions excluding i
    occupied = set()
    for j, p in enumerate(chain):
        if j != i:
            occupied.add(p)
    
    if new_pos in occupied:
        return 0, False
    
    # Also check that new_pos is unit distance from i-1 and i+1
    if i > 0:
        prev = chain[i-1]
        dist = sum(abs(new_pos[d] - prev[d]) for d in range(len(prev)))
        if dist != 1:
            return 0, False
    if i < n - 1:
        nxt = chain[i+1]
        dist = sum(abs(new_pos[d] - nxt[d]) for d in range(len(nxt)))
        if dist != 1:
            return 0, False
    
    # Compute energy delta
    delta = 0
    is_h = sequence[i] == 'H'
    
    # Remove contacts involving old position
    if is_h:
        for q in _neighbours(old_pos):
            # Find index at q
            for j, p in enumerate(chain):
                if j != i and p == q and sequence[j] == 'H':
                    if abs(j - i) > 1:
                        delta += 1  # removing a contact increases energy by 1
                    break
    
    # Add contacts involving new position
    if is_h:
        for q in _neighbours(new_pos):
            for j, p in enumerate(chain):
                if j != i and p == q and sequence[j] == 'H':
                    if abs(j - i) > 1:
                        delta -= 1  # adding a contact decreases energy by 1
                    break
    
    return delta, True


def _greedy_growth(sequence, dimension, rng, temperature=0.5):
    """
    Grow a chain on the lattice using a probabilistic greedy approach.
    At each step, choose among valid sites based on:
      - Number of new H-H contacts formed
      - Number of free neighbours (to avoid dead-ends)
      - Random noise
    Returns a chain (list of tuples) or None if stuck.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        res = sequence[i]
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this site would form
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free neighbours (avoid trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: favor contacts strongly, then free space, plus noise
            score = gain * 3.0 + 0.3 * free + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        # Choose best option with some randomness
        options.sort(key=lambda x: -x[0])
        if temperature > 0 and len(options) > 1:
            # Softmax-like selection
            top = options[0][0]
            probs = []
            total = 0.0
            for sc, _ in options[:min(4, len(options))]:
                p = math.exp((sc - top) / max(temperature, 0.1))
                probs.append(p)
                total += p
            if total > 0:
                probs = [p / total for p in probs]
                r = rng.random()
                cum = 0.0
                chosen_idx = len(probs) - 1
                for idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen_idx = idx
                        break
                site = options[chosen_idx][1]
            else:
                site = options[0][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain


def _local_search(sequence, chain, dimension, rng, max_iter=200):
    """
    Improve a valid chain via local moves:
      - Single residue move
      - Segment rotation (rotate a short segment 90 degrees)
      - Diagonal shift (move a residue diagonally via two half-steps conceptually)
    Returns the improved chain and energy.
    """
    if chain is None:
        return None, 1

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    
    # Precompute position lookup for faster overlap checks
    def _make_pos_map(ch):
        return {p: i for i, p in enumerate(ch)}
    
    pos_map = _make_pos_map(current_chain)

    for it in range(max_iter):
        if _remaining() < 0.5:
            break

        # Choose move type
        move_type = rng.random()
        
        best_delta = 0  # We want to minimize energy, so accept negative deltas
        best_new_chain = None
        best_new_pos = None
        best_i = -1
        
        if move_type < 0.6:
            # Single residue move
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                # Check if q is occupied
                if q in pos_map and pos_map[q] != i:
                    continue
                # Check unit steps
                prev = current_chain[i-1]
                nxt = current_chain[i+1]
                if sum(abs(q[d] - prev[d]) for d in range(dimension)) != 1:
                    continue
                if sum(abs(q[d] - nxt[d]) for d in range(dimension)) != 1:
                    continue
                
                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    # Add new contacts
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = q
        
        elif move_type < 0.85:
            # Segment rotation: pick a segment of length 2-4 and rotate it
            seg_len = rng.choice([2, 3, 4])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                # Segment is from i to i+seg_len-1
                # The pivot is at i-1 (fixed) and i+seg_len (fixed)
                # Try rotating the segment 90 degrees around the connection point
                
                # Try shifting the entire segment by one step perpendicular
                # Get the direction from i-1 to i
                d1 = [current_chain[i][d] - current_chain[i-1][d] for d in range(dimension)]
                # Perpendicular directions in 2D
                if dimension == 2:
                    # Rotate d1 by 90 degrees
                    perp = [(-d1[1], d1[0])]
                    for px, py in perp:
                        # Shift all segment residues by (px, py)
                        new_seg = []
                        valid = True
                        for k in range(seg_len):
                            new_p = (current_chain[i+k][0] + px, current_chain[i+k][1] + py)
                            new_seg.append(new_p)
                        
                        # Check if new positions are free
                        for new_p in new_seg:
                            j = pos_map.get(new_p)
                            if j is not None and not (i <= j < i + seg_len):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Check unit steps within segment and connections
                        # Connection from i-1 to new segment start
                        if sum(abs(new_seg[0][d] - current_chain[i-1][d]) for d in range(dimension)) != 1:
                            continue
                        # Connection from new segment end to i+seg_len
                        if sum(abs(new_seg[-1][d] - current_chain[i+seg_len][d]) for d in range(dimension)) != 1:
                            continue
                        # Internal unit steps
                        for k in range(seg_len - 1):
                            if sum(abs(new_seg[k+1][d] - new_seg[k][d]) for d in range(dimension)) != 1:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute energy delta
                        delta = 0
                        for k in range(seg_len):
                            old_p = current_chain[i+k]
                            new_p = new_seg[k]
                            if sequence[i+k] == 'H':
                                # Remove old contacts
                                for q in _neighbours(old_p):
                                    j = pos_map.get(q)
                                    if j is not None and j != i+k and sequence[j] == 'H' and abs(j - (i+k)) > 1:
                                        # Check if j is in the segment (will be moved)
                                        if not (i <= j < i + seg_len):
                                            delta += 1
                                # Add new contacts
                                for q in _neighbours(new_p):
                                    j = pos_map.get(q)
                                    if j is not None and j != i+k and sequence[j] == 'H' and abs(j - (i+k)) > 1:
                                        if not (i <= j < i + seg_len):
                                            delta -= 1
                        # Also check new contacts between segment residues and themselves
                        # (these are internal, but we need to check H-H contacts within the moved segment)
                        # Actually, contacts within the segment that were already there stay the same
                        # Only contacts with outside residues change, which we've accounted for
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_new_pos = new_seg  # Store as list
        
        else:
            # Diagonal move: move a residue by (1,1) or similar, but this isn't a valid single step
            # Instead, try a "kink" move: move residue i to a position that's 2 steps away but
            # adjust the path. For simplicity, just do a single move with different sampling.
            i = rng.randint(1, n - 2)
            p = current_chain[i]
            # Try all 4 neighbours with higher acceptance
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue
                prev = current_chain[i-1]
                nxt = current_chain[i+1]
                if sum(abs(q[d] - prev[d]) for d in range(dimension)) != 1:
                    continue
                if sum(abs(q[d] - nxt[d]) for d in range(dimension)) != 1:
                    continue
                
                delta = 0
                if sequence[i] == 'H':
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = q

        # Apply best move if it improves energy
        if best_delta < 0 and best_i >= 0:
            if isinstance(best_new_pos, list):
                # Segment move
                for k in range(len(best_new_pos)):
                    old_p = current_chain[best_i + k]
                    del pos_map[old_p]
                    current_chain[best_i + k] = best_new_pos[k]
                    pos_map[best_new_pos[k]] = best_i + k
            else:
                # Single move
                old_p = current_chain[best_i]
                del pos_map[old_p]
                current_chain[best_i] = best_new_pos
                pos_map[best_new_pos] = best_i
            
            current_energy += best_delta
        
        else:
            # No improving move found; try a random perturbation to escape local minima
            if rng.random() < 0.3:
                i = rng.randint(1, n - 2)
                p = current_chain[i]
                for q in _neighbours(p):
                    if q == current_chain[i - 1] or q == current_chain[i + 1]:
                        continue
                    if q in pos_map and pos_map[q] != i:
                        continue
                    prev = current_chain[i-1]
                    nxt = current_chain[i+1]
                    if sum(abs(q[d] - prev[d]) for d in range(dimension)) != 1:
                        continue
                    if sum(abs(q[d] - nxt[d]) for d in range(dimension)) != 1:
                        continue
                    
                    # Accept with simulated annealing
                    delta = 0
                    if sequence[i] == 'H':
                        for old_q in _neighbours(p):
                            j = pos_map.get(old_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta += 1
                        for new_q in _neighbours(q):
                            j = pos_map.get(new_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta -= 1
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / max(1.0, (max_iter - it) / max_iter * 5.0)):
                        old_p = current_chain[i]
                        del pos_map[old_p]
                        current_chain[i] = q
                        pos_map[q] = i
                        current_energy += delta
                        break

    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses multiple greedy growths with varying temperatures, followed by local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 1  # worst possible (unfolded has 0 contacts, energy 0; we want negative)

    # Phase 1: Greedy growths with varying parameters
    temps = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]

    # Do multiple restarts, scaled by sequence length
    if n <= 25:
        max_restarts = 800
        ls_iter = 100
    elif n <= 40:
        max_restarts = 400
        ls_iter = 70
    elif n <= 60:
        max_restarts = 200
        ls_iter = 50
    else:
        max_restarts = 100
        ls_iter = 35

    for restart in range(max_restarts):
        if _remaining() < 1.5:
            break

        # Vary temperature
        temp = temps[restart % len(temps)] + rng.random() * 0.3
        temp = min(temp, 2.5)

        chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_chain = chain
            best_energy = energy

        # Phase 2: Local search improvement
        if _remaining() < 2.0:
            break

        improved_chain, improved_energy = _local_search(
            sequence, chain, dimension, rng,
            max_iter=ls_iter
        )

        if improved_chain is not None and improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # If no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
