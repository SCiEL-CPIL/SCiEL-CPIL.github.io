import random
import time
import sys
import math

# EVOLVE-BLOCK-START

# Global time management
_TOTAL_BUDGET = 115.0  # Leave 5s safety margin for 120s limit
_START_TIME = None
_sequences_done = 0
_total_sequences = 10  # Known from task description

def _get_time_budget():
    """
    Distribute remaining time among remaining sequences.
    Returns time limit for the current sequence.
    """
    global _START_TIME, _sequences_done
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    remaining = _TOTAL_BUDGET - elapsed
    remaining_seqs = _total_sequences - _sequences_done
    
    if remaining_seqs <= 0:
        return 0.1 # Minimal time if we are over budget or done
    
    # Allocate proportional time, with a minimum of 2s per sequence to avoid starvation
    # and a maximum to prevent one sequence from hogging time if others are fast
    per_seq = remaining / remaining_seqs
    # Cap at 15s per sequence to ensure we don't overshoot if early ones are fast
    # and floor at 1s to ensure we actually try
    return max(1.0, min(15.0, per_seq))

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos):
    """
    Compute the change in energy when moving residue i from old_pos to new_pos.
    pos_dict maps position -> index.
    """
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    
    # Remove contacts with old_pos
    for q in _neighbours(old_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1  # Losing a contact increases energy by 1
            
    # Add contacts with new_pos
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1  # Gaining a contact decreases energy by 1
            
    return delta

def _greedy_grow(sequence, rng=None, compactness=1.0):
    """
    Fast greedy growth. 
    compactness > 1 favors positions closer to the centroid of existing chain.
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Track centroid for compactness
    sum_x, sum_y = 0, 0
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Compactness: distance to centroid
            dist_to_centroid = math.hypot(site[0] - (sum_x / i), site[1] - (sum_y / i))
            
            # Openness: how many free neighbors
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            
            # Score: Gain is most important, then compactness (lower dist is better), then openness
            # We want to minimize dist, so we subtract it
            score = gain * 10 - dist_to_centroid * compactness + open_count * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        if rng is not None and len(candidates) > 1 and rng.random() < 0.1:
            # Small chance of random choice to escape local traps early
            best_site = rng.choice(candidates)[1]
        else:
            candidates.sort(key=lambda x: -x[0])
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
        sum_x += best_site[0]
        sum_y += best_site[1]
    
    return chain, _energy(sequence, chain)

def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in dirs:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        if not candidates:
            return None
        new_pos = rng.choice(candidates)
        chain.append(new_pos)
        occupied.add(new_pos)
    return chain

def _pinch_move(sequence, chain, pos_dict, rng):
    """
    Attempt a pinch move: find two H residues that are close but not adjacent,
    and move a segment to bring them together.
    Returns (new_chain, new_energy, success)
    """
    n = len(chain)
    if n < 5:
        return None, 0, False
    
    # Find H residues
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    if len(h_indices) < 2:
        return None, 0, False
    
    # Pick a random pair of H residues that are not adjacent
    # To be efficient, we don't check all pairs, just a random sample
    # Or better, check pairs within a small index range or spatial range
    # Here we pick a random H and look for another H nearby
    
    # Optimization: Instead of all pairs, pick a random H residue i
    i = rng.choice(h_indices)
    pos_i = chain[i]
    
    # Find other H residues within Manhattan distance 4
    candidates_j = []
    for j in h_indices:
        if abs(i - j) <= 1:
            continue
        pos_j = chain[j]
        dist = abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1])
        if 2 <= dist <= 4:
            candidates_j.append(j)
            
    if not candidates_j:
        return None, 0, False
    
    j = rng.choice(candidates_j)
    
    # We want to move the segment between i and j (or j and i) to bring them closer
    # Assume i < j for simplicity, if not swap
    if i > j:
        i, j = j, i
        
    # The segment is from i+1 to j-1
    # We can try to move this segment to a new location that brings i and j closer
    # This is complex. A simpler "pinch" is to move a single residue in the segment
    # or a small sub-segment.
    
    # Let's try moving a single residue k in (i, j) to a position closer to i or j
    # Actually, a true pinch usually involves moving the whole segment.
    # Given complexity, let's implement a "segment re-insertion" move which is similar
    # but specifically targets closing the gap.
    
    # For now, let's use a simpler heuristic:
    # If i and j are distance 2 apart, there is exactly one spot between them.
    # If distance 3, there are spots.
    
    # Let's try to move residue i+1 or j-1 to a position that creates a contact with the other end.
    # This is essentially a local move but guided.
    
    # Alternative: Try to move the segment [i+1, j-1] to a new location.
    # This is expensive. Let's stick to the existing segment move in _local_search
    # but bias the selection of i, j for segment moves to be H-H pairs.
    
    # Since implementing a full segment pinch is risky for time, 
    # we will return failure here and let _local_search handle segment moves 
    # with a bias towards H-rich segments.
    return None, 0, False

def _local_search(sequence, chain, deadline, rng):
    """
    Local search with single-residue moves and segment moves.
    """
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.5
    
    start_time = time.time()
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Choose move type
        r = rng.random()
        if r < 0.5 and n > 5:
            # Single residue move
            i = rng.randint(1, n - 2)
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            old_pos = current_chain[i]
            
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb not in pos_dict or pos_dict[nb] == i:
                        candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                del pos_dict[old_pos]
                current_chain[i] = new_pos
                pos_dict[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Segment move
            max_seg = min(n // 2, 10)
            if max_seg < 2:
                max_seg = 2
            
            # Bias segment selection towards H-rich areas
            # Pick a random start, but if we pick an H, extend to include more Hs if possible
            i = rng.randint(0, n - max_seg)
            # Simple bias: if sequence[i] is H, try to make segment longer
            seg_len = rng.randint(2, max_seg)
            if sequence[i] == 'H' and rng.random() < 0.5:
                seg_len = min(max_seg, seg_len + 2)
                
            j = i + seg_len - 1
            
            head = current_chain[:i]
            tail = current_chain[j+1:]
            
            occupied_ht = set(head) | set(tail)
            
            anchor = head[-1] if head else (0, 0)
            start_candidates = []
            for nb in _neighbours(anchor):
                if nb not in occupied_ht:
                    start_candidates.append(nb)
            
            if not start_candidates:
                continue
            
            start_pos = rng.choice(start_candidates)
            new_segment = [start_pos]
            new_occupied = set(occupied_ht) | {start_pos}
            
            success = True
            for k in range(1, seg_len):
                last_pos = new_segment[-1]
                next_candidates = []
                for nb in _neighbours(last_pos):
                    if nb not in new_occupied:
                        next_candidates.append(nb)
                
                if not next_candidates:
                    success = False
                    break
                
                next_pos = rng.choice(next_candidates)
                new_segment.append(next_pos)
                new_occupied.add(next_pos)
            
            if not success:
                continue
            
            if tail:
                tail_start = tail[0]
                d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
                if d != 1:
                    continue
            
            new_chain = head + new_segment + tail
            
            if len(set(new_chain)) != n:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        if no_improve_count > 500:
            T = max(T * 2.0, 5.0)
            no_improve_count = 0
    
    return best_chain, best_energy

def _polish_pass(sequence, chain, deadline, rng):
    """
    High-precision polish: try all possible single-residue moves and accept improvements.
    """
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            old_pos = current_chain[i]
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb not in pos_dict or pos_dict[nb] == i:
                        candidates.append(nb)
            
            for new_pos in candidates:
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                if delta < 0:
                    del pos_dict[old_pos]
                    current_chain[i] = new_pos
                    pos_dict[new_pos] = i
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
    
        if not improved:
            break
    
    return best_chain, best_energy

def _restart_local_search(sequence, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    
    # Number of restarts depends on remaining time
    # Estimate time per restart: greedy is O(n^2), LS is variable.
    # Let's do at least 5 restarts, and more if time permits.
    
    min_restarts = 5
    max_restarts = 50
    
    # We will loop until deadline, doing a restart each iteration
    # But we need to ensure we don't spend too much time on one LS
    # We will allocate a portion of the remaining time to each LS run
    
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # Calculate time left for this restart's LS phase
        time_left = deadline - time.time()
        if time_left < 0.5:
            break
            
        # Allocate 80% of remaining time to LS, 20% to next restarts
        # But ensure we have at least 0.5s for LS
        ls_time = max(0.5, time_left * 0.7)
        
        # Generate initial conformation
        # Vary compactness and temperature
        compactness = 1.0 + (restart_count % 5) * 0.2
        init_chain, init_energy = _greedy_grow(sequence, rng, compactness=compactness)
        
        if init_chain is None:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        
        # Run local search
        ls_deadline = time.time() + ls_time
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        # Polish pass with remaining time
        if time.time() < deadline - 0.2:
            polish_deadline = deadline - 0.1
            polished_chain, polished_energy = _polish_pass(sequence, ls_chain, polish_deadline, rng)
            if polished_energy < ls_energy:
                ls_chain, ls_energy = polished_chain, polished_energy
        
        if best_chain is None or ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
            
        # If we have done enough restarts and are converging, we can break
        # Simple heuristic: if we haven't improved in 5 restarts, break
        # (We don't track history here, so we just rely on time)
        
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    global _sequences_done
    
    if not sequence:
        return []
        
    # Get time budget for this sequence
    seq_deadline = time.time() + _get_time_budget()
    
    n = len(sequence)
    rng = random.Random(12345 + n + _sequences_done)
    
    best_chain, best_energy = _restart_local_search(sequence, seq_deadline, rng)
    
    _sequences_done += 1
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
