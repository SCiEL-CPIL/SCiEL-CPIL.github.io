import random
import time
import sys
import math

# Global time tracking
_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin from 120s

# EVOLVE-BLOCK-START

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

def _greedy_grow(sequence, rng=None, temperature=0.0):
    """Fast greedy growth to get an initial conformation."""
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
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
            # Also consider future potential: prefer sites with more open neighbors
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            # Combine gain and openness
            score = gain * 10 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        # Pick best candidate with some randomness based on temperature
        if temperature > 0 and len(candidates) > 1:
            # Softmax selection
            scores = [c[0] for c in candidates]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumsum = 0
            chosen = candidates[-1][1]
            for idx, es in enumerate(exp_scores):
                cumsum += es
                if r <= cumsum:
                    chosen = candidates[idx][1]
                    break
        else:
            candidates.sort(key=lambda x: -x[0])
            chosen = candidates[0][1]
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain, _energy(sequence, chain)

def _greedy_grow_compact(sequence, rng=None):
    """Greedy growth that prefers compact conformations."""
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
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
            # Prefer sites closer to center of mass
            if len(chain) > 1:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy)
            else:
                dist = 0
            # Compactness bonus: prefer sites near existing chain
            near_count = 0
            for q in _neighbours(site):
                if q in occupied:
                    near_count += 1
            score = gain * 8 + near_count * 3 - dist * 0.1
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        candidates.sort(key=lambda x: -x[0])
        chosen = candidates[0][1]
        chain.append(chosen)
        occupied[chosen] = i
    
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
        
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain

def _local_search(sequence, chain, deadline, rng, aggressive=False):
    """
    Local search with single-residue moves, 2-residue moves, and segment moves.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    if aggressive:
        T = 3.0
        cooling = 0.999
        min_T = 0.1
    else:
        T = 1.5
        cooling = 0.9995
        min_T = 0.05
    
    start_time = time.time()
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Choose move type
        r = rng.random()
        if r < 0.4 and n > 5:
            # Single residue move
            i = rng.randint(1, n - 2)
            
            if i == 0 or i == n - 1:
                # End move
                neighbor_idx = 1 if i == 0 else n - 2
                neighbor_pos = current_chain[neighbor_idx]
                old_pos = current_chain[i]
                
                # Candidates are neighbors of neighbor_pos, excluding old_pos and other occupied
                candidates = []
                for nb in _neighbours(neighbor_pos):
                    if nb == old_pos:
                        continue
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
                
                if not candidates:
                    continue
                
                new_pos = rng.choice(candidates)
                
                # Calculate delta
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    # Accept
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
                # Internal move
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                old_pos = current_chain[i]
                
                # New pos must be neighbor of prev_pos AND next_pos
                # And not occupied (except by i)
                candidates = []
                for nb in _neighbours(prev_pos):
                    if nb == old_pos:
                        continue
                    if nb in _neighbours(next_pos):
                        if nb in pos_dict and pos_dict[nb] != i:
                            continue
                        candidates.append(nb)
                
                if not candidates:
                    continue
                
                new_pos = rng.choice(candidates)
                
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    # Accept
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
        elif r < 0.6 and n > 6:
            # 2-residue move: move two adjacent residues
            i = rng.randint(1, n - 3)
            i2 = i + 1
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i2 + 1]
            old_pos1 = current_chain[i]
            old_pos2 = current_chain[i2]
            
            # Find new positions for both residues
            # pos1 must be neighbor of prev_pos, pos2 must be neighbor of next_pos
            # pos1 and pos2 must be adjacent, and neither can be occupied
            candidates = []
            for nb1 in _neighbours(prev_pos):
                if nb1 == old_pos1:
                    continue
                if nb1 in pos_dict and pos_dict[nb1] not in (i, i2):
                    continue
                for nb2 in _neighbours(next_pos):
                    if nb2 == old_pos2:
                        continue
                    if nb2 in pos_dict and pos_dict[nb2] not in (i, i2):
                        continue
                    # Check adjacency between nb1 and nb2
                    d = abs(nb1[0] - nb2[0]) + abs(nb1[1] - nb2[1])
                    if d == 1:
                        candidates.append((nb1, nb2))
            
            if not candidates:
                continue
            
            new_pos1, new_pos2 = rng.choice(candidates)
            
            # Compute energy delta
            delta = 0
            if sequence[i] == 'H':
                # Remove old contacts for residue i
                for q in _neighbours(old_pos1):
                    j = pos_dict.get(q)
                    if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i) > 1:
                        delta += 1
                # Add new contacts for residue i
                for q in _neighbours(new_pos1):
                    j = pos_dict.get(q)
                    if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i) > 1:
                        delta -= 1
            if sequence[i2] == 'H':
                # Remove old contacts for residue i2
                for q in _neighbours(old_pos2):
                    j = pos_dict.get(q)
                    if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i2) > 1:
                        delta += 1
                # Add new contacts for residue i2
                for q in _neighbours(new_pos2):
                    j = pos_dict.get(q)
                    if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i2) > 1:
                        delta -= 1
            # Check if new positions create a contact between i and i2 (shouldn't count)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept
                del pos_dict[old_pos1]
                del pos_dict[old_pos2]
                current_chain[i] = new_pos1
                current_chain[i2] = new_pos2
                pos_dict[new_pos1] = i
                pos_dict[new_pos2] = i2
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Segment move
            max_seg = min(n // 2, 12)
            if max_seg < 2:
                max_seg = 2
            seg_len = rng.randint(2, max_seg)
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            
            head = current_chain[:i]
            tail = current_chain[j+1:]
            
            # Build occupied set for head and tail
            occupied_ht = set(head) | set(tail)
            
            # Find valid start positions for the segment
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
            
            # Check connection to tail
            if tail:
                tail_start = tail[0]
                d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
                if d != 1:
                    continue
            
            new_chain = head + new_segment + tail
            
            # Verify self-avoidance
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
        
        # Restart if stuck too long
        if no_improve_count > 300:
            T = max(T * 1.5, 1.0)
            no_improve_count = 0
    
    return best_chain, best_energy

def _polish_pass(sequence, chain, deadline, rng):
    """
    A high-precision polish pass using only improving moves (hill climbing).
    This is used after the main local search to squeeze out last contacts.
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try all single-residue moves that improve energy
        for i in range(n):
            if i == 0 or i == n - 1:
                neighbor_idx = 1 if i == 0 else n - 2
                neighbor_pos = current_chain[neighbor_idx]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(neighbor_pos):
                    if nb == old_pos:
                        continue
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
                
                for new_pos in candidates:
                    delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                    if delta < 0:
                        # Accept improving move
                        del pos_dict[old_pos]
                        current_chain[i] = new_pos
                        pos_dict[new_pos] = i
                        current_energy += delta
                        improved = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break
            else:
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(prev_pos):
                    if nb == old_pos:
                        continue
                    if nb in _neighbours(next_pos):
                        if nb in pos_dict and pos_dict[nb] != i:
                            continue
                        candidates.append(nb)
                
                for new_pos in candidates:
                    delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                    if delta < 0:
                        # Accept improving move
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
            no_improve += 1
        else:
            no_improve = 0
    
    # Second pass: try 2-residue improving moves
    no_improve2 = 0
    while time.time() < deadline and no_improve2 < 50:
        improved2 = False
        for i in range(1, n - 2):
            i2 = i + 1
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i2 + 1]
            old_pos1 = current_chain[i]
            old_pos2 = current_chain[i2]
            
            candidates = []
            for nb1 in _neighbours(prev_pos):
                if nb1 == old_pos1:
                    continue
                if nb1 in pos_dict and pos_dict[nb1] not in (i, i2):
                    continue
                for nb2 in _neighbours(next_pos):
                    if nb2 == old_pos2:
                        continue
                    if nb2 in pos_dict and pos_dict[nb2] not in (i, i2):
                        continue
                    d = abs(nb1[0] - nb2[0]) + abs(nb1[1] - nb2[1])
                    if d == 1:
                        candidates.append((nb1, nb2))
            
            for new_pos1, new_pos2 in candidates:
                delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos1):
                        j = pos_dict.get(q)
                        if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    for q in _neighbours(new_pos1):
                        j = pos_dict.get(q)
                        if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1
                if sequence[i2] == 'H':
                    for q in _neighbours(old_pos2):
                        j = pos_dict.get(q)
                        if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i2) > 1:
                            delta += 1
                    for q in _neighbours(new_pos2):
                        j = pos_dict.get(q)
                        if j is not None and j not in (i, i2) and sequence[j] == 'H' and abs(j - i2) > 1:
                            delta -= 1
                
                if delta < 0:
                    del pos_dict[old_pos1]
                    del pos_dict[old_pos2]
                    current_chain[i] = new_pos1
                    current_chain[i2] = new_pos2
                    pos_dict[new_pos1] = i
                    pos_dict[new_pos2] = i2
                    current_energy += delta
                    improved2 = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break
        
        if not improved2:
            no_improve2 += 1
        else:
            no_improve2 = 0
    
    return best_chain, best_energy

def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation with more diversity
        if r % 5 == 0:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 1:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=2.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 2:
            init_chain, init_energy = _greedy_grow_compact(sequence, rng)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 5 == 3:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        else:
            # Greedy with higher temperature
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=5.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        
        # Run local search with more aggressive settings for later restarts
        aggressive = r >= n_restarts // 2
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng, aggressive=aggressive)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    global _GLOBAL_DEADLINE
    start_time = time.time()
    n = len(sequence)
    
    # Use global deadline to ensure we finish within the total budget
    deadline = min(start_time + 11.0, _GLOBAL_DEADLINE - 1.0)
    
    if n == 0:
        return []
    
    # Number of restarts - scale with sequence length and available time
    available_time = deadline - start_time
    if n <= 30:
        n_restarts = min(30, int(available_time * 2.5))
    elif n <= 50:
        n_restarts = min(25, int(available_time * 2.0))
    else:
        n_restarts = min(20, int(available_time * 1.5))
    n_restarts = max(5, n_restarts)
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    # Final polish pass
    polish_deadline = deadline - 0.1
    if time.time() < polish_deadline:
        best_chain, best_energy = _polish_pass(sequence, best_chain, polish_deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
