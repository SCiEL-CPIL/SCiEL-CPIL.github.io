"""Rewritten program: HP lattice protein folding on cubic lattice."""
import time
import random
import sys

_START_TIME = time.time()
# Total budget for all sequences; leave safety margin
_TOTAL_BUDGET = 240.0
_SAFETY_MARGIN = 15.0


def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME) - _SAFETY_MARGIN


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    d = len(point)
    for axis in range(d):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy: -1 per non-adjacent H-H contact."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _greedy_grow(sequence, dimension, rng, use_local_search=True):
    """
    Grow a chain greedily, then optionally do a local search (Monte Carlo) to improve.
    Returns (chain, energy) or (None, 0) if stuck.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            # Score: prioritize contacts, then free space, then random
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        if not options:
            return None, 0
        options.sort(reverse=True)
        # Pick from top few with some randomness
        k = min(3, len(options))
        _, site = options[rng.randint(0, k - 1)]
        occupied[site] = i
        chain.append(site)
    
    energy = _energy(sequence, chain)
    
    if use_local_search and _time_left() > 5:
        energy = _local_search(sequence, chain, energy, rng)
    
    return chain, energy


def _local_search(sequence, chain, energy, rng, max_iter=2000):
    """
    Simple Monte Carlo local search: try moves (pick a random internal segment,
    flip it to a random valid new position) and accept if energy improves or with
    a small probability (simulated annealing lite).
    """
    n = len(sequence)
    if n < 5:
        return energy
    
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    best_chain = list(chain)
    best_energy = energy
    current_energy = energy
    current_chain = list(chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Temperature schedule
    temp = 2.0
    cooling = 0.995
    
    for it in range(max_iter):
        if _time_left() < 2:
            break
        
        # Pick a random internal segment to move
        seg_len = rng.randint(2, min(6, n - 2))
        seg_start = rng.randint(1, n - seg_len - 1)
        seg_end = seg_start + seg_len - 1
        
        # Remove the segment from occupied
        removed = []
        for i in range(seg_start, seg_end + 1):
            p = current_chain[i]
            del current_occupied[p]
            removed.append(p)
        
        # Try to reinsert the segment at a random valid position
        # The segment must connect: chain[seg_start-1] -> new_pos -> ... -> new_pos_last -> chain[seg_end+1]
        left = current_chain[seg_start - 1]
        right = current_chain[seg_end + 1]
        
        # Try random new positions for the first point of the segment
        found = False
        new_positions = []
        for attempt in range(20):
            # Pick a random neighbour of left as start of new segment
            candidates = [q for q in _neighbours(left) if q not in current_occupied and q != right]
            if not candidates:
                break
            start_pos = rng.choice(candidates)
            
            # Try to grow the segment from start_pos
            temp_chain = [start_pos]
            temp_occupied = dict(current_occupied)
            temp_occupied[start_pos] = seg_start
            success = True
            
            for k in range(1, seg_len):
                idx = seg_start + k
                next_candidates = [q for q in _neighbours(temp_chain[-1]) 
                                   if q not in temp_occupied and q != left]
                if not next_candidates:
                    success = False
                    break
                nxt = rng.choice(next_candidates)
                temp_occupied[nxt] = idx
                temp_chain.append(nxt)
            
            if success:
                # Check connection to right
                if right in _neighbours(temp_chain[-1]):
                    new_positions = temp_chain
                    found = True
                    break
        
        if not found:
            # Restore removed segment
            for i, p in enumerate(removed):
                current_occupied[p] = seg_start + i
            continue
        
        # Compute energy change
        # Remove old contacts involving the segment
        old_energy_change = 0
        # Add new contacts
        new_energy_change = 0
        
        # For simplicity, recompute energy if segment is small
        # Build new chain
        new_chain = current_chain[:seg_start] + new_positions + current_chain[seg_end + 1:]
        
        # Check self-avoidance
        new_occ = set()
        valid = True
        for p in new_chain:
            if p in new_occ:
                valid = False
                break
            new_occ.add(p)
        
        if not valid:
            for i, p in enumerate(removed):
                current_occupied[p] = seg_start + i
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta >= 0 or rng.random() < (1.0 / (1.0 + temp)):
            # Accept
            current_chain = new_chain
            current_energy = new_energy
            current_occupied = {p: i for i, p in enumerate(current_chain)}
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Reject: restore
            for i, p in enumerate(removed):
                current_occupied[p] = seg_start + i
        
        temp *= cooling
    
    return best_energy


def _beam_search(sequence, dimension, beam_width=10, max_depth=None):
    """
    Beam search: keep the top `beam_width` partial chains at each step,
    scoring by energy gained so far plus a heuristic for potential future contacts.
    """
    n = len(sequence)
    if max_depth is None:
        max_depth = n
    
    # Each beam entry: (chain, occupied_dict, energy_so_far)
    start = (0, 0, 0)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, n):
        new_beams = []
        for chain, occupied, energy in beams:
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Heuristic score: energy + small bonus for free space
                score = energy - gain * 1.0 + free * 0.01
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_beams.append((chain + [site], new_occupied, energy - gain, score))
        
        if not new_beams:
            return None
        
        # Sort by score descending, keep top beam_width
        new_beams.sort(key=lambda x: x[3], reverse=True)
        beams = new_beams[:beam_width]
        
        if _time_left() < 10:
            break
    
    if not beams:
        return None
    
    # Return the best complete chain
    best_chain, _, best_energy = max(beams, key=lambda x: x[2])
    if len(best_chain) < n:
        # Incomplete; try to finish greedily
        for i in range(len(best_chain), n):
            options = [site for site in _neighbours(best_chain[-1]) 
                       if site not in {p for p in best_chain}]
            if not options:
                return None
            best_chain.append(options[0])
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the HP sequence on a cubic lattice.
    Uses multiple strategies: greedy growth with local search, beam search,
    and returns the best conformation found.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    # Strategy 1: Greedy growth with local search (multiple restarts)
    num_restarts = 50
    for r in range(num_restarts):
        if _time_left() < 5:
            break
        chain, energy = _greedy_grow(sequence, dimension, rng, use_local_search=True)
        if chain is not None:
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Strategy 2: Beam search
    if _time_left() > 10:
        chain = _beam_search(sequence, dimension, beam_width=8)
        if chain is not None and len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
