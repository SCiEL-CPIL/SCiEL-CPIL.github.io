"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _pivot_move(sequence, chain, i, angle, rng):
    """
    Perform a pivot move at index i.
    Rotates the suffix chain[i:] by 'angle' degrees around chain[i-1].
    angle is in {90, 180, 270}.
    Returns (new_chain, new_occupied, energy_delta) or None if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    # Generate rotated positions
    new_suffix = []
    for p in suffix:
        # Translate p relative to pivot
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        # Rotate
        if angle == 90:
            # Rotate around Z axis? Or arbitrary? 
            # In cubic lattice, we can rotate around any axis passing through pivot.
            # Simplest: Rotate around X, Y, or Z axis.
            # Let's pick a random axis for the rotation to explore more space.
            # But for a specific "angle" parameter, we need a specific rotation.
            # Let's define rotations around X, Y, Z axes.
            # For simplicity, let's just try all 6 possible moves for the first step of the suffix?
            # No, pivot moves rotate the *entire* suffix.
            # Let's implement a generic 90-degree rotation around a random axis (X, Y, or Z) passing through pivot.
            pass
        
    # This implementation is getting complex. Let's use a simpler move: 
    # "Slide" a segment or just use a random walk with backtracking.
    # Actually, the simplest robust move is: pick a random i, remove chain[i], 
    # and try to re-insert it at a random valid position adjacent to chain[i-1] or chain[i+1]? 
    # No, that breaks connectivity.
    
    # Let's stick to the standard Pivot Move:
    # 1. Pick i.
    # 2. Pick a random axis (0, 1, or 2) and a random sign (-1, 1).
    # 3. Rotate the vector (p - pivot) by 90 degrees around that axis.
    # 4. Check if the new suffix is self-avoiding and doesn't clash with the prefix.
    
    axis = rng.randint(0, 2)
    sign = rng.choice([-1, 1])
    
    new_suffix = []
    for p in suffix:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        dz = p[2] - pivot[2]
        
        if axis == 0: # Rotate around X
            new_dx = dx
            new_dy = -sign * dz
            new_dz = sign * dy
        elif axis == 1: # Rotate around Y
            new_dx = sign * dz
            new_dy = dy
            new_dz = -sign * dx
        else: # Rotate around Z
            new_dx = -sign * dy
            new_dy = sign * dx
            new_dz = dz
            
        new_p = (pivot[0] + new_dx, pivot[1] + new_dy, pivot[2] + new_dz)
        new_suffix.append(new_p)
        
    # Check validity
    # 1. Self-avoidance within suffix
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    # 2. No clash with prefix (chain[0..i-1])
    # Note: chain[i-1] is the pivot, so it's allowed to be adjacent, but not occupied by another part.
    # The prefix is chain[0..i-1]. The new suffix starts at new_suffix[0].
    # new_suffix[0] must not be in chain[0..i-2].
    # Also, any p in new_suffix must not be in chain[0..i-1] (except the pivot which is shared? No, pivot is chain[i-1], new_suffix[0] is the new position of chain[i]).
    # So new_suffix[0] != chain[i-1] is guaranteed by rotation (distance 1).
    # But new_suffix[k] must not equal chain[j] for any j < i.
    
    prefix_set = set(chain[:i])
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    # 3. Check that the first step is valid (adjacent to pivot) - guaranteed by rotation.
    
    # Construct new chain
    new_chain = chain[:i] + new_suffix
    
    # Compute energy delta
    old_occupied = {p: idx for idx, p in enumerate(chain)}
    new_occupied = {p: idx for idx, p in enumerate(new_chain)}
    
    # Energy delta calculation
    # We only need to check residues that moved or whose neighbors changed.
    # Moved residues: i to n-1.
    # Unmoved residues: 0 to i-1.
    # Contacts can only change if they involve a moved residue.
    
    delta = 0
    
    # Check H-H contacts involving moved residues in the new state
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    # This is a contact in the new state.
                    # Was it a contact in the old state?
                    # If j < i, j didn't move. p moved.
                    # If j >= i, both moved.
                    # We need to check if (p_old, q_old) was a contact.
                    # This is tricky. Easier: compute energy of old and new restricted to affected residues?
                    # Or just compute full energy? For n=136, full energy is O(n * 6) = ~800 ops.
                    # If we do this for every move, it's slow.
                    # Let's do a delta calculation.
                    
                    # Contact in new state: (idx, j)
                    # If j < i:
                    #   Old position of idx was chain[idx].
                    #   Was chain[idx] adjacent to chain[j]?
                    #   If yes, it was a contact before (if both H).
                    #   If no, it's a new contact.
                    # If j >= i:
                    #   Both moved.
                    #   Check if old_chain[idx] was adjacent to old_chain[j].
                    
                    old_p = chain[idx]
                    old_j_p = chain[j]
                    was_contact = (abs(old_p[0]-old_j_p[0]) + abs(old_p[1]-old_j_p[1]) + abs(old_p[2]-old_j_p[2]) == 1)
                    
                    if not was_contact:
                        delta += 1 # Gained a contact (energy decreases, so delta_energy = -1, but we track delta_contacts = +1)
                    # If was_contact, no change in count for this pair.
                    
    # We also need to check if we LOST any contacts.
    # The above loop counts contacts in the NEW state.
    # It adds +1 for each new contact that wasn't there before.
    # But it doesn't subtract for contacts that existed before but are gone now.
    # Let's refine:
    
    delta = 0
    # Count contacts in old state involving moved residues
    old_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                j = old_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    old_contacts += 1
                    
    # Count contacts in new state involving moved residues
    new_contacts = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                j = new_occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    new_contacts += 1
                    
    # Note: This counts each contact twice if both residues are in the moved set?
    # If idx and j are both >= i, the contact (idx, j) is counted once for idx and once for j?
    # No, the loop `for q in _neighbours(p)` checks all neighbors.
    # If idx and j are both H and adjacent, when idx is processed, j is found. When j is processed, idx is found.
    # So old_contacts and new_contacts are double the number of H-H pairs in the moved set?
    # And what about contacts between moved (idx) and unmoved (j < i)?
    # When idx is processed, j is found. When j is processed (not in loop), it's not counted.
    # So:
    # Pairs (idx, j) with idx>=i, j<i: Counted once in old_contacts (when idx processed) and once in new_contacts (when idx processed).
    # Pairs (idx, j) with idx>=i, j>=i: Counted twice in old_contacts (once for idx, once for j) and twice in new_contacts.
    
    # So delta_contacts = new_contacts - old_contacts.
    # Since both are double-counted for internal pairs and single for cross pairs, the difference is correct for the total change in contact count.
    
    delta = new_contacts - old_contacts
    
    if delta <= 0:
        # We only accept if delta > 0 in hill climbing, but in MC we might accept worse.
        # For now, return delta.
        pass
        
    return new_chain, new_occupied, delta


def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


# Removed _compute_energy_delta, replaced by logic in _pivot_move


def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy growth with simulated annealing-like acceptance."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * temperature
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best with some randomness
        candidates.sort(reverse=True)
        if rng.random() < 0.8:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


# Removed _single_residue_move


def _segment_move(sequence, chain, rng, occupied):
    """
    Try to rotate a segment of the chain. Pick a random segment [i, j]
    and try to rotate it by 90 or 180 degrees around a pivot.
    Returns (new_chain, new_occupied, energy_delta) or None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick a random segment length (small segments for efficiency)
    seg_len = rng.choice([2, 3, 4, 5])
    if n <= seg_len + 2:
        seg_len = max(2, n - 2)
    
    # Pick random start index for segment
    i = rng.randint(1, n - seg_len - 1)
    j = i + seg_len - 1
    
    # The segment is chain[i..j]
    # It's connected to chain[i-1] on one end and chain[j+1] on the other
    # We need to find a new position for the entire segment such that:
    # - chain[i] is adjacent to chain[i-1]
    # - chain[j] is adjacent to chain[j+1]
    # - No part of the segment overlaps with the rest of the chain
    
    prev_pos = chain[i-1]
    next_pos = chain[j+1]
    
    # Try different rotations: we'll try translating the segment
    # Actually, let's try a simpler approach: pick a new position for chain[i]
    # that is adjacent to prev_pos, then try to grow the segment from there
    
    # Get the relative positions of the segment
    seg_positions = [(chain[k][0] - chain[i][0], chain[k][1] - chain[i][1], chain[k][2] - chain[i][2]) for k in range(i, j+1)]
    
    # Try all 6 possible directions from prev_pos (excluding the direction back to chain[i])
    old_first = chain[i]
    possible_new_firsts = []
    for site in _neighbours(prev_pos):
        if site != old_first:
            # Check if placing chain[i] here doesn't immediately conflict
            # with non-adjacent parts
            conflict = False
            for k in range(i+1, j+1):
                # Check if the new position of chain[k] would conflict
                new_k = (site[0] + seg_positions[k-i][0], site[1] + seg_positions[k-i][1], site[2] + seg_positions[k-i][2])
                occ = occupied.get(new_k)
                if occ is not None and occ != k:
                    conflict = True
                    break
            if not conflict:
                # Also check that chain[j] (last of segment) is adjacent to next_pos
                new_last = (site[0] + seg_positions[seg_len-1][0], site[1] + seg_positions[seg_len-1][1], site[2] + seg_positions[seg_len-1][2])
                if new_last != next_pos:
                    # Check adjacency
                    diff = abs(new_last[0] - next_pos[0]) + abs(new_last[1] - next_pos[1]) + abs(new_last[2] - next_pos[2])
                    if diff == 1:
                        possible_new_firsts.append(site)
    
    if not possible_new_firsts:
        return None
    
    new_first = rng.choice(possible_new_firsts)
    
    # Build new chain
    new_chain = list(chain)
    for k in range(i, j+1):
        new_chain[k] = (new_first[0] + seg_positions[k-i][0], new_first[1] + seg_positions[k-i][1], new_first[2] + seg_positions[k-i][2])
    
    # Verify self-avoidance
    seen = set()
    for k, pos in enumerate(new_chain):
        if pos in seen:
            return None
        seen.add(pos)
    
    # Compute energy delta
    old_occupied = dict(occupied)
    new_occupied = dict(occupied)
    for k in range(i, j+1):
        del new_occupied[chain[k]]
        new_occupied[new_chain[k]] = k
    
    delta = 0
    for k in range(i, j+1):
        if sequence[k] == 'H':
            # Lost contacts at old position
            for q in _neighbours(chain[k]):
                j2 = old_occupied.get(q)
                if j2 is not None and j2 != k and abs(j2 - k) > 1 and sequence[j2] == 'H':
                    # Check if this contact still exists in new configuration
                    # i.e., is there still an H at the corresponding new position?
                    # The neighbor q in old config corresponds to... this is complex
                    # Let's just recompute energy for affected residues
                    pass
            
            # Gained contacts at new position
            for q in _neighbours(new_chain[k]):
                j2 = new_occupied.get(q)
                if j2 is not None and j2 != k and abs(j2 - k) > 1 and sequence[j2] == 'H':
                    # Check if this contact existed before
                    # Find the old position of j2
                    old_j2_pos = chain[j2]
                    # Is new_chain[k] adjacent to old_j2_pos?
                    if abs(new_chain[k][0] - old_j2_pos[0]) + abs(new_chain[k][1] - old_j2_pos[1]) + abs(new_chain[k][2] - old_j2_pos[2]) != 1:
                        delta += 1
            for q in _neighbours(chain[k]):
                j2 = old_occupied.get(q)
                if j2 is not None and j2 != k and abs(j2 - k) > 1 and sequence[j2] == 'H':
                    # Check if this contact is lost
                    old_j2_pos = chain[j2]
                    new_j2_pos = new_chain[j2]
                    # The contact was between chain[k] and old_j2_pos
                    # Now chain[k] is at new_chain[k], and j2 is at new_j2_pos
                    # The contact is lost if new_chain[k] is not adjacent to new_j2_pos
                    # AND the old contact position q is no longer occupied by an H
                    # This is getting complex. Let's just compute full energy delta by
                    # checking all H-H pairs involving the moved segment
                    pass
    
    # Simpler: compute energy of old and new chains restricted to affected residues
    # Actually, let's just compute the full energy difference
    old_energy = _energy(sequence, chain)
    new_energy = _energy(sequence, new_chain)
    delta = old_energy - new_energy  # positive if new is better (more negative)
    
    if delta <= 0:
        return None
    
    return new_chain, new_occupied, delta


def _local_search(sequence, chain, rng, deadline):
    """
    Monte Carlo local search using pivot moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temperature = 2.0
    cooling_rate = 0.999
    
    while time.time() < deadline and temperature > 0.1:
        # Pick a random pivot index
        i = rng.randint(1, n - 1)
        
        result = _pivot_move(sequence, current_chain, i, 90, rng)
        if result is not None:
            new_chain, new_occupied, delta = result
            # delta is change in contacts (positive = better)
            # Energy change is -delta
            if delta > 0:
                current_chain = new_chain
                current_energy -= delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            else:
                # Accept worse move with probability exp(delta / temperature)
                # delta is negative
                if delta > -50:
                    if rng.random() < math.exp(delta / temperature):
                        current_chain = new_chain
                        current_energy -= delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        temperature *= cooling_rate
        
    return best_chain, best_energy


# Removed _simulated_annealing


# Removed _parallel_tempering


def _heuristic_grow(sequence, dimension, rng):
    """
    More sophisticated growth that considers future potential.
    """
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    # Precompute H positions for faster lookup
    h_positions = [i for i, r in enumerate(sequence) if r == 'H']
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score based on:
            # 1. Immediate H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # 2. Potential for future contacts (number of H residues ahead that could contact)
            potential = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q not in occupied:
                        potential += 0.5
            
            # 3. Free space for future growth
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # 4. Avoid trapping
            trap_score = 0
            if free < 2:
                trap_score = 1.0
            
            score = gain * 2.0 + potential * 0.5 + 0.1 * free - trap_score + rng.random() * 0.3
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(reverse=True)
        k = min(3, len(candidates))
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            site = candidates[rng.randint(0, k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Use a fraction of the remaining time
    budget = min(remaining * 0.95, 20.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Generate multiple initial conformations
    n_grows = max(2, int((deadline - time.time()) * 3))
    n_grows = min(n_grows, 40)
    
    grow_deadline = time.time() + (deadline - time.time()) * 0.25
    
    for _ in range(n_grows):
        if time.time() > grow_deadline:
            break
        
        strategy = rng.random()
        if strategy < 0.4:
            chain = _greedy_grow(sequence, dimension, rng, temperature=1.0)
        elif strategy < 0.7:
            chain = _heuristic_grow(sequence, dimension, rng)
        else:
            chain = _greedy_grow(sequence, dimension, rng, temperature=2.0)
        
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [[i, 0, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best
    if time.time() < deadline - 1.0:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, deadline - 1.0)
    
    return best_chain

# EVOLVE-BLOCK-END
