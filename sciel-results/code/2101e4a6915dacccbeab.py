"""HP lattice protein folding using vectorized energy and multi-start simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _build_pos_array(chain):
    """Build numpy arrays for fast energy computation."""
    n = len(chain)
    pos = np.array(chain, dtype=np.int32)
    return pos

def _energy_from_pos(sequence, pos):
    """Compute HP contact energy from numpy position array."""
    n = len(pos)
    if n < 2:
        return 0
    # Create a 2D grid mapping
    min_x = pos[:, 0].min()
    min_y = pos[:, 1].min()
    max_x = pos[:, 0].max()
    max_y = pos[:, 1].max()
    grid_w = max_x - min_x + 1
    grid_h = max_y - min_y + 1
    
    # Map positions to grid indices
    grid_idx = (pos[:, 0] - min_x) * grid_h + (pos[:, 1] - min_y)
    
    # Build lookup: grid index -> residue index
    lookup = {}
    for i, gi in enumerate(grid_idx):
        lookup[gi] = i
    
    # H residue mask
    h_mask = np.array([1 if c == 'H' else 0 for c in sequence], dtype=np.int8)
    
    contacts = 0
    for i in range(n):
        if h_mask[i] == 0:
            continue
        gi = grid_idx[i]
        # Check 4 neighbours
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = pos[i, 0] + dx - min_x
            ny = pos[i, 1] + dy - min_y
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                ngi = nx * grid_h + ny
                j = lookup.get(ngi)
                if j is not None and j > i + 1 and h_mask[j] == 1:
                    contacts += 1
    return -contacts

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation."""
    pos = _build_pos_array(chain)
    return _energy_from_pos(sequence, pos)

def _compute_energy_delta(sequence, chain, i, old_pos, new_pos):
    """Compute energy change when moving residue i from old_pos to new_pos.
    Only contacts involving residue i or its neighbours change."""
    n = len(chain)
    if n < 3:
        return 0
    
    # Build position lookup for O(1) neighbour checks
    where = {p: idx for idx, p in enumerate(chain)}
    
    def contact_count(pos, residue_idx):
        """Count H-H contacts involving residue at given position and index."""
        if sequence[residue_idx] != 'H':
            return 0
        count = 0
        for q in _neighbours(pos):
            j = where.get(q)
            if j is not None and j != residue_idx and j != residue_idx - 1 and j != residue_idx + 1:
                if sequence[j] == 'H':
                    count += 1
        return count
    
    # Contacts lost: those involving old_pos
    # We need to be careful: only count contacts where i is the smaller index
    # Actually, let's just recompute contacts for residues near i
    
    # Residues that could have contacts with residue i: i-2, i-3, ..., i+n-2, i+n-3, ...
    # But practically, only residues within Manhattan distance 1 of old_pos or new_pos matter
    # and they must be H and non-adjacent in chain
    
    # Simpler approach: compute contacts for old position and new position
    # Contact at old_pos: pairs (i, j) where j is H, non-adjacent, and at neighbour of old_pos
    # Contact at new_pos: same but for new_pos
    
    # But we also need to account for contacts that other residues had with old_pos
    # which are now broken, and contacts other residues gain with new_pos
    
    # Let's do it properly:
    # Old contacts involving position old_pos (where residue i was):
    old_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(old_pos):
            j = where.get(q)
            if j is not None and j != i and abs(j - i) > 1:
                if sequence[j] == 'H':
                    old_contacts += 1
    
    # New contacts involving position new_pos (where residue i will be):
    new_contacts = 0
    if sequence[i] == 'H':
        for q in _neighbours(new_pos):
            j = where.get(q)
            # j must not be i (we're moving i), and not adjacent in chain
            if j is not None and j != i and abs(j - i) > 1:
                if sequence[j] == 'H':
                    new_contacts += 1
    
    # Also, other H residues that were contacting old_pos from their side
    # are already counted above since we check all neighbours of old_pos
    
    # But wait: if residue k (k != i) was at position p_k which is a neighbour of old_pos,
    # and sequence[k] == 'H', that contact was counted. Now that i moves away, it's lost.
    # Similarly, if residue k is at a neighbour of new_pos, a new contact is formed.
    # The above counts both directions, but we need to avoid double-counting.
    
    # Actually, the contact count is symmetric. A contact between i and j is counted once.
    # When we compute old_contacts, we count contacts (i, j) for all valid j at neighbours of old_pos.
    # When we compute new_contacts, we count contacts (i, j) for all valid j at neighbours of new_pos.
    # The delta is new_contacts - old_contacts.
    
    # But we also need to consider: did any other residue k have a contact with old_pos?
    # No, because old_pos is where residue i was. Other residues don't "have" contacts with old_pos
    # unless they are at neighbours of old_pos, which is already captured.
    
    # Wait, I think there's a subtlety. Let me reconsider.
    # The total energy = -number of H-H contacts.
    # A contact is between two H residues at neighbouring positions, non-adjacent in chain.
    # When residue i moves from old_pos to new_pos:
    # - Contacts involving i and old_pos are removed
    # - Contacts involving i and new_pos are added
    # - No other contacts change because no other residue moved
    
    # So delta_energy = -(new_contacts - old_contacts) = old_contacts - new_contacts
    # Since energy = -contacts, delta = new_energy - old_energy = -(new_contacts) - (-old_contacts) = old_contacts - new_contacts
    
    return old_contacts - new_contacts

def _greedy_grow(sequence, rng):
    """Grow chain greedily, prioritizing H-H contact opportunities."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            score = 0.0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 10.0
            
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free_count * 0.1
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.1 and len(candidates) > 1:
            idx = rng.randint(1, min(2, len(candidates) - 1))
            best_site = candidates[idx][1]
        else:
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _spiral_template(n, rng):
    """Generate a compact spiral-shaped self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    for i in range(1, n):
        found = False
        for step in range(4):
            x, y = chain[-1]
            dx, dy = direction
            new_pos = (x + dx, y + dy)
            
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
            
            direction = (dy, -dx)
        
        if not found:
            return None
    
    return chain

def _random_walk(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for i in range(1, n):
        x, y = chain[-1]
        shuffled_dirs = direction[:]
        rng.shuffle(shuffled_dirs)
        
        found = False
        for dx, dy in shuffled_dirs:
            new_pos = (x + dx, y + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                found = True
                break
        
        if not found:
            return None
    
    return chain

def _polish_pass(sequence, chain, rng, max_rounds=50):
    """Aggressive hill-climbing polish: repeatedly find and apply improving moves."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _round in range(max_rounds):
        improved = False
        
        # Try each residue in random order
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            old_pos = best_chain[i]
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = best_chain[i - 1]
                next_pos = best_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev != 1 or dist_next != 1:
                    continue
                
                # Compute energy delta
                delta = _compute_energy_delta(sequence, best_chain, i, old_pos, new_pos)
                
                if delta < 0:  # Energy decreases (more negative = better)
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    best_chain[i] = new_pos
                    best_energy += delta
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy

def _simulated_annealing(sequence, chain, rng, deadline, time_budget):
    """Simulated annealing with incremental energy updates."""
    n = len(sequence)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 4.0
    cooling = 0.998
    min_T = 0.05
    
    while T > min_T and time.time() < deadline - 5.0:
        improved_this_cycle = False
        
        # Process residues in random order
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5.0:
                break
            
            old_pos = current_chain[i]
            valid_moves = []
            
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                
                dist_prev = abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - next_pos[1])
                dist_next = abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1])
                
                if dist_prev == 1 and dist_next == 1:
                    valid_moves.append(new_pos)
            
            if not valid_moves:
                continue
            
            new_pos = rng.choice(valid_moves)
            
            # Compute energy delta incrementally
            delta = _compute_energy_delta(sequence, current_chain, i, old_pos, new_pos)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                del occupied[old_pos]
                occupied[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    improved_this_cycle = True
            else:
                pass  # Rejected, no change needed
            
            # Small probability of early exit if stuck
            if not improved_this_cycle and rng.random() < 0.01:
                break
        
        T *= cooling
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s kill
    n = len(sequence)
    
    if n <= 1:
        return [[0, 0]] if n == 1 else []
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains (fast)
    initial_chains = []
    
    # Greedy (contact-maximizing)
    for attempt in range(5):
        chain = _greedy_grow(sequence, rng)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    # Spiral
    for attempt in range(3):
        chain = _spiral_template(n, rng)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    # Random walks
    for attempt in range(5):
        chain = _random_walk(n, rng)
        if chain is not None and len(chain) == n:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        initial_chains = [[(i, 0) for i in range(n)]]
    
    # Phase 2: Evaluate and pick best initial chain
    for chain in initial_chains:
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_energy = e
            best_chain = list(chain)
    
    # Phase 3: Polish the best initial chain
    if time.time() < deadline - 60.0:
        polished_chain, polished_energy = _polish_pass(sequence, best_chain, rng, max_rounds=100)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    # Phase 4: Simulated annealing from best chain
    if time.time() < deadline - 15.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, deadline, 60.0)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Phase 5: Final polish
    if time.time() < deadline - 5.0:
        polished_chain, polished_energy = _polish_pass(sequence, best_chain, rng, max_rounds=200)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    # Phase 6: One more SA if time remains
    if time.time() < deadline - 5.0:
        sa_chain, sa_energy = _simulated_annealing(sequence, best_chain, rng, deadline, 10.0)
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Final polish
    if time.time() < deadline - 2.0:
        polished_chain, polished_energy = _polish_pass(sequence, best_chain, rng, max_rounds=50)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
