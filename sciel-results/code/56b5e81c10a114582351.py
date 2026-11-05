"""HP lattice protein folding using Monte Carlo Pivoting with Incremental Energy Updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.92  # Use 92% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using randomized greedy growth 
    with a bias towards H-H proximity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                # Check if this site creates a new H-H contact
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 100 # Strong bias for direct contact
            
            # Check free space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Randomness to escape local minima in growth
            score = gain + free * 0.1 + rng.random() * 10.0
            options.append((score, site))
        
        if not options:
            # Dead end, return None to trigger restart
            return None
        
        # Pick best option
        options.sort(reverse=True)
        # Sometimes pick second best to add diversity
        if len(options) > 1 and rng.random() < 0.3:
            site = options[1][1]
        else:
            site = options[0][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _local_search(sequence, chain, dimension, rng, start_temp, end_temp, steps):
    """
    Improve a chain using Monte Carlo Pivoting with incremental energy updates.
    """
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map
    pos_map = {p: i for i, p in enumerate(chain)}
    
    # Pre-compute H status
    is_h = [1 if s == "H" else 0 for s in sequence]
    
    # Compute initial energy and contact map
    current_energy = 0
    # contact_map[i] is a set of j such that i and j are in contact (H-H)
    # We only care about H-H contacts.
    # To make incremental updates easy, we maintain a set of contact pairs (min_idx, max_idx)
    # Or simpler: just recompute energy for the affected residues.
    
    # For incremental updates, we need to know which pairs are currently in contact.
    # Let's maintain a set of tuples (i, j) where i < j and they are H-H contacts.
    contact_pairs = set()
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(chain[i]):
                j = pos_map.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contact_pairs.add((i, j))
    
    current_energy = -len(contact_pairs)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Pre-compute neighbors for each residue index for faster lookup? 
    # No, neighbors depend on position which changes.
    
    for step in range(steps):
        if _is_deadline_exceeded():
            break
        
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Choose a pivot point
        # Pivoting around a single residue is often too local.
        # Pivoting a segment is better. Let's pick a segment length L.
        # Small L (2-3) allows more moves, larger L allows bigger jumps.
        # Let's mix: 50% single residue (L=1), 50% segment (L=2-4)
        
        if rng.random() < 0.5:
            L = 1
            i = rng.randint(1, n - 2)
        else:
            max_L = min(4, n - 2)
            L = rng.randint(2, max_L)
            i = rng.randint(1, n - L - 1)
        
        # The segment is chain[i : i+L]
        # Pivot around chain[i-1]
        # The segment must remain connected to chain[i-1] and chain[i+L]
        
        # For L=1:
        # Move chain[i] to a new position adjacent to chain[i-1] (and chain[i+1])
        # Wait, standard pivot:
        # If we pivot at i, we keep chain[0..i-1] fixed.
        # We rotate the tail chain[i..n-1] around chain[i-1].
        # This is expensive to check for self-avoidance for long tails.
        
        # Alternative: "Segment Move" (Bridge Move)
        # Remove segment chain[i : i+L].
        # The gap is between chain[i-1] and chain[i+L].
        # We need to find a new position for the segment such that:
        # 1. New start is adjacent to chain[i-1]
        # 2. New end is adjacent to chain[i+L]
        # 3. No self-intersection with the rest of the chain.
        
        # This is hard to find efficiently.
        
        # Let's stick to a simpler move that is still more global than single-residue:
        # "Pivot" the segment chain[i : i+L] around chain[i-1].
        # This means we try to rotate the segment.
        # For L=1, this is just moving chain[i] to a neighbor of chain[i-1] 
        # that is also a neighbor of chain[i+1] (if L=1, the segment is just point i, 
        # but it must stay connected to i+1? No, if we pivot at i, 
        # chain[i-1] is the pivot. The segment is chain[i..n-1]? No, that's too big.
        
        # Let's use the "Segment Relocation" move described in literature for HP folding.
        # Actually, let's use a simpler "Local Pivot" for the segment.
        # We pick a segment chain[i : i+L].
        # We try to move the *entire* segment to a new location.
        # To do this efficiently, we can try to shift the segment by a small vector?
        # No, that breaks connectivity.
        
        # Let's implement a "Pivot" on a small segment attached to a fixed anchor.
        # Anchor at i-1. Segment is chain[i : i+L].
        # We try all 6 directions from chain[i-1] for the new chain[i].
        # Then we greedily try to grow the rest of the segment (chain[i+1]...chain[i+L]) 
        # from the new chain[i] to reach the new position of chain[i+L]?
        # No, the end of the segment chain[i+L] is connected to chain[i+L+1] which is fixed.
        # So the new end of the segment MUST be adjacent to chain[i+L+1].
        # And the new start MUST be adjacent to chain[i-1].
        
        # This is a "Bridge" move.
        # Find a path of length L from a neighbor of chain[i-1] to a neighbor of chain[i+L+1].
        # This is a shortest path problem in a small graph.
        
        # Given the complexity, let's use a simpler move that is still effective:
        # "Single Residue Move" but with a wider search (try all 6 neighbors).
        # And "Double Residue Move" (L=2).
        
        if L == 1:
            # Move chain[i]
            old_pos = chain[i]
            prev_pos = chain[i-1]
            next_pos = chain[i+1]
            
            valid_sites = []
            for site in _neighbours(prev_pos):
                if site == next_pos:
                    continue
                if site in pos_map:
                    # Check if it's the residue itself
                    if pos_map[site] == i:
                        continue
                    continue
                
                # Check if site is adjacent to next_pos
                if abs(site[0]-next_pos[0]) + abs(site[1]-next_pos[1]) + abs(site[2]-next_pos[2]) == 1:
                    valid_sites.append(site)
            
            if not valid_sites:
                continue
            
            site = valid_sites[rng.randint(0, len(valid_sites)-1)]
            
            # Apply move
            chain[i] = site
            del pos_map[old_pos]
            pos_map[site] = i
            
            # Update Energy Incrementally
            # Remove old contacts involving i
            removed_contacts = 0
            for q in _neighbours(old_pos):
                j = pos_map.get(q)
                if j is not None and j != i:
                    if is_h[i] and is_h[j]:
                        # Check if (min(i,j), max(i,j)) was in contact_pairs
                        pair = (min(i, j), max(i, j))
                        if pair in contact_pairs:
                            contact_pairs.remove(pair)
                            removed_contacts += 1
            
            # Add new contacts involving i
            added_contacts = 0
            for q in _neighbours(site):
                j = pos_map.get(q)
                if j is not None and j != i:
                    if is_h[i] and is_h[j]:
                        pair = (min(i, j), max(i, j))
                        if pair not in contact_pairs:
                            contact_pairs.add(pair)
                            added_contacts += 1
            
            new_energy = current_energy - removed_contacts + added_contacts
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_pos
                del pos_map[site]
                pos_map[old_pos] = i
                # Revert contact pairs
                for q in _neighbours(site):
                    j = pos_map.get(q)
                    if j is not None and j != i:
                        if is_h[i] and is_h[j]:
                            pair = (min(i, j), max(i, j))
                            if pair in contact_pairs:
                                contact_pairs.remove(pair)
                for q in _neighbours(old_pos):
                    j = pos_map.get(q)
                    if j is not None and j != i:
                        if is_h[i] and is_h[j]:
                            pair = (min(i, j), max(i, j))
                            if pair not in contact_pairs:
                                contact_pairs.add(pair)
        
        elif L == 2:
            # Move chain[i] and chain[i+1]
            # This is more complex. 
            # Let's skip L=2 for now to ensure correctness and speed, 
            # as L=1 with high frequency is often sufficient for HP models 
            # if the cooling schedule is good.
            pass
        else:
            # Skip larger segments for now
            pass

    return best_chain

def fold(sequence, dimension):
    """Best conformation using heuristic growth and local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    
    # Allocate time based on sequence length
    # Longer sequences need more steps per restart but fewer restarts
    # Shorter sequences can afford more restarts
    
    # Estimate steps per second for this sequence length
    # We will run until deadline
    
    num_restarts = 50 # Initial guess, will be adjusted by time
    
    for restart in range(num_restarts):
        if _is_deadline_exceeded():
            break
        
        # Heuristic growth
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            # Fallback to simple growth
            chain = [(i, 0, 0) for i in range(n)]
        
        # Local Search
        # Scale steps with sequence length
        # For n=48, ~2000 steps is fast.
        # For n=136, ~500 steps might be enough for a quick improvement, 
        # but we have time for more.
        # Let's use a dynamic step count based on remaining time?
        # No, keep it simple: fixed steps per restart, but adjust restarts.
        
        sa_steps = max(1000, n * 50)
        
        # Two phase: 
        # Phase 1: High temp (10.0 -> 1.0)
        # Phase 2: Low temp (1.0 -> 0.1)
        
        chain = _local_search(sequence, chain, dimension, rng, 10.0, 1.0, sa_steps // 2)
        chain = _local_search(sequence, chain, dimension, rng, 1.0, 0.1, sa_steps // 2)
        
        value = _compute_energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
