"""HP lattice protein folding using Hamiltonian path initialization and local search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation.
    Energy = - (number of H-H contacts)
    Contacts are pairs of H residues on neighboring lattice sites that are not consecutive in the sequence.
    """
    if not chain:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    n = len(chain)
    
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if they are not adjacent in the sequence
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            # To avoid double counting, only count if i < j
                            if i < j:
                                contacts += 1
    return -contacts

def _hamiltonian_path(n):
    """Generate a simple Hamiltonian path on a grid that fits n residues.
    This creates a dense, self-avoiding structure.
    """
    # Find dimensions w, h such that w*h >= n
    # We want a roughly square or rectangular grid.
    w = int(math.sqrt(n))
    if w == 0: w = 1
    while w * w < n:
        w += 1
    h = (n + w - 1) // w
    
    path = []
    # Snake pattern:
    # Row 0: 0 -> w-1
    # Row 1: w-1 -> 0 (down at end)
    # Row 2: 0 -> w-1 (up at start? No, snake usually goes right, down, left, down)
    # Standard snake:
    # (0,0) -> (1,0) -> ... -> (w-1,0) -> (w-1,1) -> (w-2,1) -> ... -> (0,1) -> (0,2) -> ...
    
    for y in range(h):
        if y % 2 == 0:
            for x in range(w):
                if len(path) < n:
                    path.append((x, y))
        else:
            for x in range(w - 1, -1, -1):
                if len(path) < n:
                    path.append((x, y))
                    
    return path

def _local_search(sequence, chain, max_iterations, deadline):
    """Perform local search using Pivot and Kink moves.
    Returns the best chain found.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    # Initial state
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute occupied set for O(1) lookups
    occupied = set(current_chain)
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    # Neighbor cache for speed? Maybe not needed if we just check 4 neighbors.
    
    rng = random.Random(42)
    
    # We will use a simple cooling schedule or just random restarts within the time limit.
    # Given the strict time limit, we'll do a large number of random moves.
    
    # To speed up energy calculation, we can update incrementally, but for simplicity and robustness
    # within 210s, let's see if full recomputation is fast enough for n=100.
    # n=100, neighbors=400. Full recompute is ~1000 ops. 
    # 100,000 iterations * 1000 ops = 100M ops. Python might be slow (10s+).
    # Incremental update is better.
    
    # Let's implement incremental energy update.
    # Energy change depends on:
    # 1. Residue moving in/out of contact with H residues.
    # 2. Residues adjacent to the moving residue in/out of contact with it (if it's H).
    
    def get_energy_delta(idx, new_pos):
        """Calculate energy delta if residue at idx moves to new_pos.
        Assumes new_pos is valid (not occupied by other residues).
        """
        old_pos = current_chain[idx]
        if old_pos == new_pos:
            return 0
        
        delta = 0
        res_type = sequence[idx]
        
        # 1. Impact of moving residue idx
        # Remove contacts of idx at old_pos
        # Add contacts of idx at new_pos
        
        # We need to know which neighbors are H and not adjacent in sequence.
        
        def contact_contribution(pos, res_idx):
            """Count valid H-H contacts involving res_idx at pos."""
            count = 0
            if sequence[res_idx] != 'H':
                return 0
            for nb in _neighbours(pos):
                nb_idx = pos_to_idx.get(nb)
                if nb_idx is not None:
                    if abs(res_idx - nb_idx) > 1:
                        if sequence[nb_idx] == 'H':
                            # Count unique pairs? 
                            # In _compute_energy we counted i<j.
                            # Here we just count interactions.
                            # If we sum over all residues, each pair is counted twice (once for i, once for j).
                            # So energy = -0.5 * sum(contacts).
                            # Or we can just count all and divide by 2 at the end, or be careful.
                            # Let's stick to the definition: -contacts.
                            # If we calculate delta for the whole system:
                            # When i moves, it breaks/creates contacts with j.
                            # If i is H and j is H and |i-j|>1, it's a contact.
                            # So for each neighbor j that is H and |i-j|>1, we add +1 to contact count (if creating) or -1 (if breaking).
                            count += 1
            return count
        
        # This simple counting counts each pair twice in the total sum if we sum over all i.
        # But here we are only looking at the change for specific i.
        # If i moves, it loses its old contacts and gains new ones.
        # The neighbors j don't move, so their "perspective" of contacting i also changes, but that's the same physical contact.
        # So if i is H and j is H, the contact is counted once in the total energy.
        # When i moves away from j, the contact is lost (-1 energy, so +1 delta to energy? No, energy is -contacts. Loss of contact -> energy increases by 1).
        # When i moves to j, contact gained (energy decreases by 1).
        
        old_contacts = 0
        if res_type == 'H':
            for nb in _neighbours(old_pos):
                nb_idx = pos_to_idx.get(nb)
                if nb_idx is not None and abs(idx - nb_idx) > 1 and sequence[nb_idx] == 'H':
                    old_contacts += 1
                    
        new_contacts = 0
        if res_type == 'H':
            for nb in _neighbours(new_pos):
                nb_idx = pos_to_idx.get(nb)
                if nb_idx is not None and abs(idx - nb_idx) > 1 and sequence[nb_idx] == 'H':
                    new_contacts += 1
        
        # Note: The above counts contacts from idx's perspective.
        # If idx is H and nb is H, it's a contact.
        # This is symmetric.
        # So delta_contacts = new_contacts - old_contacts.
        # delta_energy = -delta_contacts.
        
        return -(new_contacts - old_contacts)

    # We need to update pos_to_idx and occupied when we accept a move.
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Choose a move type: Pivot (internal) or Kink (end)
        if rng.random() < 0.7:
            # Pivot move: pick an internal residue (1 to n-2) and move it to a neighbor
            idx = rng.randint(1, n - 2)
            old_pos = current_chain[idx]
            # Try all 4 neighbors
            # We should check if the move is valid (doesn't collide with itself)
            # Since we only move one residue, it can only collide with residues at idx-1 and idx+1?
            # No, it can collide with any other residue in the chain.
            # But wait, if we move residue i to pos p, p must not be occupied by any residue j != i.
            # Also, the new position must not be adjacent to i-1 or i+1 in a way that breaks the chain?
            # No, the chain connectivity is preserved because we only move the position, the sequence order stays the same.
            # The only constraint is self-avoidance: new_pos not in occupied - {old_pos}.
            
            candidates = []
            for nb in _neighbours(old_pos):
                if nb not in occupied or nb == old_pos:
                    # If nb == old_pos, no move.
                    # If nb is occupied by another residue, invalid.
                    if nb == old_pos:
                        continue
                    if nb in occupied:
                        # Check if the occupant is the one moving? No, we are moving idx.
                        # occupied contains all positions.
                        # If nb is in occupied, it means some other residue is there.
                        # So invalid.
                        continue
                    candidates.append(nb)
            
            if candidates:
                new_pos = rng.choice(candidates)
                # Calculate delta
                delta = get_energy_delta(idx, new_pos)
                
                # Acceptance: Metropolis or Greedy?
                # Let's use a simple greedy with some noise or just greedy for speed.
                # Given the tight budget, greedy is safer to ensure we don't get stuck in bad local minima for too long,
                # but simulated annealing is better.
                # Let's try a simple hill climbing first, or SA with high temp.
                
                # Let's use a simple "accept if better or equal" (greedy) to be fast.
                # Or SA: T = 1.0
                if delta <= 0 or (rng.random() < math.exp(-delta / 1.0)):
                    # Accept
                    # Update state
                    occupied.remove(old_pos)
                    occupied.add(new_pos)
                    pos_to_idx.pop(old_pos)
                    pos_to_idx[new_pos] = idx
                    current_chain[idx] = new_pos
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        else:
            # Kink move: move end residue (0 or n-1) to a neighbor
            idx = rng.choice([0, n - 1])
            old_pos = current_chain[idx]
            candidates = []
            for nb in _neighbours(old_pos):
                if nb == old_pos:
                    continue
                if nb in occupied:
                    continue
                candidates.append(nb)
            
            if candidates:
                new_pos = rng.choice(candidates)
                delta = get_energy_delta(idx, new_pos)
                
                if delta <= 0 or (rng.random() < math.exp(-delta / 1.0)):
                    occupied.remove(old_pos)
                    occupied.add(new_pos)
                    pos_to_idx.pop(old_pos)
                    pos_to_idx[new_pos] = idx
                    current_chain[idx] = new_pos
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                        
    return best_chain

def fold(sequence, dimension):
    """Fold the sequence using Hamiltonian path init and local search."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # 1. Generate initial Hamiltonian path
    init_chain = _hamiltonian_path(n)
    
    # 2. Local search
    # We want to use as much time as possible.
    # Let's run local search until deadline.
    # max_iterations is effectively infinite, controlled by time.
    best_chain = _local_search(sequence, init_chain, 10_000_000, deadline)
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
