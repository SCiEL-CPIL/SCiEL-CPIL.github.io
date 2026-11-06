<<<<<<< SEARCH
def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, rng, max_iters=5000):
    """
    Local search: try moving individual residues or flipping segments.
    Use delta energy computation for speed.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    # Precompute contact set for delta energy
    def compute_contacts(where, seq):
        contacts = set()
        for i, p in enumerate(seq):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    contact_set = compute_contacts(where, chain)
    
    def delta_energy_move(i, new_pos, where, chain, seq, contact_set):
        """Compute delta energy if we move residue i to new_pos."""
        old_pos = chain[i]
        if new_pos == old_pos:
            return 0, contact_set
        
        # Check if new_pos is valid (not occupied by another residue)
        if new_pos in where:
            j = where[new_pos]
            if j != i:
                return None, contact_set
        
        # Remove old contacts involving i
        old_contacts = set()
        for (a, b) in contact_set:
            if a == i or b == i:
                old_contacts.add((a, b))
        
        # Add new contacts involving i at new_pos
        new_contacts = set()
        if seq[i] == "H":
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i:
                    if seq[j] == "H":
                        pair = (min(i, j), max(i, j))
                        new_contacts.add(pair)
        
        # Delta energy: -len(new_contacts) + len(old_contacts)
        delta = len(old_contacts) - len(new_contacts)
        
        # Update contact_set
        new_contact_set = contact_set - old_contacts | new_contacts
        return delta, new_contact_set
    
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0
    
    for iteration in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Try to move a random residue
        i = rng.randint(1, n - 2)  # Don't move ends
        old_pos = chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                j = where[new_pos]
                if j != i:
                    continue
            
            delta, new_contact_set = delta_energy_move(i, new_pos, where, chain, sequence, contact_set)
            if delta is None:
                continue
            
            if delta < 0:
                # Accept the move
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
            elif delta == 0 and rng.random() < 0.3:
                # Accept neutral moves sometimes
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                no_improve += 1
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        if no_improve > 200:
            break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, max_iters=10000):
    """
    Simulated annealing to escape local minima.
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    def compute_contacts(where, seq, chain):
        contacts = set()
        for i, p in enumerate(chain):
            if seq[i] == "H":
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and seq[j] == "H":
                        contacts.add((i, j))
        return contacts
    
    contact_set = compute_contacts(where, sequence, chain)
    
    def delta_energy_move(i, new_pos, where, chain, seq, contact_set):
        old_pos = chain[i]
        if new_pos == old_pos:
            return 0, contact_set
        
        if new_pos in where:
            j = where[new_pos]
            if j != i:
                return None, contact_set
        
        old_contacts = set()
        for (a, b) in contact_set:
            if a == i or b == i:
                old_contacts.add((a, b))
        
        new_contacts = set()
        if seq[i] == "H":
            for q in _neighbours(new_pos):
                j = where.get(q)
                if j is not None and j != i:
                    if seq[j] == "H":
                        pair = (min(i, j), max(i, j))
                        new_contacts.add(pair)
        
        delta = len(old_contacts) - len(new_contacts)
        new_contact_set = contact_set - old_contacts | new_contacts
        return delta, new_contact_set
    
    best_chain = list(chain)
    best_energy = current_energy
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    for iteration in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        if T < T_min:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        moved = False
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                j = where[new_pos]
                if j != i:
                    continue
            
            delta, new_contact_set = delta_energy_move(i, new_pos, where, chain, sequence, contact_set)
            if delta is None:
                continue
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                # Accept the move
                chain[i] = new_pos
                where.pop(old_pos)
                where[new_pos] = i
                contact_set = new_contact_set
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                moved = True
                break
        
        if not moved:
            pass  # Stay in place
        
        T *= cooling
    
    return best_chain, best_energy
=======
def _fast_energy(sequence, chain):
    """
    Fast energy calculation using a position-to-index map.
    Only checks neighbors of H residues to reduce operations.
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # We only need to check neighbors for H residues
    for i, res in enumerate(sequence):
        if res == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = pos_to_idx.get((nx, ny))
                if j is not None:
                    # Avoid double counting and self-adjacency
                    if j > i + 1:
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """
    Randomized growth with a bias towards compactness and H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        valid_moves = []
        for nb in neighbors:
            if nb not in occupied:
                valid_moves.append(nb)
        
        if not valid_moves:
            return None # Dead end
        
        # Score moves: prefer moves that create H-H contacts
        best_move = None
        best_score = -1000
        
        for nb in valid_moves:
            score = rng.random() * 0.1 # Small random noise
            
            if sequence[i] == 'H':
                nx, ny = nb
                # Check if any neighbor of nb is an H residue already placed
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    check_pos = (nx+dx, ny+dy)
                    if check_pos in occupied:
                        j = occupied[check_pos]
                        if sequence[j] == 'H' and j < i-1:
                            score += 1.0
            
            # Prefer moves that keep the chain compact (closer to origin)
            score -= 0.01 * (abs(nx) + abs(ny))
            
            if score > best_score:
                best_score = score
                best_move = nb
        
        chain.append(best_move)
        occupied[best_move] = i
        
    return chain


def _refine_segment_relocation(sequence, chain, rng, time_limit=5.0):
    """
    Refine the chain using segment relocation (cut-and-paste).
    This is structurally different from single-residue moves, allowing
    larger conformational changes and better escape from local minima.
    """
    n = len(chain)
    if n < 3:
        return chain, _fast_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _fast_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # Parameters for segment relocation
    # Start with smaller segments and increase size? Or random?
    # Random segment length between 2 and min(10, n//3)
    max_seg_len = min(10, n // 3)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if _time_remaining() < 2.0:
            break
        
        # 1. Choose a segment to cut
        # Avoid cutting the first or last residue to maintain connectivity easily?
        # Actually, cutting from 1 to n-2 is safe.
        if n <= 2:
            break
            
        seg_len = rng.randint(2, max_seg_len)
        # Start index such that segment fits
        # We can cut from index 1 to n - seg_len - 1
        if n - seg_len - 1 < 1:
             seg_len = 2
             if n - seg_len - 1 < 1:
                 break
                 
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Extract segment
        segment = current_chain[start_idx:end_idx + 1]
        
        # Remove segment from chain
        # New chain is prefix + suffix
        prefix = current_chain[:start_idx]
        suffix = current_chain[end_idx + 1:]
        
        # We need to find a valid place to insert this segment.
        # The segment must be connected to the rest of the chain at both ends.
        # The "ends" of the remaining chain are prefix[-1] and suffix[0] (if suffix exists)
        # Actually, the chain is broken into two parts: 
        # Part 1: 0 .. start_idx-1
        # Part 2: end_idx+1 .. n-1
        # We need to attach the segment to Part 1 and Part 2? 
        # No, the segment *is* the connection. 
        # Wait, standard cut-and-paste:
        # We remove the segment. The chain is now split into two pieces? 
        # No, if we remove residues i..j, the chain is NOT connected anymore unless we re-insert it.
        # But the chain must be a single continuous walk.
        # So we must re-insert the segment such that it connects the gap.
        # This implies we are effectively moving a sub-chain.
        
        # A simpler and more robust "move" for self-avoiding walks is to pick a random sub-segment 
        # and try to re-insert it at a different location, ensuring the whole thing remains a valid walk.
        # However, maintaining connectivity is tricky.
        
        # Alternative: "Segment Relocation" usually implies:
        # 1. Cut out segment [i, j].
        # 2. The remaining chain is [0..i-1] and [j+1..n-1]. These are two separate pieces.
        # 3. We must re-insert the segment such that it connects the two pieces? 
        #    No, that would just put it back or shift it.
        #    Actually, if we remove [i, j], the residue i-1 and j+1 are now adjacent in the "logical" chain? 
        #    No, they are NOT adjacent in space.
        #    This operation is only valid if we can find a place to put the segment 
        #    such that:
        #    - It connects to i-1
        #    - It connects to j+1
        #    - It doesn't collide with the rest of the chain.
        #    This is very restrictive.
        
        # Let's use a different structural move: **Random Flip** or **Move Residue**.
        # Given the timeout issue, simple and fast is better.
        # Let's stick to **Single Residue Move** but make it MUCH faster.
        
        # The previous failure was likely due to the overhead of set operations in `_local_search`.
        # Let's implement a fast single-residue move using the `pos_to_idx` map.
        
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        x, y = old_pos
        
        # Try 4 neighbors
        moved = False
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (x+dx, y+dy)
            
            # Check if new_pos is occupied
            if new_pos in pos_to_idx:
                # If it's occupied by itself, skip (shouldn't happen as we are moving)
                # If occupied by other, invalid
                # We need to update pos_to_idx dynamically or check carefully.
                # Since we are doing this in a loop, maintaining a global pos_to_idx is hard if we accept moves.
                # Let's just recompute energy if we accept.
                
                # Optimization: Only check if new_pos is free.
                # We need a quick way to know if new_pos is free.
                # We can maintain a set of occupied positions.
                pass # Placeholder, see below implementation
        
        # To make this fast, we need a maintained set of occupied positions.
        # Let's rewrite the inner loop to be efficient.
        
        # Break out of the while loop if we implement the efficient version below.
        # I will replace the entire function body with an efficient single-residue mover.
        
        break # Stop using the complex segment logic above, use the simple fast mover below.
        
    return best_chain, best_energy
>>>>>>> REPLACE

The `_refine_segment_relocation` function above was a draft that decided to switch to a simpler approach. I will now provide the actual replacement for the refinement function and the `fold` function to ensure efficiency.

