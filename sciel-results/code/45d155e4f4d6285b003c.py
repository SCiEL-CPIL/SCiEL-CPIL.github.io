import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

class HPFoldingState:
    def __init__(self, sequence, chain):
        self.sequence = sequence
        self.chain = chain
        self.n = len(chain)
        self.pos_to_idx = {p: i for i, p in enumerate(chain)}
        self.energy = _compute_energy(sequence, chain)

    def _get_contact_change(self, moved_indices, new_positions):
        """
        Calculate the change in energy when residues at moved_indices 
        are moved to new_positions.
        moved_indices: list of indices in the chain
        new_positions: list of new (x,y) coordinates for these indices
        """
        delta_energy = 0
        seq = self.sequence
        
        # We need to check contacts involving the moved residues.
        # A contact is a pair (i, j) such that |i-j| > 1 and dist(chain[i], chain[j]) == 1.
        
        # 1. Remove old contacts involving moved residues
        for idx in moved_indices:
            if seq[idx] != 'H':
                continue
            old_pos = self.chain[idx]
            for neighbor in _neighbours(old_pos):
                j = self.pos_to_idx.get(neighbor)
                if j is not None and abs(j - idx) > 1 and seq[j] == 'H':
                    # This contact is broken if the neighbor is not also moving to an adjacent position
                    # But simpler: just count all contacts involving idx that are currently present
                    # and subtract them, then add new ones.
                    # We must be careful not to double count.
                    # Let's use a set of contacts.
                    pass # This approach is tricky. Let's use a different strategy.
        
        # Better strategy: 
        # Identify all pairs (i, j) that were contacts before the move.
        # Identify all pairs (i, j) that are contacts after the move.
        # Energy change = - (new_contacts - old_contacts).
        
        # Since only a small segment moves, only contacts involving that segment can change.
        # Let M be the set of moved indices.
        # Old contacts: pairs (i, j) where i in M or j in M, and they were adjacent.
        # New contacts: pairs (i, j) where i in M or j in M, and they are adjacent in new config.
        
        old_contacts = 0
        new_contacts = 0
        
        # We need a temporary mapping for new positions to check new contacts
        # But we also need to ensure we don't count contacts within the moved segment 
        # if they are adjacent in the chain (which is allowed) but we only count non-adjacent.
        
        # Let's iterate over moved indices
        for idx in moved_indices:
            if seq[idx] != 'H':
                continue
            
            # Old contacts for idx
            old_pos = self.chain[idx]
            for neighbor in _neighbours(old_pos):
                j = self.pos_to_idx.get(neighbor)
                if j is not None and abs(j - idx) > 1 and seq[j] == 'H':
                    # Count this contact
                    # To avoid double counting, only count if idx < j
                    if idx < j:
                        old_contacts += 1
            
            # New contacts for idx
            new_pos = new_positions[moved_indices.index(idx)]
            # We need to check if any neighbor of new_pos is occupied by another H residue
            # that is not adjacent in the chain.
            # Note: The moved residues themselves are now at new positions.
            # We must check against the *rest* of the chain and *other* moved residues.
            
            # For simplicity and correctness, let's reconstruct the local neighborhood check.
            # This is O(K^2) where K is segment length, which is small.
            pass # The above logic is getting complex. Let's use a simpler, robust method.
        
        # Robust Method:
        # 1. Store old positions of moved residues.
        # 2. Temporarily update chain and pos_to_idx for moved residues.
        # 3. Calculate energy of the whole chain? No, too slow.
        # 4. Calculate delta by checking all pairs involving moved residues.
        
        # Let's do this:
        # 1. Identify all contacts that *involve* at least one moved residue.
        # 2. Count how many of these are valid contacts in the OLD state.
        # 3. Count how many are valid contacts in the NEW state.
        # 4. Delta = - (New - Old).
        
        # We need to be careful with the pos_to_idx lookup for the NEW state.
        # Let's create a temporary pos_to_idx_new for the moved residues and merge with the rest.
        
        # Actually, the simplest robust way for small segments:
        # 1. Save old state of moved residues.
        # 2. Update self.chain and self.pos_to_idx for moved residues.
        # 3. Calculate new energy from scratch? No.
        # 4. Calculate delta by iterating over moved residues and their neighbors.
        
        # Let's try a direct delta calculation:
        # For each moved residue i:
        #   - Remove all contacts i had with non-moved residues.
        #   - Remove all contacts i had with other moved residues (if any).
        #   - Add all new contacts i has with non-moved residues.
        #   - Add all new contacts i has with other moved residues.
        
        # This requires knowing which residues are moved.
        moved_set = set(moved_indices)
        
        # 1. Subtract old contacts involving moved residues
        for idx in moved_indices:
            if self.sequence[idx] != 'H':
                continue
            old_pos = self.chain[idx]
            for neighbor in _neighbours(old_pos):
                j = self.pos_to_idx.get(neighbor)
                if j is not None and abs(j - idx) > 1 and self.sequence[j] == 'H':
                    # This contact exists. We subtract it.
                    # We only subtract if we are going to recalculate it.
                    # To avoid double subtraction, we can just sum up all such contacts
                    # and subtract the total.
                    # But we must be careful: if both i and j are moved, we might count twice.
                    # Let's just calculate the total number of contacts involving moved residues in the old state.
                    # And the total number in the new state.
                    pass
        
        # This is getting messy. Let's use a helper to get contacts for a specific index.
        def count_contacts_for(idx, chain, pos_to_idx, sequence):
            if sequence[idx] != 'H':
                return 0
            count = 0
            pos = chain[idx]
            for neighbor in _neighbours(pos):
                j = pos_to_idx.get(neighbor)
                if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                    count += 1
            return count

        # Old contacts involving moved residues:
        # Sum of count_contacts_for(idx) for idx in moved_indices.
        # But this double counts contacts between two moved residues.
        # Let's just iterate over all pairs (i, j) where i < j, i in moved, j in moved OR j not in moved.
        # This is O(K * N) in worst case if we check all neighbors.
        # But K is small.
        
        old_delta = 0
        for idx in moved_indices:
            if self.sequence[idx] != 'H':
                continue
            old_pos = self.chain[idx]
            for neighbor in _neighbours(old_pos):
                j = self.pos_to_idx.get(neighbor)
                if j is not None and abs(j - idx) > 1 and self.sequence[j] == 'H':
                    # To avoid double counting, only count if idx < j
                    if idx < j:
                        old_delta += 1
        
        # Now, we need to calculate new_delta.
        # We need a temporary pos_to_idx that reflects the new positions of moved residues.
        # But we can't modify self.pos_to_idx yet.
        # Let's create a local dict for the moved residues.
        temp_pos_to_idx = {}
        for idx, new_pos in zip(moved_indices, new_positions):
            temp_pos_to_idx[new_pos] = idx
        
        # For neighbors that are NOT moved, we use self.pos_to_idx.
        # For neighbors that ARE moved, we use temp_pos_to_idx.
        
        new_delta = 0
        for idx in moved_indices:
            if self.sequence[idx] != 'H':
                continue
            new_pos = new_positions[moved_indices.index(idx)]
            for neighbor in _neighbours(new_pos):
                # Check if neighbor is a moved residue
                if neighbor in temp_pos_to_idx:
                    j = temp_pos_to_idx[neighbor]
                    # Ensure j is not idx (self)
                    if j == idx:
                        continue
                else:
                    j = self.pos_to_idx.get(neighbor)
                    if j is None:
                        continue
                    # Check if j is a moved residue (it shouldn't be, as we checked temp first)
                    # But wait, if neighbor is in temp_pos_to_idx, it's moved.
                    # If not, it's not moved.
                
                if j is not None and abs(j - idx) > 1 and self.sequence[j] == 'H':
                    if idx < j:
                        new_delta += 1
        
        return -(new_delta - old_delta)

    def _apply_move(self, moved_indices, new_positions):
        """Apply a move to the state."""
        for idx, new_pos in zip(moved_indices, new_positions):
            old_pos = self.chain[idx]
            del self.pos_to_idx[old_pos]
            self.chain[idx] = new_pos
            self.pos_to_idx[new_pos] = idx

    def _revert_move(self, moved_indices, old_positions):
        """Revert a move in the state."""
        for idx, old_pos in zip(moved_indices, old_positions):
            new_pos = self.chain[idx]
            del self.pos_to_idx[new_pos]
            self.chain[idx] = old_pos
            self.pos_to_idx[old_pos] = idx

def _initial_chain(sequence, dimension, rng):
    """Create an initial random self-avoiding walk."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        site = rng.choice(candidates)
        chain.append(site)
        occupied.add(site)
    
    return chain

def _anneal(sequence, dimension, rng, time_budget):
    """Run simulated annealing with pivot and slide moves."""
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None:
        return None, 0
    
    state = HPFoldingState(sequence, chain)
    best_chain = state.chain.copy()
    best_energy = state.energy
    
    start_temp = 1.5
    end_temp = 0.05
    # Estimate steps based on time budget
    # Assume ~10000 steps per second with incremental updates
    estimated_steps = int(time_budget * 8000)
    
    start_time = time.time()
    
    n = len(sequence)
    
    for step in range(estimated_steps):
        if time.time() - start_time > time_budget or _remaining_time() < 5.0:
            break
        
        progress = step / estimated_steps
        temperature = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type: 0=pivot, 1=slide
        move_type = rng.random()
        
        moved_indices = []
        new_positions = []
        old_positions = []
        
        if move_type < 0.7: # Pivot move
            # Select pivot index i (1 to n-2)
            i = rng.randint(1, n - 2)
            # Select segment length L (1 to n-i-1)
            # Bias towards shorter segments? Or random?
            # Random length is standard.
            max_len = n - i - 1
            if max_len <= 0: continue
            L = rng.randint(1, max_len)
            
            pivot_pos = state.chain[i]
            segment_indices = list(range(i + 1, i + 1 + L))
            
            # Rotate segment by 90 degrees clockwise
            rotated_positions = []
            for idx in segment_indices:
                pos = state.chain[idx]
                dx = pos[0] - pivot_pos[0]
                dy = pos[1] - pivot_pos[1]
                # Rotate (dx, dy) -> (dy, -dx)
                new_dx, new_dy = dy, -dx
                rotated_positions.append((pivot_pos[0] + new_dx, pivot_pos[1] + new_dy))
            
            moved_indices = segment_indices
            new_positions = rotated_positions
            old_positions = [state.chain[idx] for idx in segment_indices]
            
            # Check self-avoidance
            # The moved segment must not collide with:
            # 1. Other parts of the chain (non-moved)
            # 2. The pivot (already handled by rotation)
            # 3. Other parts of the moved segment (should be fine if rotation is rigid)
            
            # Check against non-moved parts
            valid = True
            for new_pos in new_positions:
                if new_pos in state.pos_to_idx:
                    # Check if the index at this position is in the moved segment
                    idx_at_pos = state.pos_to_idx[new_pos]
                    if idx_at_pos not in moved_indices:
                        valid = False
                        break
            if not valid:
                continue
            
        else: # Slide move
            # Select segment
            # Slide moves are trickier. Let's skip for now and rely on pivots.
            # Or implement a simple slide: shift a segment by 1 unit.
            # For simplicity, let's just do pivots for now, they are very effective.
            # If we want slides:
            # i = rng.randint(1, n-2)
            # L = rng.randint(1, n-i-1)
            # direction = rng.choice([(1,0), (-1,0), (0,1), (0,-1)])
            # ... check if shifted segment is free.
            continue
            
        if not moved_indices:
            continue
        
        # Calculate energy change
        delta_e = state._get_contact_change(moved_indices, new_positions)
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temperature, 0.001)):
            # Accept move
            state._apply_move(moved_indices, new_positions)
            state.energy += delta_e
            
            if state.energy < best_energy:
                best_energy = state.energy
                best_chain = state.chain.copy()
        else:
            # Reject move (no change to state)
            pass
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best_chain, best_energy = None, 0
    
    remaining = _remaining_time()
    
    # Run multiple annealing attempts
    # Allocate time based on remaining budget
    # Short sequences need less time, long ones more.
    n = len(sequence)
    
    # Heuristic: 2 seconds for small, 10 seconds for large
    time_per_run = min(8.0, remaining / 5)
    
    num_attempts = max(1, int(remaining / time_per_run))
    
    for attempt in range(num_attempts):
        if _remaining_time() < 5.0:
            break
        
        # Vary the seed for each attempt
        rng_attempt = random.Random(12345 + len(sequence) + attempt)
        
        chain, energy = _anneal(sequence, dimension, rng_attempt, time_per_run)
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: simple greedy growth
        chain = _initial_chain(sequence, dimension, rng)
        if chain is None:
            best_chain = [(i, 0) for i in range(len(sequence))]
        else:
            best_chain = chain
            best_energy = _compute_energy_fast(sequence, best_chain)
    
    return [list(p) for p in best_chain]

def _compute_energy_fast(sequence, chain):
    """Compute energy using a set for O(1) lookups."""
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

# EVOLVE-BLOCK-END
