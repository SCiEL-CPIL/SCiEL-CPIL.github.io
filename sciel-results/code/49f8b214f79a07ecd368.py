"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to keep before hard limit


def _setup_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 235.0  # 235s budget, leaving 5s margin


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, rng, beam_width=50, max_time=10.0):
    """Beam search with dynamic width and improved heuristic."""
    start = (0,) * dimension
    # Beam entry: (chain_list, occupied_dict, current_energy)
    beams = [([start], {start: 0}, 0)]
    n = len(sequence)
    
    t0 = time.time()
    for i in range(1, n):
        if time.time() - t0 > max_time:
            break
        
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            idx = occupied[q]
                            if idx < i - 1 and sequence[idx] == "H":
                                gain += 1
                # Count potential future H contacts (look-ahead)
                potential = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q not in occupied:
                            # Check if future residues are H
                            for j in range(i+2, min(i+6, n)):
                                if sequence[j] == "H":
                                    potential += 1
                                    break
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Heuristic: prefer high gain, then potential, then free space
                score = gain * 100 + potential * 5 + free * 10 + rng.random() * 2
                options.append((score, site, gain))
            
            if not options:
                continue
            
            options.sort(key=lambda x: -x[0])
            # Keep top candidates per beam
            keep = min(beam_width, max(1, beam_width // 2))
            for score, site, gain in options[:keep]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy - gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    best = min(beams, key=lambda x: x[2])
    return best[0]


class _FastLocalSearch:
    def __init__(self, sequence, chain, seed=42):
        self.sequence = sequence
        self.chain = list(chain)
        self.n = len(self.chain)
        self.rng = random.Random(seed)
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.energy = _compute_energy(self.sequence, self.chain)
        self.h_positions = set(i for i in range(self.n) if self.sequence[i] == 'H')
        
    def _get_contacts_involving(self, indices):
        """
        Get the set of contact pairs (i, j) with i < j, 
        where at least one of i or j is in indices, 
        and both are H, and they are neighbors on lattice.
        """
        contacts = set()
        for i in indices:
            if i not in self.h_positions:
                continue
            p = self.chain[i]
            for q in _neighbours(p):
                j = self.occupied.get(q)
                if j is not None and j > i and j in self.h_positions and abs(j - i) > 1:
                    contacts.add((i, j))
        return contacts

    def _remove_contacts(self, indices):
        """Remove contacts involving any index in indices from energy count."""
        # We don't store the set of all contacts, we just recalculate the delta
        # But to be safe and fast, let's just compute the change in energy directly.
        pass # Placeholder, logic moved to move application

    def _apply_move(self, move_indices, new_positions):
        """
        Apply a move.
        move_indices: list of indices in self.chain that are moving.
        new_positions: dict mapping index -> new_tuple_position.
        Returns True if move was applied and valid, False otherwise.
        """
        # 1. Check self-avoidance
        # The new positions for moved indices must not collide with unmoved indices
        # or with each other.
        old_positions = {i: self.chain[i] for i in move_indices}
        
        # Check collisions with unmoved
        for i in move_indices:
            new_p = new_positions[i]
            if new_p in self.occupied:
                # It's a collision if the occupant is NOT one of the moving indices
                occ_idx = self.occupied[new_p]
                if occ_idx not in move_indices:
                    return False
            # Check collision with other moving indices (handled by set of new positions)
            
        # Check collisions among moving indices
        new_pos_set = set(new_positions.values())
        if len(new_pos_set) != len(move_indices):
            return False

        # 2. Check connectivity (unit steps)
        # For each moved index i, check its neighbors in the chain (i-1, i+1)
        # If neighbor is moved, check step between new positions.
        # If neighbor is NOT moved, check step between new pos and old pos.
        for i in move_indices:
            for neighbor_idx in (i-1, i+1):
                if 0 <= neighbor_idx < self.n:
                    if neighbor_idx in new_positions:
                        # Both moved
                        p_new = new_positions[i]
                        n_new = new_positions[neighbor_idx]
                        if abs(p_new[0]-n_new[0]) + abs(p_new[1]-n_new[1]) + abs(p_new[2]-n_new[2]) != 1:
                            return False
                    else:
                        # Only i moved
                        p_new = new_positions[i]
                        n_old = self.chain[neighbor_idx]
                        if abs(p_new[0]-n_old[0]) + abs(p_new[1]-n_old[1]) + abs(p_new[2]-n_old[2]) != 1:
                            return False
                            
        # 3. Calculate Energy Delta
        # Old energy contribution of contacts involving moved indices
        # A contact is a pair (i, j) with i<j, both H, neighbors.
        # We need to subtract contacts that are broken and add contacts that are formed.
        
        # Identify all contacts involving any moved index BEFORE the move
        old_contacts = set()
        for i in move_indices:
            if i not in self.h_positions:
                continue
            p = self.chain[i]
            for q in _neighbours(p):
                j = self.occupied.get(q)
                if j is not None and j > i and j in self.h_positions and abs(j - i) > 1:
                    old_contacts.add((i, j))
                    
        # Apply move to temporary structures to count new contacts?
        # Or just simulate:
        # Temporarily update self.chain and self.occupied? 
        # Better: Calculate new contacts involving moved indices based on new positions.
        
        # We need to be careful: A contact (i, j) where both i and j are moved 
        # might still be a contact if they are still neighbors in new positions.
        # A contact (i, j) where i is moved and j is not moved 
        # is broken if they are no longer neighbors, formed if they are now neighbors.
        
        # Let's just compute the new contacts involving moved indices.
        new_contacts = set()
        
        # We need a temporary view of the chain/occupied for the moved parts
        # to check neighbors correctly.
        
        # Optimization: 
        # 1. Remove old contacts involving moved indices from energy count.
        # 2. Apply move to self.chain and self.occupied.
        # 3. Add new contacts involving moved indices to energy count.
        
        # Step 3a: Remove old
        # We can't just subtract len(old_contacts) because some old contacts 
        # might involve two moved indices that remain in contact.
        
        # Let's do it explicitly:
        # 1. Identify all pairs (i, j) involving moved indices that are currently contacts.
        # 2. Identify all pairs (i, j) involving moved indices that will be contacts.
        # 3. Delta = New - Old.
        
        # To find "will be contacts", we need to know the new positions.
        # We can check for each moved index i, and each neighbor q of new_pos[i],
        # who is at q? 
        # If q is occupied by j, and j is not moved, we check if j is H and valid.
        # If q is occupied by j, and j IS moved, we check if new_pos[j] == q (which it is, by definition) 
        # and if they are distinct indices.
        
        # Actually, simpler:
        # 1. Compute set of all contact pairs involving moved indices in OLD state.
        # 2. Compute set of all contact pairs involving moved indices in NEW state.
        # 3. Delta = |New| - |Old|.
        
        # To compute NEW state contacts involving moved indices:
        # For each moved index i:
        #   For each neighbor q of new_pos[i]:
        #     Find j such that new_pos[j] == q (if j is moved) or self.occupied[q] == j (if j is not moved).
        #     If found, and i < j, and both H, and abs(i-j)>1: add (i,j).
        
        # We need a map of new positions for moved indices.
        new_pos_map = new_positions
        
        # Build a temporary occupied map for moved indices to resolve neighbors
        # Note: Unmoved indices remain in self.occupied.
        
        new_contact_set = set()
        for i in move_indices:
            if i not in self.h_positions:
                continue
            new_p = new_pos_map[i]
            for q in _neighbours(new_p):
                # Who is at q in the NEW state?
                j = None
                if q in new_pos_map.values():
                    # Find which moved index is at q
                    for k in move_indices:
                        if new_pos_map[k] == q:
                            j = k
                            break
                else:
                    # Not a moved index, so it's an unmoved index
                    j = self.occupied.get(q)
                    
                if j is not None:
                    if i < j and j in self.h_positions and abs(i - j) > 1:
                        new_contact_set.add((i, j))
                        
        # Now we have old_contact_set and new_contact_set.
        # But wait, old_contact_set was calculated using self.occupied (old state).
        # new_contact_set uses new_pos_map and self.occupied (old state for unmoved).
        
        # Is it possible that a contact (i, j) where NEITHER i nor j is moved is affected?
        # No, because their positions don't change.
        # Is it possible that a contact involving a moved index i and an unmoved index j 
        # is missed in new_contact_set?
        # No, we check all neighbors of new_pos[i]. If j is unmoved, it is in self.occupied.
        
        # So Delta = len(new_contact_set) - len(old_contacts)
        
        delta = len(new_contact_set) - len(old_contacts)
        
        # If we accept the move, we update state.
        # But we need to actually update self.chain and self.occupied.
        
        # Save old state for rollback if needed? 
        # The caller (search loop) should handle rollback or we return the delta 
        # and let the caller decide. 
        # However, to keep it simple, let's have this function APPLY the move 
        # and return the delta, assuming the caller has verified validity (which we did above).
        
        # Actually, the caller (search loop) picks a move, calls a "try" function.
        # Let's make this function "apply_move_inplace" and return delta.
        
        # Update self.chain
        for i in move_indices:
            self.chain[i] = new_pos_map[i]
            
        # Update self.occupied
        for i in move_indices:
            old_p = old_positions[i]
            new_p = new_pos_map[i]
            # Remove old
            # self.occupied[old_p] should be i
            # But if two moved indices swap? 
            # In our moves (turns, shifts), indices don't swap positions in a way that 
            # creates ambiguity in the dict update if we just assign new.
            # But we must remove the old entries first to avoid overwriting if 
            # an old position is a new position of another?
            # In 180 turn: i-1 and i+1 move. 
            # In 3-shift: i-1, i, i+1 move.
            # It is possible that new_pos[i-1] == old_pos[i+1]? 
            # If so, we must be careful with dict updates.
            
            # Safe way:
            # 1. Delete all old positions from occupied
            # 2. Add all new positions to occupied
            
        for i in move_indices:
            del self.occupied[old_positions[i]]
        for i in move_indices:
            self.occupied[new_pos_map[i]] = i
            
        self.energy += delta
        
        return delta


def _local_search(sequence, chain, max_iters=2000, seed=42):
    """Improve a conformation via fast local search."""
    n = len(chain)
    if n < 4:
        return chain
    
    searcher = _FastLocalSearch(sequence, chain, seed)
    
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        improved = False
        
        for _attempt in range(50):
            move_type = searcher.rng.random()
            i = searcher.rng.randint(1, n - 2)
            
            if move_type < 0.6:
                # 180-degree turn at i
                # Moves i-1 and i+1. i stays.
                a, b, c = searcher.chain[i-1], searcher.chain[i], searcher.chain[i+1]
                new_a = (2*b[0] - c[0], 2*b[1] - c[1], 2*b[2] - c[2])
                new_c = (2*b[0] - a[0], 2*b[1] - a[1], 2*b[2] - a[2])
                
                if new_a == new_c or new_a == b or new_c == b:
                    continue
                
                move_indices = [i-1, i+1]
                new_positions = {i-1: new_a, i+1: new_c}
                
                # Check validity and apply
                # We need to check if new_a or new_c collide with unmoved parts
                # The _apply_move function does this check.
                # But we need to pass the indices and new positions.
                
                # Let's wrap the apply logic to handle rollback if invalid
                # Or, _apply_move returns False if invalid.
                
                # Since _apply_modify modifies state, we need to be careful.
                # Let's create a helper that checks validity first.
                
                # Check self-avoidance manually for speed?
                # _apply_move does it.
                
                # To allow rollback, we need to save old state.
                # Or, we can just check validity before calling apply.
                
                # Let's add a validity check function to the searcher or here.
                
                # Validity Check:
                valid = True
                # 1. Collisions with unmoved
                for idx in move_indices:
                    np = new_positions[idx]
                    if np in searcher.occupied:
                        occ_idx = searcher.occupied[np]
                        if occ_idx not in move_indices:
                            valid = False
                            break
                # 2. Collisions between moved
                if valid and len(set(new_positions.values())) != len(move_indices):
                    valid = False
                # 3. Connectivity
                if valid:
                    for idx in move_indices:
                        for nb in (idx-1, idx+1):
                            if 0 <= nb < n:
                                if nb in new_positions:
                                    p1 = new_positions[idx]
                                    p2 = new_positions[nb]
                                    if manhattan(p1, p2) != 1:
                                        valid = False
                                        break
                                else:
                                    p1 = new_positions[idx]
                                    p2 = searcher.chain[nb]
                                    if manhattan(p1, p2) != 1:
                                        valid = False
                                        break
                        if not valid:
                            break
                            
                if not valid:
                    continue
                    
                # Apply
                delta = searcher._apply_move(move_indices, new_positions)
                # _apply_move updates energy
                improved = True
            else:
                # 3-residue shift
                a, b, c = searcher.chain[i-1], searcher.chain[i], searcher.chain[i+1]
                moved = False
                for axis in range(3):
                    if moved: break
                    for step in (-1, 1):
                        new_a = list(a)
                        new_b = list(b)
                        new_c = list(c)
                        new_a[axis] += step
                        new_b[axis] += step
                        new_c[axis] += step
                        new_a = tuple(new_a)
                        new_b = tuple(new_b)
                        new_c = tuple(new_c)
                        
                        move_indices = [i-1, i, i+1]
                        new_positions = {i-1: new_a, i: new_b, i+1: new_c}
                        
                        # Validity Check
                        valid = True
                        for idx in move_indices:
                            np = new_positions[idx]
                            if np in searcher.occupied:
                                occ_idx = searcher.occupied[np]
                                if occ_idx not in move_indices:
                                    valid = False
                                    break
                        if valid and len(set(new_positions.values())) != len(move_indices):
                            valid = False
                        if valid:
                            for idx in move_indices:
                                for nb in (idx-1, idx+1):
                                    if 0 <= nb < n:
                                        if nb in new_positions:
                                            p1 = new_positions[idx]
                                            p2 = new_positions[nb]
                                            if manhattan(p1, p2) != 1:
                                                valid = False
                                                break
                                        else:
                                            p1 = new_positions[idx]
                                            p2 = searcher.chain[nb]
                                            if manhattan(p1, p2) != 1:
                                                valid = False
                                                break
                                if not valid:
                                    break
                                
                        if not valid:
                            continue
                            
                        delta = searcher._apply_move(move_indices, new_positions)
                        improved = True
                        moved = True
                        break
                        
        if not improved:
            if it > 10:
                break
                
    return searcher.chain


def _spiral_fold(sequence, dimension, seed=0):
    """Create a spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    dir_idx = rng.randint(0, 5)
    
    for i in range(1, n):
        placed = False
        order = [dir_idx % 6] + [d for d in range(6) if d != dir_idx % 6]
        rng.shuffle(order[1:])
        for d_idx in order:
            d = directions[d_idx % 6]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                if rng.random() < 0.3:
                    dir_idx = (d_idx + 1) % 6
                break
        
        if not placed:
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    break
            else:
                break
    
    return chain


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using multiple strategies."""
    _setup_deadline()
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple spiral initializations
    for seed in range(5):
        if _time_left() < 3.0:
            break
        chain1 = _spiral_fold(sequence, dimension, seed=seed)
        if len(chain1) == n:
            e1 = _compute_energy(sequence, chain1)
            if best_chain is None or e1 < best_energy:
                best_chain = chain1
                best_energy = e1
    
    # Strategy 2: Beam search with different widths
    if _time_left() > 5.0:
        for bw in [30, 80, 150]:
            if _time_left() < 3.0:
                break
            max_t = min(8.0, _time_left() * 0.25)
            chain2 = _beam_search(sequence, dimension, rng, beam_width=bw, max_time=max_t)
            if chain2 and len(chain2) == n:
                e2 = _compute_energy(sequence, chain2)
                if best_chain is None or e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
    
    # Strategy 3: Local search refinement on best chain
    if best_chain is not None and _time_left() > 2.0:
        for ls_seed in range(5):
            if _time_left() < 2.0:
                break
            max_iters = min(5000, int(_time_left() * 200))
            refined = _local_search(sequence, best_chain, max_iters=max_iters, seed=ls_seed + n)
            if refined and len(refined) == n:
                er = _compute_energy(sequence, refined)
                if er < best_energy:
                    best_chain = refined
                    best_energy = er
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
