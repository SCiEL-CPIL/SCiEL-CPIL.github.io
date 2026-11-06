"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 205.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


class FastChain:
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.chain = list(chain)
        self.n = len(self.chain)
        self.occupied = {p: i for i, p in enumerate(self.chain)}
        self.energy = _energy(sequence, self.chain)
        
    def _delta_energy(self, moving_indices, new_positions):
        """Calculate energy change for moving residues to new positions.
        moving_indices: list of indices in the chain that are moving.
        new_positions: list of new (x,y) tuples corresponding to moving_indices.
        """
        delta = 0
        # 1. Remove old contacts involving moving residues
        # A contact is (i, j) where i < j, |i-j| > 1, and they are neighbors.
        # If i is moving, we lose contacts (i, j) for all j in occupied neighbors.
        # If j is moving, we lose contacts (i, j) for all i in occupied neighbors.
        # We must be careful not to double count if both i and j are moving.
        
        # Strategy: 
        # For each moving index i, check its 4 neighbors in the OLD occupied set.
        # If neighbor is j:
        #   - If j is NOT moving: contact (min(i,j), max(i,j)) is lost.
        #   - If j IS moving: contact is lost, but we'll handle it via the moving set?
        # Actually, easier: 
        # Old energy contribution of moving set M:
        # Sum over all pairs (i, j) in M such that they are neighbors in OLD config.
        # Plus Sum over all pairs (i, j) where i in M, j not in M, and they are neighbors in OLD config.
        
        # Let's just iterate over moving indices and their old neighbors.
        moving_set = set(moving_indices)
        
        # Count lost contacts
        lost = 0
        for i in moving_indices:
            if self.seq[i] != 'H':
                continue
            x, y = self.chain[i]
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = self.occupied.get(q)
                if j is not None and abs(i - j) > 1:
                    if self.seq[j] == 'H':
                        # Contact exists
                        # If j is in moving_set, we count it once for i and once for j?
                        # We want to count the pair (i,j) once.
                        # If i < j, count it when processing i.
                        # If i > j, count it when processing j.
                        if i < j:
                            lost += 1
                        # If j is moving, when we process j, we will see i as neighbor.
                        # Since j > i, we won't count it there (condition i<j fails? No, we check i<j in loop).
                        # Wait, the condition is "if i < j: lost += 1".
                        # So if we process i and see j (j>i), we count it.
                        # If we process j and see i (i<j), we don't count it.
                        # This correctly counts each pair exactly once.
                        
        # Count gained contacts
        gained = 0
        # We need to check new positions against:
        # 1. Other moving residues in their NEW positions.
        # 2. Fixed residues in their OLD positions.
        
        # Create a temporary map for new positions of moving residues
        new_pos_map = {idx: pos for idx, pos in zip(moving_indices, new_positions)}
        
        for i in moving_indices:
            if self.seq[i] != 'H':
                continue
            x, y = new_pos_map[i]
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # Check if q is occupied by a fixed residue
                # If it is, and it's H, and not adjacent in chain, it's a new contact.
                # But we must exclude other moving residues from 'occupied' check?
                # The 'occupied' dict still holds OLD positions.
                # So if q is in occupied, it's a fixed residue (or a moving residue that didn't move? No, all moving residues change position or we revert).
                # Actually, if a moving residue stays in the same spot, it's not really "moving" in a way that changes energy relative to fixed, but it might change relative to other moving ones.
                # Let's assume all moving residues change position or we handle the "same position" case carefully.
                
                j = self.occupied.get(q)
                if j is not None and abs(i - j) > 1:
                    # j is a fixed residue (since q is old position, and moving residues are leaving old positions)
                    if self.seq[j] == 'H':
                        if i < j:
                            gained += 1
                        # Note: if j is moving, its old position is q, but it's moving away. So q is not occupied by j in the NEW state.
                        # So we only count contacts with FIXED residues here.
                        
                # Check if q is occupied by another moving residue in its NEW position
                # We need to check if any k in moving_indices has new_pos_map[k] == q
                # This is O(M) per neighbor, so O(4*M^2) total. M is usually small for pivots? 
                # For large pivots, this is slow.
                # Optimization: Build a set of new positions for moving residues.
        
        # Re-do gained contacts with a set for efficiency
        new_positions_set = set(new_positions)
        
        # 1. Contacts with fixed residues
        for i in moving_indices:
            if self.seq[i] != 'H':
                continue
            x, y = new_pos_map[i]
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = self.occupied.get(q)
                if j is not None and abs(i - j) > 1:
                    if self.seq[j] == 'H':
                        if i < j:
                            gained += 1
                            
        # 2. Contacts between moving residues in new positions
        # For each pair (i, k) in moving_indices where i < k:
        # Check if new_pos_map[i] and new_pos_map[k] are neighbors.
        # This is O(M^2).
        # If M is large (e.g. 50), this is 2500 checks. Acceptable?
        # Maybe. But we can do better.
        # For each i in moving_indices, check its 4 neighbors in new_positions_set.
        # If neighbor is in new_positions_set, find the index k.
        # We need a map from new_pos to index for moving residues.
        new_pos_to_idx = {pos: idx for idx, pos in zip(moving_indices, new_positions)}
        
        for i in moving_indices:
            if self.seq[i] != 'H':
                continue
            x, y = new_pos_map[i]
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                k = new_pos_to_idx.get(q)
                if k is not None and abs(i - k) > 1:
                    if self.seq[k] == 'H':
                        if i < k:
                            gained += 1
                            
        return gained - lost

    def apply_pivot(self, pivot_idx, angle):
        """
        Apply a pivot move. Returns True if successful, False if collision.
        If successful, updates chain, occupied, energy.
        """
        n = self.n
        if pivot_idx < 1 or pivot_idx >= n - 1:
            return False
            
        pivot = self.chain[pivot_idx]
        px, py = pivot
        
        # Determine moving segment
        moving_indices = list(range(pivot_idx + 1, n))
        old_positions = [self.chain[i] for i in moving_indices]
        
        # Calculate new positions
        new_positions = []
        for p in old_positions:
            dx, dy = p[0] - px, p[1] - py
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else: # 180
                ndx, ndy = -dx, -dy
            new_positions.append((px + ndx, py + ndy))
        
        # Check collisions with fixed part
        # Fixed part is indices 0..pivot_idx
        # We can check if any new_position is in occupied and its index is <= pivot_idx
        # Also check if any new_position collides with another new_position (shouldn't happen if rigid rotation of self-avoiding walk)
        
        # Optimization: 
        # 1. Check if any new_pos is in occupied with index <= pivot_idx.
        # 2. Check if any new_pos is in the set of new_positions (internal collision). 
        #    Since rotation is rigid and old was self-avoiding, new is self-avoiding. So no internal collision.
        
        new_pos_set = set(new_positions)
        
        # Collision check
        for new_p in new_positions:
            idx = self.occupied.get(new_p)
            if idx is not None:
                if idx <= pivot_idx:
                    # Collision with fixed part
                    return False
                # If idx > pivot_idx, it's a moving residue. 
                # Since new_pos_set contains unique positions (rigid rotation of SAW), 
                # and old positions were unique, this shouldn't happen unless 
                # the rotation maps a point to another point's OLD position?
                # Wait, if new_p is in occupied, it means some residue is currently at new_p.
                # If that residue is moving, it will leave new_p. 
                # But does any other moving residue end up at new_p?
                # No, because new_positions are unique.
                # So if new_p is in occupied, and the occupant is moving, 
                # then the occupant was at new_p and is moving away. 
                # But we are moving TO new_p. 
                # So we are swapping? No, we are moving a segment.
                # If residue k (moving) was at new_p, and residue m (moving) is moving to new_p, 
                # then k must move away from new_p. 
                # But k's new position is new_positions[k-1].
                # If new_positions[k-1] == new_p, then k stays? 
                # This is getting complicated.
                # Simpler: The set of new positions must be disjoint from the set of fixed positions.
                # Fixed positions are { self.chain[i] for i in range(pivot_idx+1) }
                # We can just check:
                # for new_p in new_positions:
                #    if new_p in fixed_set: return False
                
                # Let's use a set for fixed positions for O(1) check.
                # But building the set is O(N). 
                # We can just check occupied[new_p] > pivot_idx.
                # If occupied[new_p] is not None and occupied[new_p] <= pivot_idx, collision.
                # If occupied[new_p] is not None and occupied[new_p] > pivot_idx, 
                # it means a moving residue is currently at new_p. 
                # Since the move is a rigid rotation, the set of new positions is the same size as old positions.
                # The old positions were disjoint from fixed.
                # The new positions are a rotation of old positions.
                # So new positions are disjoint from each other.
                # The only potential collision is with fixed.
                # So if occupied[new_p] is not None, and it's a moving residue, 
                # that means new_p is one of the OLD positions of a moving residue.
                # Does that cause a problem?
                # No, because that moving residue is leaving new_p.
                # And we are placing a different moving residue at new_p?
                # No, the mapping is one-to-one.
                # So if new_p is in the set of new positions, and it is also in the set of old positions,
                # it means the segment is rotating back onto itself? 
                # This is impossible for a rigid rotation of a SAW unless the segment is symmetric?
                # Actually, it's possible that new_p == old_p for some p? 
                # Only if angle is 0 or 360, which we don't do.
                # Or if the segment has symmetry.
                # In any case, the critical check is collision with FIXED part.
                pass
        
        # Re-implement collision check clearly
        fixed_set = set(self.chain[:pivot_idx + 1])
        for new_p in new_positions:
            if new_p in fixed_set:
                return False
                
        # Calculate energy delta
        delta_e = self._delta_energy(moving_indices, new_positions)
        
        # Apply move
        for i, new_p in zip(moving_indices, new_positions):
            old_p = self.chain[i]
            del self.occupied[old_p]
            self.occupied[new_p] = i
            self.chain[i] = new_p
            
        self.energy += delta_e
        return True

    def revert_pivot(self, pivot_idx, angle):
        """
        Revert the last pivot move.
        """
        n = self.n
        if pivot_idx < 1 or pivot_idx >= n - 1:
            return False
            
        pivot = self.chain[pivot_idx]
        px, py = pivot
        
        moving_indices = list(range(pivot_idx + 1, n))
        old_positions = [self.chain[i] for i in moving_indices]
        
        # Reverse rotation
        reverse_angle = -angle
        if reverse_angle == 180:
            reverse_angle = -180 # Just to be consistent, -180 is same as 180
        
        new_positions = []
        for p in old_positions:
            dx, dy = p[0] - px, p[1] - py
            if reverse_angle == 90:
                ndx, ndy = -dy, dx
            elif reverse_angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_positions.append((px + ndx, py + ndy))
            
        # No collision check needed for revert (should be valid)
        delta_e = self._delta_energy(moving_indices, new_positions)
        
        for i, new_p in zip(moving_indices, new_positions):
            old_p = self.chain[i]
            del self.occupied[old_p]
            self.occupied[new_p] = i
            self.chain[i] = new_p
            
        self.energy += delta_e
        return True

    def apply_slither(self, rng):
        """
        Slither move: Move the last residue to a different valid neighbor of the second-to-last.
        Returns True if successful.
        """
        n = self.n
        if n < 2:
            return False
            
        prev = self.chain[-2]
        last = self.chain[-1]
        last_idx = n - 1
        
        # Find valid candidates for new last position
        candidates = []
        for site in _neighbours(prev):
            if site == last:
                continue
            if site not in self.occupied:
                candidates.append(site)
                
        if not candidates:
            return False
            
        new_last = rng.choice(candidates)
        
        # Update energy
        # Only last residue moves.
        delta_e = self._delta_energy([last_idx], [new_last])
        
        # Update data structures
        del self.occupied[last]
        self.occupied[new_last] = last_idx
        self.chain[-1] = new_last
        self.energy += delta_e
        
        return True

    def revert_slither(self, old_last):
        """
        Revert the last slither move.
        """
        n = self.n
        if n < 2:
            return False
            
        prev = self.chain[-2]
        last = self.chain[-1]
        last_idx = n - 1
        
        # Check if old_last is a valid neighbor of prev and not occupied
        if old_last not in _neighbours(prev):
            return False
        if old_last in self.occupied:
            return False
            
        delta_e = self._delta_energy([last_idx], [old_last])
        
        del self.occupied[last]
        self.occupied[old_last] = last_idx
        self.chain[-1] = old_last
        self.energy += delta_e
        
        return True


def _grow_greedy(sequence, bias, rng):
    """
    Greedy growth with bias towards contacts and compactness.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        curr = chain[-1]
        options = []
        
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)
        
        for site in _neighbours(curr):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    q = (sx+dx, sy+dy)
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing local search using FastChain.
    """
    fc = FastChain(sequence, initial_chain)
    best_chain = list(fc.chain)
    best_energy = fc.energy
    
    n = fc.n
    
    # Cooling schedule
    T = 5.0
    T_min = 0.01
    
    # Estimate iterations per second to adjust cooling?
    # Or just use time-based loop.
    
    start_time = time.time()
    
    # Adaptive cooling
    # We want to reach T_min near the end of time_limit.
    # If we assume ~10000 iterations in 1 sec, then...
    # Let's just use a fixed cooling rate that works reasonably well.
    # For N=100, 0.9999 is slow. For N=20, 0.9995 is fast.
    if n > 80:
        cooling = 0.99995
    elif n > 50:
        cooling = 0.9999
    else:
        cooling = 0.9995
        
    while time.time() - start_time < time_limit:
        if _time_left() < 10:
            break
            
        r = rng.random()
        
        if r < 0.85:
            # Pivot
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            
            if fc.apply_pivot(pivot_idx, angle):
                delta = fc.energy - best_energy # No, delta is change from current
                # We need delta from current state before move.
                # FastChain updates energy in place.
                # So we need to store current energy before move.
                
                # Let's restructure:
                # current_energy = fc.energy
                # if fc.apply_pivot(...):
                #    new_energy = fc.energy
                #    delta = new_energy - current_energy
                #    if delta < 0 or rng.random() < exp(-delta/T):
                #        # accept
                #    else:
                #        fc.revert_pivot(pivot_idx, angle)
                
                # My current code above doesn't store current_energy.
                # Let's fix the loop logic.
                
                # (This is a draft thought, the actual code below will be correct)
                pass
        else:
            # Slither
            old_last = fc.chain[-1]
            if fc.apply_slither(rng):
                pass

    # Correct Implementation of Loop
    # Reset fc to start
    fc = FastChain(sequence, initial_chain)
    current_energy = fc.energy
    best_chain = list(fc.chain)
    best_energy = fc.energy
    
    while time.time() - start_time < time_limit:
        if _time_left() < 10:
            break
            
        r = rng.random()
        
        if r < 0.85:
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            
            if fc.apply_pivot(pivot_idx, angle):
                new_energy = fc.energy
                delta = new_energy - current_energy
                
                if delta < 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                    # Accept
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(fc.chain)
                else:
                    # Reject
                    fc.revert_pivot(pivot_idx, angle)
        else:
            old_last = fc.chain[-1]
            if fc.apply_slither(rng):
                new_energy = fc.energy
                delta = new_energy - current_energy
                
                if delta < 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(fc.chain)
                else:
                    fc.revert_slither(old_last)
                    
        T *= cooling
        
    return best_chain


def fold(sequence, dimension):
    """
    Main entry point.
    """
    seq_len = len(sequence)
    remaining = _time_left()
    
    if remaining < 5:
        return [[i, 0] for i in range(seq_len)]
        
    rng = random.Random(12345 + seq_len)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Initialization
    init_time_limit = min(2.0, remaining * 0.1)
    init_start = time.time()
    candidates = []
    
    num_greedy = 50 # Increased diversity
    for i in range(num_greedy):
        if time.time() - init_start > init_time_limit:
            break
        bias = 0.5 + rng.random() * 2.0
        chain = _grow_greedy(sequence, bias, rng)
        if chain:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
            
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5] # Keep top 5
    
    if not top_candidates:
        best_chain = [[i, 0] for i in range(seq_len)]
        best_energy = _energy(sequence, best_chain)
        return best_chain
        
    # Phase 2: Local Search
    search_time_limit = min(remaining * 0.8, 15.0) # Cap search time per candidate? No, total search time.
    # Actually, we have total remaining time.
    # Let's allocate 80% of remaining to search.
    total_search_time = remaining * 0.8
    search_start = time.time()
    
    for e, chain in top_candidates:
        if time.time() - search_start > total_search_time:
            break
        if _time_left() < 15:
            break
            
        # Allocate time to this candidate
        # If we have 5 candidates, give each 1/5 of the time?
        # Or just run until time runs out?
        # Better to run all of them for a shorter time, or one for a longer time?
        # Usually, polishing the best few is better.
        # Let's give each candidate an equal share of the search time.
        time_per_candidate = total_search_time / len(top_candidates)
        
        improved = _local_search(sequence, chain, time_per_candidate, rng)
        
        if improved:
            e_imp = _energy(sequence, improved)
            if best_chain is None or e_imp < best_energy:
                best_energy = e_imp
                best_chain = improved

    # Phase 3: Polish
    if best_chain and _time_left() > 15:
        polish_time = min(3.0, _time_left() - 15)
        # Run local search again on the best result with lower T?
        # Or just hill climb.
        # Let's run a short SA with low T.
        improved = _local_search(sequence, best_chain, polish_time, rng)
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy:
                best_energy = e_imp
                best_chain = improved

    if best_chain is None:
        best_chain = [[i, 0] for i in range(seq_len)]
        
    return best_chain
# EVOLVE-BLOCK-END
