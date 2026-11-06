"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time

# Constants for Simulated Annealing
SA_START_TEMP = 5.0
SA_COOLING_RATE = 0.995
SA_MIN_TEMP = 0.01

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy of a chain. Lower is better (more negative)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _is_valid_move(chain, occupied, new_pos, old_pos=None, exclude_indices=None):
    """
    Check if a move to new_pos is valid (self-avoiding).
    occupied: set of tuples
    old_pos: if moving, the old position will be freed
    exclude_indices: indices of residues that are not part of the 'moving' part but are in the chain.
    Actually, simpler: just check if new_pos is in occupied, excluding the residue itself if it's a move.
    """
    if old_pos is not None:
        # If we are moving a residue, we can move to its old position? No, it's already occupied by itself.
        # But if we are rotating a segment, the positions change.
        pass
    
    # For a simple move of residue i to new_pos:
    # new_pos must not be in occupied, unless new_pos == old_pos (no move)
    # But occupied contains old_pos. So we must check:
    # Is new_pos in occupied?
    # If new_pos == old_pos, it is in occupied, but that's fine (no change).
    # If new_pos != old_pos, it must NOT be in occupied.
    
    if new_pos == old_pos:
        return True
    
    if new_pos in occupied:
        return False
    
    return True

def _apply_move(chain, occupied, i, new_pos):
    """Apply a move of residue i to new_pos. Returns True if successful."""
    old_pos = chain[i]
    if new_pos == old_pos:
        return True
    if new_pos in occupied:
        return False
    
    # Update structures
    del occupied[old_pos]
    occupied[new_pos] = i
    chain[i] = new_pos
    return True

def _revert_move(chain, occupied, i, new_pos, old_pos):
    """Revert a move of residue i from new_pos to old_pos."""
    if new_pos == old_pos:
        return
    del occupied[new_pos]
    occupied[old_pos] = i
    chain[i] = old_pos

def _pivot_move(chain, occupied, sequence, rng):
    """
    Perform a pivot move: rotate a segment of the chain.
    Returns (new_chain, new_occupied, delta_energy) or (None, None, 0) if failed.
    """
    n = len(chain)
    if n < 4:
        return None, None, 0
    
    # Choose pivot point i (1 to n-2)
    i = rng.randint(1, n - 2)
    # Choose length of segment to rotate (from i+1 to j)
    # Max length is n-1-i
    max_len = n - 1 - i
    if max_len < 1:
        return None, None, 0
    length = rng.randint(1, max_len)
    j = i + length
    
    # The segment to rotate is chain[i+1 : j+1]
    # We rotate it around chain[i].
    # The new positions must be valid.
    
    # Let's try a few random rotations.
    # A rotation in 3D can be complex. 
    # Simpler: try to reflect the segment through the plane defined by chain[i] and the vector chain[i+1]-chain[i]?
    # Or just try to place the segment in a random orientation?
    
    # Alternative: "Criss-cross" move.
    # Pick i, j. Swap the directions of the segment?
    
    # Let's stick to a simpler move for now: "Flip" or "Reflect"
    # Actually, let's implement a "Move" of a sub-chain.
    # Remove sub-chain [i+1, j], and try to re-insert it at a random location?
    # That's expensive.
    
    # Let's use "Pivot" by trying all 6 possible next steps from chain[i] for the first element of the segment,
    # and then extending greedily? No, that's not a local move.
    
    # Let's implement a standard "Pivot" where we rotate the vector (chain[k] - chain[i]) for all k in i+1..j
    # by 180 degrees around an axis?
    
    # Simpler and effective: "Local Move" of a single residue is weak.
    # "Segment Move": Move a contiguous segment of the chain to a new location.
    
    # Let's implement "Segment Move":
    # 1. Pick start s and end e (s < e).
    # 2. The segment is chain[s:e+1].
    # 3. Remove it from the chain (temporarily).
    # 4. Try to place it at a new start position P, maintaining the relative coordinates.
    # 5. Check for collisions.
    
    # This is O(N^2) per move which might be slow.
    
    # Let's stick to "Single Residue Move" and "Pivot (180 deg rotation of a segment)"
    
    # Pivot: Rotate segment [i+1..j] by 180 degrees around the axis defined by chain[i] and chain[i+1]?
    # Or just try to reflect the segment across the point chain[i]?
    # If we reflect chain[k] across chain[i], new_pos = 2*chain[i] - chain[k].
    # This is a valid isometry.
    
    new_chain = list(chain)
    # We need to check if the reflected positions are free.
    # The segment is chain[i+1 : j+1]
    # Reflected positions: 2*chain[i] - chain[k]
    
    # We must remove the old positions of the segment from 'occupied' before checking,
    # because the segment moves.
    
    segment_indices = range(i + 1, j + 1)
    old_positions = [chain[k] for k in segment_indices]
    
    # Temporarily remove
    for p in old_positions:
        if p in occupied:
            del occupied[p]
            
    # Calculate new positions
    new_positions = []
    valid = True
    pivot_pos = chain[i]
    for k in segment_indices:
        old_p = chain[k]
        new_p = (2*pivot_pos[0] - old_p[0], 2*pivot_pos[1] - old_p[1], 2*pivot_pos[2] - old_p[2])
        if new_p in occupied:
            valid = False
            break
        new_positions.append(new_p)
        
    if not valid:
        # Revert removal
        for p in old_positions:
            occupied[p] = next(idx for idx, pos in enumerate(chain) if pos == p) # This is inefficient.
            # Better: just re-add. We know the indices.
            pass
        # Re-add correctly
        for k in segment_indices:
            occupied[chain[k]] = k
        return None, None, 0
        
    # Apply move
    for idx, new_p in zip(segment_indices, new_positions):
        new_chain[idx] = new_p
        occupied[new_p] = idx
        
    # Calculate delta energy
    # Energy change only happens for H residues in the segment and their neighbors.
    # This is complex to calculate incrementally without full re-calc or careful bookkeeping.
    # For simplicity and correctness, let's just re-calculate energy? 
    # No, that's O(N^2). 
    # Let's return the new chain and let the caller decide.
    # But we need delta E for acceptance.
    
    # Let's just return the new state.
    return new_chain, occupied, 0 # Delta E calculation is skipped for now, we will re-calc or estimate.
    # Actually, re-calculating energy for every move is too slow for N=100+.
    # We need incremental energy update.
    
    # Let's restart the design.
    # Use a class or careful state management.

class HPFolder:
    def __init__(self, sequence, dimension=3):
        self.sequence = sequence
        self.dimension = dimension
        self.n = len(sequence)
        self.chain = [None] * self.n
        self.occupied = {}
        self.energy = 0
        self._build_initial()
        self._calculate_energy()

    def _build_initial(self):
        # Simple greedy growth
        start = (0,) * self.dimension
        self.chain[0] = start
        self.occupied[start] = 0
        for i in range(1, self.n):
            placed = False
            # Try neighbors in random order
            neighbors = list(_neighbours(self.chain[i-1]))
            random.shuffle(neighbors)
            for p in neighbors:
                if p not in self.occupied:
                    self.chain[i] = p
                    self.occupied[p] = i
                    placed = True
                    break
            if not placed:
                # If stuck, just extend in a straight line (might collide, but initial is just a seed)
                # Actually, if stuck, we fail. Let's just pick a random unoccupied neighbor if possible.
                # If no neighbors, we are trapped.
                # For robustness, let's just place it at a random far away spot? No, must be connected.
                # If we can't place, we restart initialization.
                raise Exception("Failed to build initial chain")

    def _calculate_energy(self):
        contacts = 0
        for i in range(self.n):
            if self.sequence[i] == 'H':
                p = self.chain[i]
                for q in _neighbours(p):
                    j = self.occupied.get(q)
                    if j is not None and j > i + 1 and self.sequence[j] == 'H':
                        contacts += 1
        self.energy = -contacts

    def _get_delta_energy_for_move(self, i, new_pos):
        """
        Calculate change in energy if residue i moves to new_pos.
        Assumes new_pos is valid (not occupied).
        """
        old_pos = self.chain[i]
        if old_pos == new_pos:
            return 0
        
        delta = 0
        res_i = self.sequence[i]
        
        # 1. Remove contacts involving residue i at old_pos
        if res_i == 'H':
            for q in _neighbours(old_pos):
                j = self.occupied.get(q)
                if j is not None and j != i:
                    if self.sequence[j] == 'H':
                        # This contact is lost
                        delta += 1 # Energy increases by 1 (less negative)
        
        # 2. Add contacts involving residue i at new_pos
        if res_i == 'H':
            for q in _neighbours(new_pos):
                j = self.occupied.get(q)
                if j is not None and j != i:
                    if self.sequence[j] == 'H':
                        # This contact is gained
                        delta -= 1 # Energy decreases by 1 (more negative)
                        
        return delta

    def _try_move_single(self, rng):
        """Try to move a single residue (not 0 or n-1)."""
        if self.n < 3:
            return False
        i = rng.randint(1, self.n - 2)
        old_pos = self.chain[i]
        neighbors = list(_neighbours(old_pos))
        rng.shuffle(neighbors)
        
        for new_pos in neighbors:
            if new_pos == old_pos:
                continue
            if new_pos in self.occupied:
                continue
                
            # Check if moving i to new_pos breaks connectivity?
            # The chain is a path. Moving i to new_pos requires that new_pos is adjacent to chain[i-1] or chain[i+1]?
            # No, the chain is a fixed sequence of residues. The "walk" must be unit steps.
            # If we move residue i, it must still be adjacent to i-1 and i+1.
            # So new_pos must be a neighbor of chain[i-1] AND a neighbor of chain[i+1].
            
            prev_pos = self.chain[i-1]
            next_pos = self.chain[i+1]
            
            if new_pos not in _neighbours(prev_pos):
                continue
            if new_pos not in _neighbours(next_pos):
                continue
            
            # Valid move
            delta_e = self._get_delta_energy_for_move(i, new_pos)
            return delta_e, i, new_pos, old_pos
            
        return None

    def _try_pivot(self, rng):
        """
        Try a pivot move: rotate a segment [i+1..j] around i.
        We will try to reflect the segment across the axis defined by chain[i] and chain[i+1]?
        Or just try to place the segment in a new random orientation?
        
        Let's try a "180 degree rotation" of the segment vector relative to chain[i].
        Vector v_k = chain[k] - chain[i].
        New vector v'_k = -v_k.
        New pos = chain[i] + v'_k = 2*chain[i] - chain[k].
        """
        if self.n < 4:
            return None
        
        i = rng.randint(1, self.n - 2)
        max_len = self.n - 1 - i
        if max_len < 1:
            return None
        length = rng.randint(1, max_len)
        j = i + length
        
        segment_indices = list(range(i + 1, j + 1))
        old_positions = [self.chain[k] for k in segment_indices]
        pivot_pos = self.chain[i]
        
        new_positions = []
        for k in segment_indices:
            old_p = self.chain[k]
            new_p = (2*pivot_pos[0] - old_p[0], 2*pivot_pos[1] - old_p[1], 2*pivot_pos[2] - old_p[2])
            new_positions.append(new_p)
            
        # Check validity
        # 1. New positions must not be occupied (except by the segment itself, which we are moving)
        # 2. The first new position (new_positions[0]) must be adjacent to pivot_pos.
        #    Since it's a reflection, distance is preserved, so it is adjacent.
        # 3. The last new position (new_positions[-1]) must be adjacent to chain[j+1] (if j < n-1).
        #    Wait, the segment is [i+1, j]. The next residue is j+1.
        #    The reflection preserves distances, so dist(chain[j], chain[j+1]) = dist(new_pos_j, chain[j+1]).
        #    So if the original was valid, the new one is adjacent to j+1.
        #    BUT, we must ensure the new positions don't collide with the rest of the chain.
        
        # Temporarily remove segment from occupied
        for k in segment_indices:
            del self.occupied[self.chain[k]]
            
        valid = True
        for new_p in new_positions:
            if new_p in self.occupied:
                valid = False
                break
                
        if not valid:
            # Revert
            for k in segment_indices:
                self.occupied[self.chain[k]] = k
            return None
            
        # Apply move
        for k, new_p in zip(segment_indices, new_positions):
            self.chain[k] = new_p
            self.occupied[new_p] = k
            
        # Calculate delta energy
        # This is complex. We need to remove old contacts and add new ones for all residues in the segment.
        # Let's just re-calculate energy for simplicity? No, too slow.
        # We can calculate delta by summing deltas for each residue?
        # But the relative positions of residues within the segment are preserved?
        # No, they are reflected. So contacts between H's in the segment are preserved?
        # Yes, reflection is an isometry. So H-H contacts within the segment are unchanged.
        # Only contacts between the segment and the rest of the chain change.
        
        delta_e = 0
        # Remove old contacts between segment and rest
        for k in segment_indices:
            if self.sequence[k] == 'H':
                old_p = old_positions[k - (i+1)] # index in old_positions
                # Wait, old_positions is a list.
                for q in _neighbours(old_p):
                    j = self.occupied.get(q)
                    if j is not None and j not in segment_indices:
                        if self.sequence[j] == 'H':
                            delta_e += 1 # Lost contact
                    
        # Add new contacts between segment and rest
        for k in segment_indices:
            if self.sequence[k] == 'H':
                new_p = self.chain[k]
                for q in _neighbours(new_p):
                    j = self.occupied.get(q)
                    if j is not None and j not in segment_indices:
                        if self.sequence[j] == 'H':
                            delta_e -= 1 # Gained contact
                            
        self.energy += delta_e
        return delta_e

    def _revert_pivot(self, segment_indices, old_positions):
        for k, old_p in zip(segment_indices, old_positions):
            self.chain[k] = old_p
            self.occupied[old_p] = k
            # Note: self.occupied currently has new positions. We need to remove them first?
            # In _try_pivot, we removed old, added new.
            # To revert, we must remove new, add old.
            pass

    def step(self, rng, temp):
        """Perform one SA step."""
        # Choose move type
        # 50% Single Move, 50% Pivot
        if rng.random() < 0.5:
            res = self._try_move_single(rng)
            if res is None:
                return False
            delta_e, i, new_pos, old_pos = res
            
            # Acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                # Apply
                del self.occupied[old_pos]
                self.occupied[new_pos] = i
                self.chain[i] = new_pos
                self.energy += delta_e
                return True
            else:
                return False
        else:
            # Pivot
            # We need to store state to revert
            # Let's implement a simpler pivot: just try it, if accepted, keep, else revert.
            # But _try_pivot modifies state.
            # Let's make a copy of the relevant parts?
            # Or just implement _try_pivot to return success and store old state for revert.
            
            # For now, let's just use single moves to ensure correctness and speed.
            # Pivots are harder to implement correctly with incremental energy.
            # Let's add a "Random Restart" of a segment?
            
            # Let's try a "Segment Shift": Move a segment [s:e] by one step in a random direction.
            # This is similar to single move but for a segment.
            
            return self._try_segment_shift(rng, temp)

    def _try_segment_shift(self, rng, temp):
        if self.n < 5:
            return False
        s = rng.randint(1, self.n - 2)
        max_len = self.n - 1 - s
        if max_len < 1:
            return False
        e = rng.randint(s, min(s + max_len, self.n - 1))
        
        # Segment is [s, e]
        # We want to move the whole segment by vector v.
        # v must be a unit vector.
        # The new start pos = chain[s] + v.
        # The new end pos = chain[e] + v.
        # We need:
        # 1. new_start adjacent to chain[s-1]
        # 2. new_end adjacent to chain[e+1] (if e < n-1)
        # 3. No collisions.
        
        # This is constrained. Let's just try 6 directions.
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        rng.shuffle(directions)
        
        for v in directions:
            new_start = (self.chain[s][0] + v[0], self.chain[s][1] + v[1], self.chain[s][2] + v[2])
            
            # Check adjacency to s-1
            if new_start not in _neighbours(self.chain[s-1]):
                continue
                
            # Check collision
            # We need to check if any new position in the segment is occupied.
            # And if the old positions are freed.
            
            # This is getting complex. Let's stick to Single Moves and Random Restarts.
            pass
        return False

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 280 # 20s safety margin
    
    best_chain = None
    best_energy = 0
    
    # Try a few different random seeds for SA
    for seed in range(100):
        if time.time() > deadline:
            break
            
        rng = random.Random(seed)
        try:
            folder = HPFolder(sequence, dimension)
        except Exception:
            continue
            
        current_energy = folder.energy
        if best_chain is None or current_energy < best_energy:
            best_chain = list(folder.chain)
            best_energy = current_energy
            
        temp = SA_START_TEMP
        steps = 0
        max_steps = 5000 # Adjust based on time
        
        while temp > SA_MIN_TEMP and steps < max_steps:
            if time.time() > deadline:
                break
                
            folder.step(rng, temp)
            temp *= SA_COOLING_RATE
            steps += 1
            
            if folder.energy < best_energy:
                best_energy = folder.energy
                best_chain = list(folder.chain)
                
        # If we have time, do a quick local search on the best found?
        # Or just restart SA.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
