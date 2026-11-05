import random
import math

def _get_neighbors(point):
    """Yield the 6 neighbors of a point in 3D."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _initial_chain(sequence, dimension):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Simple random growth
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    for i in range(1, n):
        placed = False
        while not placed:
            last = chain[-1]
            neighbors = list(_get_neighbors(last))
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied.add(nb)
                    placed = True
                    break
            if not placed:
                # Backtrack or restart
                # For simplicity in SA init, just restart if stuck
                # But for long chains, this might be slow. 
                # Let's use a slightly smarter init: greedy with randomness
                break
        if not placed:
             # Fallback: straight line then perturb
             # Actually, for SA, a valid start is crucial.
             # Let's try a simple spiral or just restart the whole function
             return _initial_chain(sequence, dimension)
             
    return chain

def _calculate_energy(sequence, chain):
    """Calculate energy of a chain."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for nb in _get_neighbors(p):
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j > i + 1: # Avoid double counting and adjacent
                        if sequence[j] == 'H':
                            energy -= 1
    return energy

def _try_pivot(sequence, chain, i, j, direction, occupied):
    """
    Try to pivot the segment [i, j] around the bond (i-1, i).
    direction: 0, 1, 2, 3 for 0, 90, 180, 270 deg? 
    In 3D, we have 3 axes. 
    Let's define moves more simply: 
    Flip: Reverse segment [i, j].
    Pivot: Rotate segment [i, j] by 90 deg around one of the 3 axes perpendicular to the bond (i-1, i).
    
    This is getting complex to implement efficiently in Python.
    Let's use a simpler move set that is still effective:
    1. Single step move (move one residue to a neighbor).
    2. Segment reversal (flip).
    
    Actually, "kink" moves and "pivot" moves are standard.
    Let's implement a generic "move segment" that tries to shift or rotate.
    
    Given the time constraints and complexity, let's use a robust 
    **Multi-start Hill Climbing with Random Restarts** using 
    **Local Moves** (swap positions of non-adjacent residues? No, must maintain connectivity).
    
    The most effective simple move in Python for this is often 
    **Random Local Search**: 
    Pick a random index i. Try to move residue i to a random empty neighbor.
    If valid, calculate delta E.
    
    This is slow to escape minima.
    
    Let's go back to **Simulated Annealing** but with a simpler move: 
    **Segment Reversal**.
    Pick i < j. Reverse chain[i:j+1].
    Check for collisions.
    Calculate Delta E.
    
    This is easy to implement and quite effective.
    """
    pass

# Let's implement a cleaner SA with Segment Reversal and Single Step moves.

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    best_chain = None
    best_energy = 0 # Energy is negative, so we want lower (more negative). 
                    # Wait, score is energy / best_known. 
                    # Best known is negative. 
                    # If we find energy -10 and best is -10, ratio is 1.
                    # If we find -8, ratio is 0.8.
                    # So we want to MINIMIZE the energy value (make it more negative).
    
    num_starts = 10
    max_iter = 2000
    T_start = 10.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iter)
    
    rng = random.Random(12345)
    
    for _ in range(num_starts):
        # Init
        chain = _initial_chain(sequence, dimension)
        if not chain:
            continue
            
        occupied = set(chain)
        energy = _calculate_energy(sequence, chain)
        
        curr_chain = chain
        curr_energy = energy
        curr_occupied = occupied
        
        if energy < best_energy:
            best_energy = energy
            best_chain = curr_chain[:]
            
        T = T_start
        for _ in range(max_iter):
            # Generate Move
            move_type = rng.random()
            if move_type < 0.5:
                # Segment Reversal
                i = rng.randint(0, n-2)
                j = rng.randint(i+1, n-1)
                # Try reversing i to j
                new_segment = curr_chain[i:j+1][::-1]
                
                # Check collisions
                # The rest of the chain is fixed.
                # We need to check if new_segment collides with 
                # curr_chain[0:i] and curr_chain[j+1:n]
                # Note: The bond connections (i-1, i) and (j, j+1) are 
                # handled by the fact that we are just checking positions.
                # But we must ensure the new segment doesn't overlap with itself 
                # (it doesn't, it's a reversal of a valid SAW segment? 
                # No, reversing a valid SAW segment is NOT necessarily self-avoiding 
                # with the rest, but IS self-avoiding internally? 
                # Yes, if the original was self-avoiding, the reversed set of points is the same, 
                # so no internal collisions.
                # We only need to check against the stationary parts.
                
                stationary = set(curr_chain[:i]) | set(curr_chain[j+1:])
                # Optimization: We only need to check if any point in new_segment 
                # is in stationary.
                
                valid = True
                for p in new_segment:
                    if p in stationary:
                        valid = False
                        break
                
                if valid:
                    # Calculate Delta Energy
                    # Only contacts involving the segment [i, j] and the rest change.
                    # Since the set of points in the segment is the same, 
                    # and the set of points in the rest is the same, 
                    # the *potential* contacts are the same pairs of points.
                    # However, the *identity* of the points changes? 
                    # No, the points are the same. The sequence indices change.
                    # Energy depends on sequence[i] and sequence[j] at positions P_i, P_j.
                    # If we reverse the segment, the residue at position P_k (which was 
                    # originally at index i+m) is now at index j-m.
                    # So we need to re-evaluate H-H contacts for all pairs (u, v) 
                    # where one is in [i, j] and the other is outside.
                    
                    # Let's just recalculate energy for the whole chain if valid?
                    # N=136, N^2 is ~18k. 2000 iter * 10 starts = 20k recalc.
                    # 20k * 18k = 360M ops. Might be too slow in Python.
                    
                    # We need incremental energy.
                    # Let's store the chain and calculate delta.
                    
                    # Old Energy contributions involving [i, j]:
                    # 1. Contacts within [i, j]: Unchanged (same points, but indices swapped).
                    #    Wait, if residue A was at idx 5 and B at idx 10, and they are H-H neighbors,
                    #    after reversal, A is at idx 10 and B at idx 5. They are still neighbors.
                    #    So internal energy is invariant under reversal.
                    # 2. Contacts between [i, j] and [0, i-1] or [j+1, n-1]:
                    #    The points are the same. The residues are permuted.
                    #    We need to check all pairs (p, q) where p in segment, q in stationary.
                    
                    # This is still O(L_seg * L_stat).
                    
                    # Let's try a faster move: **Single Step Move**.
                    # Pick a random index k. Try to move residue k to a neighbor.
                    # This is O(1) to check collision.
                    # Delta E is O(1) (only 6 neighbors of old pos and 6 of new pos).
                    
                    pass # Skip reversal for now, use single step.
            else:
                # Single Step Move
                k = rng.randint(0, n-1)
                old_pos = curr_chain[k]
                neighbors = list(_get_neighbors(old_pos))
                rng.shuffle(neighbors)
                
                moved = False
                for new_pos in neighbors:
                    if new_pos in curr_occupied:
                        continue
                    
                    # Check if new_pos is adjacent to old_pos? 
                    # Yes, by definition of neighbor.
                    # But we must ensure the new position doesn't collide with 
                    # any other part of the chain.
                    # Since we check `new_pos in curr_occupied`, and `old_pos` is 
                    # occupied by k, we must remove old_pos from consideration 
                    # or check against `curr_occupied - {old_pos}`.
                    
                    # If new_pos is not in curr_occupied, it's free.
                    # But wait, if k is moved, the bond (k-1, k) and (k, k+1) 
                    # must still be valid. 
                    # Moving k to a neighbor of k preserves the bond length (1).
                    # So connectivity is maintained.
                    
                    # Collision check:
                    # new_pos must not be in curr_occupied.
                    # (It is not, we checked).
                    
                    # So the move is valid geometrically.
                    
                    # Calculate Delta E
                    # Remove old_pos from occupied, add new_pos.
                    # Delta E = Energy(new) - Energy(old)
                    
                    # Only contacts involving k change.
                    # Old contacts: k with neighbors of old_pos.
                    # New contacts: k with neighbors of new_pos.
                    
                    # Let's calculate the change in H-H contacts involving index k.
                    # Note: Contacts between other pairs are unchanged.
                    
                    # We need to know which neighbors of old_pos were H and occupied by H.
                    # And which neighbors of new_pos are H and occupied by H.
                    
                    # Build a map of position -> index for quick lookup?
                    # We have curr_occupied set, but not index map.
                    # Let's maintain a pos_to_idx dict.
                    
                    # This is getting complex. 
                    
                    pass

    # Given the complexity of implementing efficient incremental energy in pure Python 
    # within the time limit, and the risk of bugs, let's use a simpler but robust 
    # **Genetic Algorithm** or **Multi-start Greedy** with **Local Refinement**.
    
    # Actually, the seed program scored 1.0. 
    # This implies the seed program found the ground state for all 26 chains.
    # If the seed program already scores 1.0, we cannot beat it with a score > 1.0 
    # unless we find a better energy than the "best known" (which is impossible by definition 
    # of "best known" in the context of the benchmark, usually meaning the proven ground state).
    # 
    # Wait, the task says: "Score (maximize): mean over the 26 benchmark sequences of 
    # energy / best_known_energy ... a value above 1.0 beats one".
    # This implies it is possible to beat the best known? 
    # Or does "best_known_energy" refer to a *published* value that might not be the 
    # true ground state? 
    # "the ten 48-residue sequences with proven ground states plus six longer chains...".
    # If the ground states are *proven*, then the best possible energy is the ground state energy.
    # So the maximum score is 1.0.
    # 
    # If the current program scores 1.0, it means it found the ground state for all sequences.
    # Therefore, we cannot improve the score. We can only try to maintain it.
    # 
    # However, the prompt asks to "Rewrite the program from scratch with a stronger algorithm."
    # And "forecast ... probability the child beats the island best".
    # If the island best is 1.0, and the max is 1.0, the probability of beating it is 0.
    # 
    # But maybe the "best known" for the longer chains (64-136) are *not* proven ground states?
    # "proven ground states" applies to the ten 48-residue ones.
    # The six longer chains might have "best known" that are heuristic lower bounds.
    # In that case, we *could* potentially beat them if we find a better fold.
    # 
    # So the goal is to find better folds for the longer chains.
    # 
    # The seed program uses a simple greedy growth. It likely failed to find optimal 
    # folds for the longer chains, or the "best known" for those chains is actually 
    # the ground state found by the seed program? 
    # If the seed program scores 1.0, it means `energy / best_known` is 1.0 for all.
    # This implies `energy == best_known` for all.
    # So the seed program found the "best known" for all.
    # 
    # If the "best known" for the longer chains are not proven ground states, 
    # and the seed program found them, then we are at the current best.
    # To beat it, we need to find a *better* fold than the current "best known".
    # 
    # So we need a stronger search algorithm to potentially find lower energies 
    # (more negative) for the longer chains.
    
    # Let's implement a proper Simulated Annealing.

    # Refined SA Plan:
    # 1. Init: Random SAW.
    # 2. Moves: 
    #    - Pivot (90 deg rotation of a segment).
    #    - Flip (reversal of a segment).
    # 3. Energy Delta: Incremental calculation.
    
    # To make incremental energy work, we maintain:
    # - chain: list of tuples
    # - pos_to_idx: dict of tuple -> index
    # - energy: current energy
    
    # Move: Flip [i, j]
    # 1. Extract segment.
    # 2. Reverse it.
    # 3. Check collision with stationary parts.
    # 4. If valid, calculate Delta E.
    #    - Remove old contacts involving [i, j].
    #    - Add new contacts involving [i, j].
    #    - Internal contacts of [i, j] are unchanged in terms of geometry, 
    #      but the *indices* change. 
    #      Wait, Energy is based on *positions* and *residue types*.
    #      If we reverse the segment, the residue at position P_k changes.
    #      So we MUST recalculate all contacts involving the segment.
    
    # This is O(L_seg * 6).
    
    # Let's implement this.

    best_chain = None
    best_energy = 0
    
    num_starts = 5
    max_iter = 1000
    T_start = 10.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iter)
    
    rng = random.Random(12345)
    
    for start in range(num_starts):
        chain = _initial_chain(sequence, dimension)
        if not chain:
            continue
            
        pos_to_idx = {p: i for i, p in enumerate(chain)}
        energy = _calculate_energy(sequence, chain)
        
        curr_chain = chain
        curr_pos_to_idx = pos_to_idx
        curr_energy = energy
        
        if energy < best_energy:
            best_energy = energy
            best_chain = curr_chain[:]
            
        T = T_start
        for _ in range(max_iter):
            # Choose move: Flip or Pivot
            # Flip is easier to implement correctly.
            i = rng.randint(0, n-2)
            j = rng.randint(i+1, n-1)
            
            # Extract segment
            seg = curr_chain[i:j+1]
            new_seg = seg[::-1]
            
            # Check collision
            # Stationary parts: [0, i-1] and [j+1, n-1]
            # We need to check if any point in new_seg is in the stationary set.
            # Optimization: 
            # The points in new_seg are the same as in seg.
            # So if seg was valid, new_seg is valid *internally*.
            # We only need to check against stationary.
            
            # Build set of stationary points? 
            # That's O(N). 
            # Better: Check each point in new_seg against curr_pos_to_idx.
            # If a point p in new_seg is in curr_pos_to_idx, 
            # and its index is NOT in [i, j], then it's a collision.
            
            valid = True
            for p in new_seg:
                idx = curr_pos_to_idx.get(p)
                if idx is not None and not (i <= idx <= j):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Delta E
            # Old Energy contributions from [i, j]:
            # We need to subtract contacts where one end is in [i, j] and the other is outside.
            # And add new contacts.
            
            # Since the *points* are the same, the *potential* neighbor pairs are the same.
            # But the *residues* at those points are different.
            
            # Let's just recalculate the energy for the whole chain if the move is valid.
            # N=136. 
            # 1000 iter * 5 starts = 5000 recalc.
            # Recalc is O(N * 6) ~ 800 ops.
            # 5000 * 800 = 4M ops.
            # This is fast in Python.
            
            new_chain = curr_chain[:i] + new_seg + curr_chain[j+1:]
            new_energy = _calculate_energy(sequence, new_chain)
            
            delta_e = new_energy - curr_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                curr_chain = new_chain
                curr_pos_to_idx = {p: i for i, p in enumerate(curr_chain)}
                curr_energy = new_energy
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_chain = curr_chain[:]
                    
            T *= alpha
            
    if best_chain is None:
        # Fallback
        best_chain = _initial_chain(sequence, dimension)
        
    return [list(p) for p in best_chain]

# The above code is a draft. Let's refine it into the final block.
# One issue: _initial_chain might be slow or fail for long chains if it gets stuck.
# We'll add a timeout or retry mechanism.

# Also, for very short chains, SA is overkill.
# We'll handle N < 10 with simple brute force or greedy.

# Final check on energy calculation:
# Energy = - (number of H-H contacts).
# My _calculate_energy returns negative number.
# Best energy is most negative.
# So `if energy < best_energy` is correct (initializing best_energy to 0 or a large positive number? 
# No, energy is negative. So 0 is "unfolded". We want to go lower.
# Initialize best_energy to 0? 
# If we find -5, -5 < 0 is True. Good.
# If we find -10, -10 < -5 is True. Good.

# Let's ensure the fallback is a valid chain.

