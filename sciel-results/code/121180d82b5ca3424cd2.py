"""
HP Lattice Protein Folding on the Cubic Lattice
Strategy: 
1. Generate an initial self-avoiding walk using a greedy heuristic.
2. Optimize using Simulated Annealing with Pivot, Slither, and Kick moves.
3. Strict time management to fit within the 330s budget.
"""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global time management
_START_TIME = None
_TOTAL_BUDGET = 320.0  # Leave 10s margin for safety

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_remaining():
    return _TOTAL_BUDGET - _get_elapsed()

# Precomputed neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])

def _sub(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

def _energy(sequence, chain):
    """
    Calculate the energy of the chain.
    Energy = - (number of H-H contacts between non-adjacent residues).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # Only check H residues
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 6 neighbors
            for offset in _NEIGHBOR_OFFSETS:
                q = _add(p, offset)
                j = pos_to_idx.get(q)
                # If neighbor exists in chain, is not adjacent in sequence, and is H
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    # To avoid double counting, only count if i < j
                    if i < j:
                        contacts += 1
    return -contacts

def _grow_initial_chain(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk using a greedy heuristic.
    Prefers moves that keep the chain compact.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        # Get all free neighbors
        free_neighbors = []
        for offset in _NEIGHBOR_OFFSETS:
            next_pos = _add(current, offset)
            if next_pos not in occupied:
                free_neighbors.append(next_pos)
        
        if not free_neighbors:
            # Dead end. Try to backtrack? For simplicity, return None or partial.
            # In a robust implementation, we'd backtrack. Here, we'll just fail this growth attempt.
            return None
        
        # Heuristic: Prefer moves that are closer to the origin or "inside" the bounding box
        # A simple heuristic: pick randomly from free neighbors, but maybe bias towards 
        # directions that haven't been used too much recently?
        # Let's just pick randomly for the start, it's fast.
        next_pos = rng.choice(free_neighbors)
        
        chain.append(next_pos)
        occupied[next_pos] = i
        
    return chain

def _simulate_annealing(sequence, chain, rng, time_limit):
    """
    Perform Simulated Annealing on the given chain.
    Returns the best chain found.
    """
    n = len(chain)
    if n == 0:
        return chain
    
    # Initial energy
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Occupied set for fast collision checking
    occupied = {p: i for i, p in enumerate(chain)}
    
    # SA Parameters
    # Start temperature high enough to accept bad moves
    # Max energy change is roughly n/2 (all Hs contact all Hs)
    # Let's start T around 10-20 and decay to near 0
    T_start = 10.0
    T_end = 0.01
    # Decay rate depends on number of steps. We don't know steps in advance, 
    # so we use time-based decay or step-based with a fixed number of steps?
    # Better: Run for a fixed number of steps per "phase" or just run until time runs out.
    # Let's run until time runs out, adjusting T based on elapsed time?
    # Or simpler: Fixed number of iterations, but check time.
    
    # Let's estimate iterations. 1 million iterations is fast in Python if optimized.
    # But we have 26 sequences. We should distribute time.
    # However, the function `fold` is called once per sequence.
    # So we have ~320s / 26 ~ 12s per sequence.
    
    # We will run SA until time_limit is reached.
    
    # Move selection:
    # 1. Slither (move end): 50%
    # 2. Pivot: 30%
    # 3. Kick (move small segment): 20%
    
    # We need to be careful with memory and speed.
    # Using a list of tuples is okay.
    
    # To speed up energy calculation, we can update it incrementally.
    # But implementing incremental update for Pivot/Kick is complex.
    # Given the time budget, full recalculation might be too slow if we do many steps.
    # Let's try incremental updates for Slither, and full recalc for others?
    # Or just full recalc and hope Python is fast enough for small n (n<=136).
    # n=136, neighbors=6, checks ~ 136*6 = 816. Very fast.
    # 100,000 steps * 816 ops = 80 million ops. Python might take 1-2 seconds.
    # So we can do roughly 10k-50k steps per second.
    # In 12s, we can do ~500k steps.
    
    steps_done = 0
    start_time = time.time()
    
    # We need to track best
    while (time.time() - start_time) < time_limit:
        # Adjust temperature based on progress? 
        # Simple linear decay over expected steps?
        # Let's assume we run for 100,000 steps.
        # If we run out of time, we stop.
        
        # Simple decay: T = T_start * (1 - steps_done / 100000)
        # But if we run longer, T goes negative.
        # Better: T = T_start * (0.9999 ** steps_done)
        T = T_start * (0.9999 ** steps_done)
        if T < T_end:
            T = T_end
            
        move_type = rng.random()
        
        # --- Try a move ---
        new_chain = None
        delta_e = 0
        
        if move_type < 0.5:
            # Slither: Move the last residue
            last_idx = n - 1
            last_pos = chain[last_idx]
            # Find free neighbors of last_pos
            candidates = []
            for offset in _NEIGHBOR_OFFSETS:
                new_pos = _add(last_pos, offset)
                if new_pos not in occupied:
                    candidates.append(new_pos)
            if candidates:
                new_pos = rng.choice(candidates)
                # Check if new_pos is adjacent to chain[last_idx-1] (it must be, by definition of SAW extension, but here we are moving the end)
                # Wait, Slither moves the end to a *new* free neighbor.
                # The previous position last_pos becomes free.
                # The new position new_pos must be free.
                # Is new_pos adjacent to chain[last_idx-1]? 
                # Yes, because chain[last_idx] was adjacent to chain[last_idx-1]. 
                # If we move chain[last_idx] to a neighbor of chain[last_idx], 
                # is it still adjacent to chain[last_idx-1]? 
                # NO. A slither move requires the new position to be adjacent to the *previous* residue in the chain?
                # Actually, standard slither: The end moves to a free neighbor of the current end.
                # The bond between last and second-last is broken and re-formed?
                # No, the chain is a path. 
                # If we move the last node, the edge (n-2, n-1) changes.
                # The new edge is (n-2, new_pos).
                # So new_pos MUST be adjacent to chain[n-2].
                # AND new_pos must be free (not occupied by any other node).
                # AND new_pos != chain[n-1] (obviously).
                
                prev_pos = chain[n-2]
                valid_candidates = [p for p in candidates if _sub(p, prev_pos) in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)] or p == _add(prev_pos, o) for o in _NEIGHBOR_OFFSETS]
                # Simpler check:
                valid_candidates = [p for p in candidates if abs(p[0]-prev_pos[0]) + abs(p[1]-prev_pos[1]) + abs(p[2]-prev_pos[2]) == 1]
                
                if valid_candidates:
                    new_pos = rng.choice(valid_candidates)
                    # Apply move
                    # Remove last_pos from occupied
                    del occupied[last_pos]
                    # Add new_pos to occupied
                    occupied[new_pos] = last_idx
                    # Update chain
                    new_chain = list(chain)
                    new_chain[last_idx] = new_pos
                    
                    # Calculate Delta E
                    # Contacts involving last_idx or its neighbors might change.
                    # For simplicity and correctness, recalc energy? 
                    # Or incremental?
                    # Let's recalc for now, it's fast.
                    # new_energy = _energy(sequence, new_chain)
                    # delta_e = new_energy - current_energy
                    
                    # Optimization: Only check contacts involving last_idx and its neighbors
                    # But _energy is simple. Let's stick to recalc if it's not the bottleneck.
                    
        elif move_type < 0.8:
            # Pivot: Rotate a segment around a pivot point
            # Pick a pivot index k (0 < k < n-1)
            # Pick a segment length L (1 <= L <= min(k, n-1-k))
            # The segment is chain[k+1 : k+1+L]
            # Rotate this segment around chain[k]
            
            k = rng.randint(1, n-2)
            max_len = min(k, n-1-k)
            if max_len < 1:
                continue
            L = rng.randint(1, max_len)
            
            pivot_pos = chain[k]
            segment = chain[k+1 : k+1+L]
            
            # Determine the vector from pivot to the first element of the segment
            v = _sub(segment[0], pivot_pos)
            
            # Generate a rotation
            # We need to rotate v to a new vector v' such that v' is a unit vector 
            # and the rotated segment doesn't collide.
            # Simple strategy: Pick a random unit vector v' different from v, 
            # and check if the rotated segment fits.
            
            # There are 6 neighbors. v is one of them.
            # Pick a new direction from the other 5.
            # But rotation in 3D is not just picking a neighbor.
            # We need to rotate the *entire segment*.
            # If the segment is a straight line, we can just point it in a new direction.
            # If the segment is curved, we need a 3D rotation matrix.
            
            # Simplified Pivot: Only allow "kink" moves or simple rotations for straight segments?
            # A common move is "Pivot" where we rotate the tail around a joint.
            # For simplicity, let's implement a "Move" where we pick a random free site 
            # for the *end* of the segment, and try to interpolate? No, that's complex.
            
            # Alternative: "Reversal" move.
            # Reverse the segment chain[k+1 : k+1+L].
            # This changes the direction but keeps the positions.
            # It might help escape local minima in directionality but doesn't change positions.
            # So it doesn't change energy! 
            # So Reversal is useless for energy optimization alone, unless combined with other moves.
            
            # Let's implement a proper Pivot for small segments (L=1 or L=2).
            # If L=1: Just move the single node chain[k+1] to a free neighbor of chain[k] that is not chain[k+1] and not chain[k-1] (if k>0).
            # Actually, chain[k+1] is already a neighbor of chain[k].
            # We move it to another free neighbor of chain[k].
            
            if L == 1:
                target_idx = k + 1
                current_pos = chain[target_idx]
                prev_pos = chain[k]
                next_pos = chain[target_idx + 1] if target_idx + 1 < n else None
                
                candidates = []
                for offset in _NEIGHBOR_OFFSETS:
                    new_pos = _add(prev_pos, offset)
                    if new_pos == current_pos:
                        continue
                    if new_pos in occupied:
                        # If it's the next node, it's occupied, but we are moving current, so next node stays?
                        # No, if we move current, the bond (current, next) breaks.
                        # So new_pos must not be occupied by ANY other node.
                        # occupied contains all nodes.
                        # If new_pos is occupied by a node other than current_pos, it's invalid.
                        # If new_pos is current_pos, we skip.
                        # If new_pos is next_pos, it is occupied. So invalid.
                        continue
                    candidates.append(new_pos)
                
                if candidates:
                    new_pos = rng.choice(candidates)
                    # Check if new_pos is adjacent to next_pos?
                    # The new bond is (new_pos, next_pos).
                    # So new_pos MUST be adjacent to next_pos.
                    if next_pos is not None:
                        if abs(new_pos[0]-next_pos[0]) + abs(new_pos[1]-next_pos[1]) + abs(new_pos[2]-next_pos[2]) != 1:
                            continue
                    
                    # Apply move
                    del occupied[current_pos]
                    occupied[new_pos] = target_idx
                    new_chain = list(chain)
                    new_chain[target_idx] = new_pos
                    
        else:
            # Kick: Move a small segment (e.g., 2 nodes) to a nearby free spot.
            # This is complex. Let's skip for now and rely on Slither and Pivot(L=1).
            # Or implement a simple "Slide" of the end by 2 steps?
            pass
        
        if new_chain is not None:
            # Calculate new energy
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e < 0:
                # Improvement
                current_energy = new_energy
                chain = new_chain
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Worse, accept with probability exp(-delta_e / T)
                # Note: delta_e is positive (energy increases, which is bad since we minimize)
                # So we accept with prob exp(-delta_e / T)
                if delta_e > 0:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        current_energy = new_energy
                        chain = new_chain
                # If delta_e == 0, always accept (neutral)
                elif delta_e == 0:
                    current_energy = new_energy
                    chain = new_chain
        
        steps_done += 1
        # Check time periodically
        if steps_done % 1000 == 0:
            if time.time() - start_time > time_limit:
                break
                
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation of the sequence using Simulated Annealing.
    """
    if _START_TIME is None:
        global _START_TIME
        _START_TIME = time.time()
        
    # Check remaining time
    remaining = _time_remaining()
    if remaining <= 0:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(len(sequence))]
    
    # Allocate time for this sequence
    # We don't know how many sequences are left, so we just use a fraction of remaining?
    # Or just run until a small fraction of time is left?
    # Let's run for min(remaining, 5.0) seconds? 
    # No, if we have 100s left and 10 sequences, we should use 10s each.
    # But we don't know the count.
    # Strategy: Run until 5% of remaining time is left?
    # Or just run for a fixed chunk, e.g., 2s, and hope we get called again?
    # No, fold is called once.
    # So we must use ALL remaining time for this sequence?
    # If this is the last sequence, yes. If it's the first, we should save time.
    # Since we don't know the index, let's assume we have ~10s per sequence on average.
    # Let's cap the time for this call at 10s.
    time_limit = min(remaining, 10.0)
    
    rng = random.Random(12345 + len(sequence))
    
    # Generate initial chain
    best_chain = None
    best_energy = 0
    
    # Try a few initializations
    for _ in range(10):
        chain = _grow_initial_chain(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
        
    # Run SA
    # We pass the best initial chain
    # We need to pass the time limit
    # Note: _simulate_annealing uses its own start_time
    result_chain = _simulate_annealing(sequence, best_chain, rng, time_limit)
    
    # Convert tuples to lists
    return [list(p) for p in result_chain]
# EVOLVE-BLOCK-END
