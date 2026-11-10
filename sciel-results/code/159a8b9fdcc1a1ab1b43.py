"""Rewritten program: HP lattice protein folding using simulated annealing with chain moves."""
# EVOLVE-BLOCK-START
import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow_straight(sequence, dimension):
    """Generate a straight line conformation."""
    return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]

def _local_search(sequence, dimension, chain, rng, time_budget=5.0):
    """Local search using pivot moves and local rearrangements."""
    chain = list(chain)
    n = len(chain)
    if n < 4:
        return chain
    
    occupied = set(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    while time.time() - start_time < time_budget:
        # Try a pivot move: pick a pivot point and rotate a segment
        pivot_idx = rng.randint(1, n - 2)
        segment_len = rng.randint(1, min(10, n - pivot_idx - 1))
        
        # Extract segment
        segment = chain[pivot_idx:pivot_idx + segment_len + 1]
        
        # Try rotating the segment in different directions
        # For simplicity, try small local moves instead
        # Try moving a single residue (if possible)
        if rng.random() < 0.5:
            # Single step move
            idx = rng.randint(1, n - 2)
            current_pos = chain[idx]
            moved = False
            for new_pos in _neighbours(current_pos):
                if new_pos in occupied:
                    continue
                # Check if this move keeps the chain connected
                # The new position must be adjacent to chain[idx-1] or chain[idx+1]
                prev_pos = chain[idx - 1]
                next_pos = chain[idx + 1] if idx + 1 < n else None
                
                # Check distance to prev
                dist_prev = sum(abs(a - b) for a, b in zip(new_pos, prev_pos))
                if dist_prev != 1:
                    continue
                    
                # If there's a next, check distance to next
                if next_pos is not None:
                    dist_next = sum(abs(a - b) for a, b in zip(new_pos, next_pos))
                    if dist_next != 1:
                        continue
                
                # Perform the move
                occupied.remove(current_pos)
                chain[idx] = new_pos
                occupied.add(new_pos)
                
                new_energy = _energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                else:
                    # Revert
                    occupied.remove(new_pos)
                    chain[idx] = current_pos
                    occupied.add(current_pos)
                moved = True
                break
        else:
            # Try a 180-degree flip of a short segment
            seg_len = rng.randint(2, min(4, n - 1))
            start = rng.randint(1, n - seg_len - 1)
            
            # Extract segment
            segment = chain[start:start + seg_len]
            
            # Try flipping the segment
            flipped = []
            for i, pos in enumerate(segment):
                # Flip relative to the first point
                base = segment[0]
                new_pos = tuple(2 * base[k] - pos[k] for k in range(dimension))
                flipped.append(new_pos)
            
            # Check if flipped segment is valid
            valid = True
            temp_occupied = set(occupied)
            for pos in segment:
                temp_occupied.remove(pos)
            
            for i, pos in enumerate(flipped):
                if pos in temp_occupied:
                    valid = False
                    break
                temp_occupied.add(pos)
            
            if not valid:
                continue
            
            # Check connectivity at boundaries
            if start > 0:
                prev_pos = chain[start - 1]
                first_flipped = flipped[0]
                dist = sum(abs(a - b) for a, b in zip(prev_pos, first_flipped))
                if dist != 1:
                    continue
            
            end_idx = start + seg_len
            if end_idx < n:
                next_pos = chain[end_idx]
                last_flipped = flipped[-1]
                dist = sum(abs(a - b) for a, b in zip(last_flipped, next_pos))
                if dist != 1:
                    continue
            
            # Perform the flip
            for i, pos in enumerate(segment):
                occupied.remove(pos)
            for pos in flipped:
                occupied.add(pos)
            chain[start:start + seg_len] = flipped
            
            new_energy = _energy(sequence, chain)
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                for i, pos in enumerate(flipped):
                    occupied.remove(pos)
                for pos in segment:
                    occupied.add(pos)
                chain[start:start + seg_len] = segment
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multiple restarts with local search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Start with a straight line
    best_chain = _grow_straight(sequence, dimension)
    best_energy = _energy(sequence, best_chain)
    
    # Multiple restarts with local search
    num_restarts = 50 if n <= 50 else 20
    
    for restart in range(num_restarts):
        # Start from a slightly perturbed straight line
        chain = _grow_straight(sequence, dimension)
        
        # Perturb some random positions
        for _ in range(min(5, n // 10)):
            idx = rng.randint(1, n - 2)
            current_pos = chain[idx]
            for new_pos in _neighbours(current_pos):
                if new_pos not in set(chain):
                    # Check connectivity
                    prev_pos = chain[idx - 1]
                    dist_prev = sum(abs(a - b) for a, b in zip(new_pos, prev_pos))
                    if dist_prev == 1:
                        if idx + 1 < n:
                            next_pos = chain[idx + 1]
                            dist_next = sum(abs(a - b) for a, b in zip(new_pos, next_pos))
                            if dist_next == 1:
                                chain[idx] = new_pos
                                break
                        else:
                            chain[idx] = new_pos
                        break
        
        # Local search
        improved_chain = _local_search(sequence, dimension, chain, rng, time_budget=0.5)
        energy = _energy(sequence, improved_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = improved_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
