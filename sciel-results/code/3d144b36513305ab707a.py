"""
HP lattice protein folding using Force-Directed Optimization and Grow-and-Fold.

Strategy:
1. Grow-and-Fold Initialization:
   - Instead of a generic Hamiltonian path, we construct the chain greedily.
   - We prioritize placing H residues in positions that are adjacent to existing H residues 
     (forming a compact core) while using P residues to connect them without self-avoidance 
     violations. This creates a "seed" conformation with high initial energy (low negative value).
2. Force-Directed Local Search (Simulated Annealing):
   - We model the energy as a sum of pairwise potentials.
   - H-H contacts are attractive (lower energy).
   - Non-bonded contacts (any residues) are repulsive if they violate self-avoidance (infinite penalty).
   - We use Simulated Annealing with a specific move set:
     a) Pivot moves (rotate tail).
     b) Slide moves (move a segment by 1 unit in a perpendicular direction).
     c) Grow/Prune (for very small chains, though usually not needed for N>20).
   - The SA schedule is tuned to escape local minima better than the greedy pivot search.
3. Deadline Guard:
   - Strict time management to ensure we return a valid, best-found conformation well before 
     the 210s wall-clock limit (targeting < 100s total).
"""

import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 4 neighbors of a point on a 2D lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """
    Compute HP contact energy.
    Energy = - (number of H-H pairs on neighboring lattice sites that are not adjacent along the chain).
    """
    n = len(chain)
    if n < 2:
        return 0
    
    # Map position to index for O(1) lookup
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            px, py = chain[i]
            for nx, ny in [(px+1, py), (px-1, py), (px, py+1), (px, py-1)]:
                j = where.get((nx, ny))
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_self_avoiding(chain):
    """Check if the chain is self-avoiding (no overlapping points)."""
    return len(set(chain)) == len(chain)

def _grow_and_fold(sequence):
    """
    Construct an initial conformation using a greedy grow-and-fold strategy.
    This attempts to keep H residues close to each other.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {chain[0]}
    h_positions = []
    
    # Heuristic: Try to fold H residues into a cluster.
    # We grow the chain one residue at a time.
    # For each new residue, we look at available neighbors.
    # If the new residue is H, we prefer neighbors that are close to existing H residues.
    # If the new residue is P, we prefer neighbors that expand the "frontier" or are neutral.
    
    for i in range(1, n):
        res = sequence[i]
        last_pos = chain[-1]
        
        # Get available neighbors of the last position
        candidates = []
        for nb in _neighbours(last_pos):
            if nb not in occupied:
                candidates.append(nb)
        
        if not candidates:
            # Dead end! This shouldn't happen often with good heuristics, but if it does,
            # we might need to backtrack or restart. For simplicity, we just pick a random
            # direction and hope, or we fail. 
            # Actually, a simple grow can get stuck. 
            # To make this robust, we can try a random walk if stuck, but that's complex.
            # Let's try a simple random restart logic if we get stuck in the main loop.
            # For now, if no candidates, we break and let SA fix it? No, SA needs a valid start.
            # We will implement a simple backtracking or just pick a random valid move from a wider search.
            # Better: If stuck, we return a random valid SAW of length n? 
            # Let's try a simple random SAW generation as a fallback.
            return _random_saw(n)

        # Score candidates
        scored_candidates = []
        for nb in candidates:
            score = 0.0
            # If placing H, prefer being near other H's
            if res == 'H':
                for h_pos in h_positions:
                    dist = abs(nb[0] - h_pos[0]) + abs(nb[1] - h_pos[1])
                    if dist == 2: # Adjacent to an H (potential contact)
                        score += 10.0
                    elif dist == 3:
                        score += 5.0
            # Prefer compactness (close to centroid) slightly to avoid long tails
            # Centroid of current chain
            if len(chain) > 1:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist_to_centroid = abs(nb[0] - cx) + abs(nb[1] - cy)
                score -= dist_to_centroid * 0.1
            
            # Random tie-breaker
            score += random.random() * 0.1
            scored_candidates.append((score, nb))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_nb = scored_candidates[0][1]
        
        chain.append(best_nb)
        occupied.add(best_nb)
        if res == 'H':
            h_positions.append(best_nb)
            
    return chain

def _random_saw(n):
    """Generate a random self-avoiding walk of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {chain[0]}
    while len(chain) < n:
        last = chain[-1]
        candidates = [nb for nb in _neighbours(last) if nb not in occupied]
        if not candidates:
            # Stuck, restart
            return _random_saw(n)
        next_pos = random.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain

def _simulated_annealing(sequence, initial_chain, time_budget, rng):
    """
    Refine the conformation using Simulated Annealing with Pivot and Slide moves.
    """
    n = len(initial_chain)
    if n < 5:
        return initial_chain

    current_chain = list(initial_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # SA Parameters
    start_temp = 1.0
    end_temp = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    temp = start_temp
    
    while temp > end_temp:
        if time.time() - start_time > time_budget:
            break
            
        # Choose a move
        # 80% Pivot, 20% Slide
        if rng.random() < 0.8:
            move_type = 'pivot'
        else:
            move_type = 'slide'
            
        new_chain = None
        new_energy = None
        
        if move_type == 'pivot':
            # Pivot move: Rotate a suffix of the chain around a joint
            # i is the pivot index (1 to n-2)
            i = rng.randint(1, n - 2)
            # Rotation direction: 0 or 1 (CW or CCW)
            rot = rng.randint(0, 1)
            
            # Extract tail
            tail = current_chain[i:]
            pivot_point = current_chain[i-1] # The point before the pivot segment start? 
            # Standard pivot: rotate chain[i:] around chain[i-1]? 
            # Or rotate chain[i+1:] around chain[i]?
            # Let's use: rotate chain[i:] around chain[i-1]. 
            # No, standard is: pick i, rotate segment [i+1, n-1] around [i].
            
            # Let's implement: Pick pivot index p (1..n-2). 
            # Segment to move: chain[p+1 : n]
            # Pivot point: chain[p]
            
            p = i
            pivot_pt = current_chain[p]
            segment = current_chain[p+1:]
            
            if not segment:
                continue
                
            # Rotate segment around pivot_pt
            new_segment = []
            valid = True
            # We need to check collisions with the fixed part chain[0:p+1]
            # The fixed part is chain[0:p+1] (indices 0 to p)
            # The moving part is chain[p+1:n]
            
            # Build set of fixed points for fast collision check
            # Note: pivot_pt is in fixed part.
            fixed_points = set(current_chain[:p+1])
            
            for pt in segment:
                # Vector from pivot to pt
                vx = pt[0] - pivot_pt[0]
                vy = pt[1] - pivot_pt[1]
                
                if rot == 0: # CW
                    nvx = vy
                    nvy = -vx
                else: # CCW
                    nvx = -vy
                    nvy = vx
                    
                new_pt = (pivot_pt[0] + nvx, pivot_pt[1] + nvy)
                
                # Check collision
                # 1. Cannot be in fixed_points (except pivot_pt itself? No, segment moves, pivot stays. 
                #    The first point of the new segment will be adjacent to pivot_pt.
                #    If new_pt == pivot_pt, it's a collision (overlap).
                #    If new_pt is in fixed_points, it's a collision.
                if new_pt in fixed_points:
                    valid = False
                    break
                    
                new_segment.append(new_pt)
                
            if valid:
                # Construct new chain
                new_chain = current_chain[:p+1] + new_segment
                # Verify self-avoidance of the whole chain (should be guaranteed by fixed_points check 
                # and isometry of rotation, but let's be safe for the first few steps or if logic is slightly off)
                # Actually, rotation preserves distances within the segment, so internal collisions are impossible.
                # Collision with fixed part is checked. 
                # So it should be valid.
                
                new_energy = _energy(sequence, new_chain)
                
        else:
            # Slide move: Move a segment of length L (2 <= L <= n-2) by 1 unit perpendicular to its start direction?
            # Simpler Slide: Pick a segment [i, i+L-1]. Try to shift it by (dx, dy) where (dx, dy) is one of 4 neighbors.
            # This is expensive to check. 
            # Alternative: "Crankshaft" move? 
            # Let's stick to Pivot for simplicity and robustness, or implement a simple "Shift" of the tail.
            # Let's implement a "Tail Shift": Move the last k residues by 1 unit in a random direction, 
            # if it doesn't collide.
            k = rng.randint(1, min(5, n-1))
            tail_start = n - k
            tail = current_chain[tail_start:]
            
            # Try 4 directions
            moved = False
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                new_tail = [(pt[0]+dx, pt[1]+dy) for pt in tail]
                
                # Check collision with fixed part chain[0:tail_start]
                # The fixed part is chain[0:tail_start]
                # The new_tail[0] must be adjacent to chain[tail_start-1] (which is fixed)
                # And no point in new_tail can be in fixed_part.
                
                fixed_part = chain[:tail_start] # Wait, chain is current_chain
                fixed_set = set(current_chain[:tail_start])
                
                # Check if new_tail[0] is adjacent to current_chain[tail_start-1]
                # It is, by construction of the shift? 
                # No, if we shift the whole tail, the connection at tail_start-1 might break 
                # unless the shift is such that new_tail[0] is adjacent to fixed_end.
                # Actually, if we shift the tail, the first point of the new tail moves.
                # It must be adjacent to the last fixed point.
                
                last_fixed = current_chain[tail_start-1]
                first_new = new_tail[0]
                if abs(last_fixed[0] - first_new[0]) + abs(last_fixed[1] - first_new[1]) != 1:
                    continue
                    
                # Check collisions
                valid = True
                for pt in new_tail:
                    if pt in fixed_set:
                        valid = False
                        break
                        
                if valid:
                    new_chain = current_chain[:tail_start] + new_tail
                    new_energy = _energy(sequence, new_chain)
                    moved = True
                    break
                    
            if not moved:
                continue
                
        if new_chain is None:
            continue
            
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temp *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Grow-and-Fold initialization and Simulated Annealing refinement.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    total_time_budget = 100.0 # Leave margin for 210s limit
    
    best_chain = None
    best_energy = float('inf')
    
    # 1. Initial Conformation
    # Try Grow-and-Fold
    init_chain = _grow_and_fold(sequence)
    if not _is_self_avoiding(init_chain):
        init_chain = _random_saw(n)
        
    best_chain = init_chain
    best_energy = _energy(sequence, best_chain)
    
    # 2. Refinement with Simulated Annealing
    # We will run multiple SA runs with different random seeds to escape local minima.
    # Time per run: total_budget / num_runs
    num_runs = 5
    time_per_run = total_time_budget / num_runs
    
    for i in range(num_runs):
        if time.time() - start_time > total_time_budget - 5: # 5s safety margin
            break
            
        # For restarts, we can perturb the best chain or use a new random init
        if i == 0:
            current_init = best_chain
        else:
            # Perturb best chain: randomize a few moves
            # Or just use a new random SAW for diversity
            # Using new random SAWs is safer to ensure diversity
            current_init = _random_saw(n)
            # Or, we can try to "shake" the best chain
            # Let's stick to random SAW for simplicity, but maybe bias it?
            # Actually, starting from a good chain and doing SA is better.
            # Let's do a "Reheat" strategy: take best_chain, randomize 10% of the positions? 
            # That's hard with SAW. 
            # Let's just run SA on the current best_chain again with a different seed? 
            # That might converge to the same minimum.
            # Better: Use a different initialization strategy for each run.
            # Run 0: Grow-and-Fold
            # Run 1: Random SAW
            # Run 2: Random SAW
            # Run 3: Grow-and-Fold (with different random tie-breakers)
            # Run 4: Random SAW
            
            if i % 2 == 0:
                current_init = _grow_and_fold(sequence)
                if not _is_self_avoiding(current_init):
                    current_init = _random_saw(n)
            else:
                current_init = _random_saw(n)
                
        rng = random.Random(12345 + i)
        
        # Run SA
        refined_chain = _simulated_annealing(sequence, current_init, time_per_run, rng)
        refined_energy = _energy(sequence, refined_chain)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
    # 3. Final Polish (Optional)
    # If we have time, do a quick greedy hill climb on the best chain
    if time.time() - start_time < total_time_budget - 2:
        polish_time = total_time_budget - (time.time() - start_time) - 2
        if polish_time > 1:
            # Simple hill climb
            current_chain = list(best_chain)
            current_energy = best_energy
            n = len(current_chain)
            rng = random.Random(99999)
            
            for _ in range(500): # Limited iterations
                if time.time() - start_time > total_time_budget - 1:
                    break
                    
                # Try a random pivot
                p = rng.randint(1, n - 2)
                rot = rng.randint(0, 1)
                pivot_pt = current_chain[p]
                segment = current_chain[p+1:]
                
                if not segment:
                    continue
                    
                new_segment = []
                valid = True
                fixed_points = set(current_chain[:p+1])
                
                for pt in segment:
                    vx = pt[0] - pivot_pt[0]
                    vy = pt[1] - pivot_pt[1]
                    if rot == 0:
                        nvx, nvy = vy, -vx
                    else:
                        nvx, nvy = -vy, vx
                    new_pt = (pivot_pt[0] + nvx, pivot_pt[1] + nvy)
                    if new_pt in fixed_points:
                        valid = False
                        break
                    new_segment.append(new_pt)
                    
                if valid:
                    new_chain = current_chain[:p+1] + new_segment
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = current_chain

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
