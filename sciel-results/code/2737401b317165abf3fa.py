"""Rewritten program using CSP-style backtracking with heuristic ordering and local search."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield all 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive residues on adjacent sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _generate_chain(sequence, dimension, rng, noise_scale=0.5):
    """
    Generate a chain using greedy backtracking with heuristic scoring and forward checking.
    Prioritizes moves that create H-H contacts and maintain open space.
    Uses constraint propagation to prune branches that lead to dead-ends.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Precompute H positions for quick check
    h_positions = [i for i, c in enumerate(sequence) if c == "H"]
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Calculate score
            score = 0.0
            
            # 1. Gain: H-H contacts formed
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        score += 10.0 # High weight for contacts
            
            # 2. Space: Free neighbors (to avoid trapping)
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            score += 1.0 * free_count
            
            # 3. Compactness: Prefer moves closer to center of mass of current chain
            # Simple proxy: distance from (0,0,0) or average position
            # Let's use distance from origin to keep it simple and compact
            dist = sum(c*c for c in site)
            score -= 0.1 * dist
            
            # 4. Noise for diversity
            score += rng.uniform(0, noise_scale)
            
            # 5. Forward checking: verify that the next residue has at least one valid position
            # This is a simple constraint propagation step
            if i < n - 1:
                next_valid = False
                for next_site in _neighbours(site):
                    if next_site not in occupied:
                        next_valid = True
                        break
                if not next_valid:
                    continue  # Skip this candidate as it leads to a dead-end
            
            candidates.append((score, site))
        
        if not candidates:
            return None
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pick the best one (or top few with probability)
        # For determinism in improvement, pick best. For diversity, we call this multiple times with different rngs.
        best_score, best_site = candidates[0]
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search_sa(sequence, dimension, chain, rng, time_limit=2.0):
    """
    Simulated Annealing local search with Kink and Segment Shift moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    start_time = time.time()
    
    # Precompute contact map for faster energy update?
    # For N<=136, O(N^2) energy calc is ~18k ops. 
    # 1000 iterations * 18k = 18M ops. This is fine in Python if optimized?
    # _energy is O(N * 6 * dict_lookup). ~ 6N.
    # So 1000 * 6 * 136 = 816k ops. Very fast.
    
    while time.time() - start_time < time_limit:
        # Cooldown
        T = max(T_end, T_start - (time.time() - start_time) / time_limit * (T_start - T_end))
        
        move_type = rng.random()
        
        valid_move = False
        new_chain = None
        new_energy = None
        
        if move_type < 0.5:
            # Kink move: move a single internal residue
            i = rng.randint(1, n - 2)
            # Get neighbors of current position
            for new_pos in _neighbours(current_chain[i]):
                if new_pos in {p for p in current_chain if p != current_chain[i]}:
                    continue
                # Check connectivity: new_pos must be adjacent to i-1 and i+1
                prev_pos = current_chain[i-1]
                next_pos = current_chain[i+1]
                
                dist_prev = sum(abs(new_pos[a] - prev_pos[a]) for a in range(dimension))
                dist_next = sum(abs(new_pos[a] - next_pos[a]) for a in range(dimension))
                
                if dist_prev == 1 and dist_next == 1:
                    # Check if new_pos is actually different from old
                    if new_pos == current_chain[i]:
                        continue
                    
                    # Apply move
                    new_chain = list(current_chain)
                    new_chain[i] = new_pos
                    valid_move = True
                    break
            
        else:
            # Segment shift move
            # Pick a segment [i, j]
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 2)
            
            # Try shifting by 1 in random direction
            direction = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
            
            # Check validity
            new_chain = list(current_chain)
            valid = True
            occupied_set = set()
            
            # Add non-moved parts
            for k in range(n):
                if i <= k <= j:
                    continue
                occupied_set.add(current_chain[k])
            
            # Check moved parts
            for k in range(i, j + 1):
                old_pos = current_chain[k]
                new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
                
                if new_pos in occupied_set:
                    valid = False
                    break
                
                # Check connectivity with previous element in segment or start
                if k == i:
                    prev_pos = current_chain[i-1]
                else:
                    prev_pos = new_chain[k-1] # This is wrong, new_chain not updated yet for k-1 if k-1 is in segment
                    
                # We need to check connectivity properly.
                # Let's construct the new positions for the segment first.
                new_positions = []
                for k in range(i, j + 1):
                    old_pos = current_chain[k]
                    new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
                    new_positions.append(new_pos)
                
                # Check self-avoidance
                for k in range(i, j + 1):
                    if new_positions[k-i] in occupied_set:
                        valid = False
                        break
                    # Also check against other new positions? No, shift is rigid, so relative distances preserved.
                    # But we need to check if any new pos collides with another new pos? No, shift is translation.
                    # However, we must ensure the segment doesn't collide with itself? No, translation preserves internal structure.
                    # But we must ensure the segment doesn't collide with the *rest* of the chain.
                    # occupied_set contains the rest.
                    
                if not valid:
                    break
                
                # Check connectivity at boundaries
                # Boundary 1: i-1 and i
                pos_prev = current_chain[i-1]
                pos_i_new = new_positions[0]
                dist1 = sum(abs(pos_prev[a] - pos_i_new[a]) for a in range(dimension))
                
                # Boundary 2: j and j+1
                pos_j_new = new_positions[-1]
                pos_next = current_chain[j+1]
                dist2 = sum(abs(pos_j_new[a] - pos_next[a]) for a in range(dimension))
                
                # Internal connectivity is preserved by translation.
                
                if dist1 != 1 or dist2 != 1:
                    valid = False
                
                if valid:
                    # Apply
                    for k in range(i, j + 1):
                        new_chain[k] = new_positions[k-i]
                    valid_move = True
                    break
            
        if not valid_move:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        # Acceptance
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            delta = new_energy - current_energy
            if T > 0 and rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find best conformation for the given HP sequence on the simple cubic lattice.
    """
    n = len(sequence)
    
    # Time budget per sequence
    # Total 240s for 26 sequences. ~9s per sequence.
    # Let's allocate 8s for search.
    start_time = time.time()
    time_limit = 8.0
    
    best_chain = None
    best_energy = 0 # Maximize energy (most negative is best, so we want min energy value, which is most negative)
    # Wait, energy is negative. "Score: mean ... energy / best_known_energy".
    # If best_known is -10, and we get -10, ratio is 1.
    # If we get -5, ratio is 0.5.
    # So we want the most negative energy.
    
    # Initial best
    # Try to generate a few chains and pick the best
    
    rng = random.Random(12345 + len(sequence))
    
    num_attempts = 20 if n > 30 else 50
    
    for i in range(num_attempts):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Generate chain
        # Vary noise scale to explore different regions
        noise = 0.5 + (i % 5) * 0.5
        chain = _generate_chain(sequence, dimension, rng, noise_scale=noise)
        
        if chain is None:
            continue
            
        e = _energy(sequence, chain)
        
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
        # Refine with Local Search
        # Allocate remaining time for this chain's refinement
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining > 0.5:
            # Spend a fraction of remaining time on SA
            sa_time = min(1.0, time_remaining * 0.5)
            refined_chain, refined_e = _local_search_sa(sequence, dimension, chain, rng, time_limit=sa_time)
            
            if refined_e < best_energy:
                best_energy = refined_e
                best_chain = refined_chain
                
    if best_chain is None:
        # Fallback: Straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
