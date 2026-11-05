"""
HP lattice protein folding using Pivot Local Search.
The pivot move rotates a segment of the chain around a pivot point, 
allowing for larger conformational changes than single-residue moves.
"""
import time
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """
    Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts)
    Contacts are between non-consecutive residues on adjacent lattice sites.
    """
    n = len(chain)
    if n < 2:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Identify H positions
    h_positions = set()
    for i, res in enumerate(sequence):
        if res == 'H':
            h_positions.add(i)
            
    if len(h_positions) < 2:
        return 0

    contacts = 0
    # Iterate over H residues
    for i in h_positions:
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None:
                # Check if j is H and not adjacent in sequence
                if j in h_positions and abs(i - j) > 1:
                    # Count each contact only once (i < j)
                    if i < j:
                        contacts += 1
                        
    return -contacts

def _greedy_growth(sequence, rng, compactness_bias=0.5, contact_bias=1.0):
    """
    Grow the chain using a greedy strategy that favors compactness and H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last_pos = chain[-1]
        candidates = []
        
        for next_pos in _neighbours(last_pos):
            if next_pos in occupied:
                continue
            
            score = 0.0
            
            # Compactness: prefer positions closer to the centroid
            if len(chain) > 2:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(next_pos[0] - cx) + abs(next_pos[1] - cy)
                score -= compactness_bias * dist
            
            # Contact potential: if current is H, prefer positions adjacent to existing H's
            if sequence[i] == 'H':
                for q in _neighbours(next_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += contact_bias * 10.0 # High weight for direct contact
            
            # Avoid dead ends
            free_neighbors = sum(1 for q in _neighbours(next_pos) if q not in occupied)
            if free_neighbors == 0 and i < n - 1:
                score -= 100.0
            elif free_neighbors == 1 and i < n - 2:
                score -= 10.0
                
            score += rng.random() * 0.5
            candidates.append((score, next_pos))
            
        if not candidates:
            return None # Failed to grow
            
        candidates.sort(key=lambda x: -x[0])
        
        # Sometimes explore a non-best candidate to add diversity
        if rng.random() < 0.1 and len(candidates) > 1:
            idx = min(rng.randint(1, 3), len(candidates) - 1)
            best_pos = candidates[idx][1]
        else:
            best_pos = candidates[0][1]
            
        chain.append(best_pos)
        occupied[best_pos] = i
        
    return chain

def _pivot_search(sequence, chain, rng, max_iters=2000, time_limit=None):
    """
    Local search using pivot moves.
    A pivot move rotates the segment [i+1, n-1] 90 degrees around chain[i].
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute H indices for faster contact check updates if needed, 
    # but for now we recompute energy for simplicity in pivot validation 
    # unless we implement delta energy. 
    # Delta energy for pivot is complex. Recomputing O(N) energy is O(N^2) per iter.
    # For N=100, N^2=10000. 2000 iters = 20M ops. Might be slow in Python.
    # Optimization: Use a set of occupied positions for collision check.
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    if not h_indices:
        return current_chain, current_energy
        
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        if no_improve_count > max_no_improve:
            break
            
        # Choose a pivot point i (1 to n-2)
        i = rng.randint(1, n - 2)
        pivot_pos = current_chain[i]
        
        # Determine the segment to rotate: [i+1, n-1]
        # We need to check if rotating this segment results in a valid self-avoiding walk.
        
        # The direction of the bond from i to i+1 determines the rotation.
        # Let v = current_chain[i+1] - current_chain[i]
        # We rotate v by 90 deg left or right.
        # Then we propagate this rotation for all subsequent residues.
        
        # To do this efficiently, we can't just rotate vectors because the chain might fold back.
        # We must simulate the walk from i+1 onwards with the new initial direction.
        
        # However, a true pivot move rotates the *entire spatial configuration* of the tail 
        # around the pivot point. This means relative positions are preserved, 
        # but absolute positions change.
        
        # Let's implement the standard pivot move:
        # 1. Take the tail T = current_chain[i+1:]
        # 2. Translate T so that T[0] is at the origin.
        # 3. Rotate T by 90 deg (left or right).
        # 4. Translate T back so that T[0] is at the new position adjacent to pivot.
        # 5. Check for collisions with the prefix [0, i] and within T.
        
        # This is O(N) per move if we use sets, but we need to rebuild the tail.
        
        # Optimization: Only try pivots that are likely to improve energy?
        # Hard to know. Just try random pivots.
        
        # To speed up, we can pre-compute the relative coordinates of the tail.
        # But the tail changes shape in the chain representation? No, the *sequence* is fixed.
        # The *conformation* changes.
        
        # Let's try a simpler move first: "Segment Move" or just stick to Pivot.
        # Pivot is the standard.
        
        # Implementation of Pivot Move Check:
        # We need to check if the rotated tail collides with the prefix.
        
        # Get tail relative to pivot
        tail = current_chain[i+1:]
        if not tail:
            continue
            
        # Translate tail so tail[0] is (0,0)
        tx, ty = tail[0]
        rel_tail = [(x - tx, y - ty) for x, y in tail]
        
        # Try left rotation
        # Left rotation of (x, y) is (-y, x)
        rot_left = [(-y, x) for x, y in rel_tail]
        
        # The new position of tail[0] must be adjacent to pivot.
        # The vector from pivot to old tail[0] was v = (tx - pivot_x, ty - pivot_y)
        # The new vector is rot_left[0]
        # We must ensure rot_left[0] is a valid unit step from pivot.
        # Since rot_left[0] is a rotation of v, it is a unit vector.
        # We just need to check which direction it points and if that spot is free.
        
        # Actually, the standard pivot move rotates the *tail* such that the bond (i, i+1) 
        # changes direction. 
        # Let's just generate the new absolute positions.
        
        # Direction of old bond i -> i+1
        dx = tx - pivot_pos[0]
        dy = ty - pivot_pos[1]
        
        # Possible new directions for bond i -> i+1:
        # Left turn: (-dy, dx)
        # Right turn: (dy, -dx)
        # (Straight is not a pivot move, it's no-op)
        
        new_dirs = [(-dy, dx), (dy, -dx)]
        
        for ndx, ndy in new_dirs:
            new_tail_start = (pivot_pos[0] + ndx, pivot_pos[1] + ndy)
            
            # Check if new_tail_start is occupied by prefix [0, i]
            if new_tail_start in occupied:
                # It might be occupied by the tail itself? No, we are moving the tail.
                # It must not be in the prefix.
                # If it's in the prefix, collision.
                # Note: occupied contains all current positions.
                # We need to check if it's in prefix [0, i].
                idx_at_new_start = occupied.get(new_tail_start)
                if idx_at_new_start is not None and idx_at_new_start <= i:
                    continue # Collision with prefix
                if idx_at_new_start is None:
                    # It's free? But wait, occupied is the CURRENT state.
                    # The tail is currently occupying some spots.
                    # We are moving the tail.
                    # So we must check collision with prefix [0, i] ONLY.
                    # And collision within the new tail.
                    pass
            
            # Let's be precise.
            # Prefix is fixed: P = current_chain[0:i+1]
            # New Tail T' starts at new_tail_start.
            # We need to generate T' by applying the same relative geometry as T, 
            # but rotated?
            # NO. A pivot move rotates the *entire rigid body* of the tail around the pivot.
            # This means the *shape* of the tail is preserved, just rotated and translated.
            
            # So, take rel_tail, rotate it, and add new_tail_start.
            
            if ndx == -dy and ndy == dx: # Left
                new_tail = [(new_tail_start[0] + rx, new_tail_start[1] + ry) for rx, ry in rot_left]
            else: # Right
                # Right rotation of (x, y) is (y, -x)
                rot_right = [(y, -x) for x, y in rel_tail]
                new_tail = [(new_tail_start[0] + rx, new_tail_start[1] + ry) for rx, ry in rot_right]
            
            # Check for self-intersection in new_tail
            if len(set(new_tail)) != len(new_tail):
                continue
                
            # Check collision with prefix [0, i]
            # Prefix positions are current_chain[0:i+1]
            # We can use the occupied dict but must exclude indices > i
            collision = False
            for p in new_tail:
                idx = occupied.get(p)
                if idx is not None and idx <= i:
                    collision = True
                    break
                    
            if collision:
                continue
                
            # Valid move!
            # Compute new energy
            new_chain = current_chain[:i+1] + new_tail
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                current_chain = new_chain
                current_energy = new_energy
                occupied = {p: k for k, p in enumerate(current_chain)}
                no_improve_count = 0
            else:
                no_improve_count += 1
            break # Only one successful pivot per iteration? Or try both?
            # Trying both is fine, but usually one is enough to update state.
            # If one fails, try the other.
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy self-avoiding conformation.
    """
    start_time = time.time()
    deadline = start_time + 195.0 # Safety margin
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple Greedy Starts
    # Generate a few diverse initial chains
    initial_chains = []
    for i in range(5):
        # Vary biases
        c_bias = 0.3 + rng.random() * 0.4
        h_bias = 0.5 + rng.random() * 1.0
        chain = _greedy_growth(sequence, rng, compactness_bias=c_bias, contact_bias=h_bias)
        if chain and len(chain) == n:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback: simple line
        initial_chains = [[(i, 0) for i in range(n)]]
        
    # Evaluate initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
            
    # Strategy 2: Pivot Local Search from Best Initial
    # Allocate time based on sequence length
    # Short sequences can afford more iterations
    if n <= 20:
        max_pivot_iters = 5000
    elif n <= 50:
        max_pivot_iters = 2000
    else:
        max_pivot_iters = 1000
        
    time_limit = deadline - 5.0
    
    if best_chain:
        improved_chain, improved_energy = _pivot_search(
            sequence, best_chain, rng, max_iters=max_pivot_iters, time_limit=time_limit
        )
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    # Strategy 3: Try Pivot Search from other initial chains if time permits
    if time.time() < deadline - 10.0:
        for chain in initial_chains[1:]:
            if time.time() > deadline - 5.0:
                break
            # Use less time for secondary searches
            sub_limit = min(deadline - 5.0, time.time() + (deadline - 5.0 - time.time()) / len(initial_chains))
            if sub_limit <= time.time():
                continue
            c_chain, c_energy = _pivot_search(
                sequence, chain, rng, max_iters=max_pivot_iters//2, time_limit=sub_limit
            )
            if c_energy < best_energy:
                best_energy = c_energy
                best_chain = c_chain
                
    # Final Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# Quick sanity check / self-test if run directly
if __name__ == "__main__":
    # Test with a small sequence
    seq = "HPPH"
    res = fold(seq, 2)
    print(res)
    print(_compute_energy(seq, [tuple(p) for p in res]))
