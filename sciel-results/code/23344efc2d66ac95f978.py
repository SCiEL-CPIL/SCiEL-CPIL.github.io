"""
HP lattice protein folding using a Hamiltonian path skeleton strategy combined with local search.

Strategy:
1. Generate deterministic space-filling grid (Hamiltonian path) skeletons that visit
   cells in a bounding box. These paths have a high density of lattice neighbors.
2. Map the input sequence onto these skeletons.
3. Use a fast pivot-move local search to refine the conformation, allowing the
   chain to "breathe" and optimize H-H contacts while maintaining self-avoidance.
"""

import random
import time
import sys

# EVOLVE-BLOCK-START

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 200.0  # Leave 10s margin for the 210s kill

def _time_exceeded():
    return (time.time() - _START_TIME) > TIME_LIMIT

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Energy = - (number of H-H contacts)."""
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    # Optimization: Only check neighbors in one direction to avoid double counting
    # and ensure i < j
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _generate_zigzag_skeleton(width, height):
    """
    Generate a zig-zag Hamiltonian path for a width x height grid.
    Starts at (0,0).
    """
    path = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append((x, y))
        if y % 2 == 1:
            row.reverse()
        path.extend(row)
    return path

def _generate_spiral_skeleton(width, height):
    """
    Generate a spiral Hamiltonian path.
    Note: Simple spirals might not fill rectangular grids perfectly if dimensions are not square or specific conditions aren't met.
    For robustness, we stick to zig-zags and simple variations which always work for any WxH.
    """
    # A robust spiral is complex. Let's use a "comb" or "serpentine" variation instead.
    # Actually, zig-zag is the most reliable. Let's create a few variations by flipping axes.
    path = []
    # This is just a placeholder to keep the structure, we will use zigzag variants in the main loop.
    return _generate_zigzag_skeleton(width, height)

def _generate_skeletons(length):
    """
    Generate a set of diverse skeleton paths of at least 'length' nodes.
    """
    skeletons = []
    
    # Estimate bounding box size
    # Area must be >= length. Let's aim for a box slightly larger than sqrt(length) x sqrt(length)
    # to allow some flexibility, but tight is better for contact density.
    # We will try a few aspect ratios.
    
    sizes = []
    # Square-ish
    side = int(length ** 0.5)
    for w in range(max(1, side - 2), side + 3):
        for h in range(max(1, side - 2), side + 3):
            if w * h >= length:
                sizes.append((w, h))
    
    # Elongated
    for w in [1, 2, 3, 4]:
        h = (length + w - 1) // w
        sizes.append((w, h))
    for h in [1, 2, 3, 4]:
        w = (length + h - 1) // h
        sizes.append((w, h))
        
    # Remove duplicates
    unique_sizes = list(set(sizes))
    
    for w, h in unique_sizes:
        if w * h < length:
            continue
        # Generate zig-zag
        path = _generate_zigzag_skeleton(w, h)
        if len(path) >= length:
            skeletons.append(path[:length])
            
        # Generate zig-zag transposed (if w != h)
        if w != h:
            path_t = _generate_zigzag_skeleton(h, w)
            # Transpose coordinates
            path_t = [(y, x) for x, y in path_t]
            if len(path_t) >= length:
                skeletons.append(path_t[:length])
                
    # If no skeletons generated (should not happen), add a straight line
    if not skeletons:
        skeletons.append([(i, 0) for i in range(length)])
        
    return skeletons

def _valid_pivot(chain, i, j, new_seg):
    """
    Check if inserting new_seg (list of points) at position i (replacing chain[i:j+1])
    results in a valid self-avoiding walk.
    """
    # new_seg must connect chain[i-1] to chain[j+1]
    # Actually, pivot moves usually rotate a segment.
    # Let's define the move:
    # Cut at i and j (i < j).
    # Segment S = chain[i:j+1].
    # We want to rotate S around some pivot point?
    # Simpler: "Pivot move" often means selecting a segment and rotating it 90, 180, 270 degrees around one end.
    # Let's implement a simpler local search: "Sliding" or "Re-insertion".
    # Or even simpler: "2-opt" style swaps?
    # For HP model, "Pivot" usually refers to rotating a sub-chain around one of its endpoints.
    
    # Let's try a specific move: Rotate segment [i..j] around point i.
    # The rest of the chain stays fixed.
    # The new segment must be self-avoiding and not collide with the rest of the chain.
    pass

def _try_pivot_move(chain, seq, i, j, direction, rng):
    """
    Attempt a pivot move: rotate segment chain[i:j+1] around chain[i] by 'direction' degrees.
    direction: 1 (90 deg CW), 2 (180), 3 (270 deg CW).
    Returns new_chain if valid, else None.
    """
    if i == j:
        return None
        
    pivot_pt = chain[i]
    seg = chain[i:j+1]
    
    # Rotate segment relative to pivot
    # Vector from pivot to point p: v = (p[0]-pivot[0], p[1]-pivot[1])
    # Rotate v by 90 deg CW: (v[1], -v[0])
    # Rotate v by 180 deg: (-v[0], -v[1])
    # Rotate v by 270 deg CW: (-v[1], v[0])
    
    new_seg = [pivot_pt]
    for p in seg[1:]:
        vx = p[0] - pivot_pt[0]
        vy = p[1] - pivot_pt[1]
        
        if direction == 1:
            nvx, nvy = vy, -vx
        elif direction == 2:
            nvx, nvy = -vx, -vy
        elif direction == 3:
            nvx, nvy = -vy, vx
        else:
            return None
            
        new_pt = (pivot_pt[0] + nvx, pivot_pt[1] + nvy)
        new_seg.append(new_pt)
        
    # Check self-avoidance of the whole new chain
    # The rest of the chain is chain[:i] and chain[j+1:]
    # new_chain = chain[:i] + new_seg[1:] + chain[j+1:]
    # Note: new_seg[0] is the pivot, which is already in chain[:i] (as the last element).
    # So we replace chain[i:j+1] with new_seg[1:].
    
    prefix = chain[:i]
    suffix = chain[j+1:]
    
    candidate_chain = prefix + new_seg[1:] + suffix
    
    # Check collisions
    # 1. New segment internal collisions
    # 2. New segment vs prefix (excluding the pivot which is shared)
    # 3. New segment vs suffix
    # 4. Prefix vs Suffix (unchanged, so if original was valid, this is valid)
    
    # Efficient check:
    # Build set of occupied points in prefix and suffix
    occupied = set(prefix)
    occupied.update(suffix)
    
    # Check new_seg[1:] against each other and occupied
    seen_in_new = set()
    for p in new_seg[1:]:
        if p in occupied or p in seen_in_new:
            return None
        seen_in_new.add(p)
        
    # Also need to check that the connection is valid
    # Connection 1: prefix[-1] (which is chain[i-1]) to new_seg[0] (pivot, chain[i]) -> Unchanged
    # Connection 2: new_seg[-1] to suffix[0] (chain[j+1])
    if suffix:
        last_new = new_seg[-1]
        first_suf = suffix[0]
        if abs(last_new[0] - first_suf[0]) + abs(last_new[1] - first_suf[1]) != 1:
            return None
            
    return candidate_chain

def _local_search(sequence, initial_chain, max_iterations=1000):
    """
    Perform local search using pivot moves.
    """
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    n = len(current_chain)
    rng = random.Random(42)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    for _ in range(max_iterations):
        if _time_exceeded():
            break
            
        # Randomly pick a segment to pivot
        # Bias towards shorter segments for faster convergence, but allow longer ones
        i = rng.randint(0, n - 2)
        length_seg = rng.randint(1, min(10, n - i - 1)) # Cap segment length for speed
        j = i + length_seg
        direction = rng.randint(1, 3)
        
        new_chain = _try_pivot_move(current_chain, sequence, i, j, direction, rng)
        
        if new_chain:
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # Allow some random moves to escape local minima? 
                # Simulated annealing could be used, but simple hill climbing is often enough for these benchmarks if restarts are used.
                # Let's stick to strict hill climbing for simplicity and speed, relying on multiple skeletons/restarts.
                
        if no_improvement_count > max_no_improvement:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    # Generate skeletons
    skeletons = _generate_skeletons(n)
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any valid folded state
    
    # Shuffle skeletons to randomize order
    rng = random.Random(12345 + n)
    rng.shuffle(skeletons)
    
    for skeleton in skeletons:
        if _time_exceeded():
            break
            
        # Initialize chain from skeleton
        initial_chain = [tuple(p) for p in skeleton]
        
        # Local Search
        chain, energy = _local_search(sequence, initial_chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Early stop if we hit a very low energy (optional, hard to know "best" a priori)
        
    # If no valid chain found (should not happen), return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple test
    seq = "HHPPHHPPHH"
    res = fold(seq, 2)
    print(res)
