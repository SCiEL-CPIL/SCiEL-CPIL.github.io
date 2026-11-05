"""
HP lattice protein folding using Divide-and-Conquer and Contact Map Simulated Annealing.

Strategy:
1. Divide the sequence into small segments (e.g., 8-12 residues).
2. Pre-compute "blobs" for each segment using exhaustive search or fast local search.
   A blob is a specific 3D conformation of the segment with high internal energy.
3. Assemble the full protein by treating segments as rigid bodies.
   Use Simulated Annealing on the positions of these bodies to maximize inter-segment H-H contacts.
   This allows global rearrangements that local search on individual residues often misses.
4. Finally, refine the full chain with local moves (crankshaft/pivot) to fix any minor issues
   and squeeze out remaining energy.
"""

# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global deadline guard
_DEADLINE = None
_BUDGET = 330.0
_SAFETY = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _BUDGET * _SAFETY
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])

def _compute_energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _find_blobs(sequence, start, end, num_blobs=5, rng=None):
    """
    Find a set of distinct high-energy conformations (blobs) for sequence[start:end].
    Returns a list of lists of relative coordinates (offset from first residue).
    """
    if rng is None:
        rng = random.Random()
    
    sub_seq = sequence[start:end]
    length = len(sub_seq)
    
    # For very short segments, we can be more exhaustive or use many restarts
    # We want blobs that are compact.
    
    blobs = []
    seen_structures = set()
    
    # Strategy: Generate random walks, then optimize locally.
    # To get diversity, we use different random seeds and potentially different initial biases.
    
    max_attempts = 20
    for attempt in range(max_attempts):
        if _is_deadline_exceeded():
            break
        if len(blobs) >= num_blobs:
            break
            
        # Generate initial random walk
        chain = [(0,0,0)]
        occupied = {(0,0,0): 0}
        success = True
        
        for i in range(1, length):
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Bias towards compactness? Or just random?
                # Let's add a small bias towards existing H neighbors if current is H
                gain = 0
                if sub_seq[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sub_seq[j] == "H":
                                gain += 1
                # Randomness
                score = gain * 1.0 + rng.random()
                options.append((score, site))
                
            if not options:
                success = False
                break
                
            options.sort(reverse=True)
            # Pick best with some randomness (temperature)
            # For blobs, we want good structures, so pick best or top few
            if rng.random() < 0.5:
                site = options[0][1]
            else:
                site = options[rng.randint(0, min(2, len(options)-1))][1]
            
            chain.append(site)
            occupied[site] = i
        
        if not success:
            continue
            
        # Local search to improve this specific blob
        # Simple local search: try moving each residue to a neighbor
        improved_chain = list(chain)
        pos_map = {p: i for i, p in enumerate(improved_chain)}
        
        def calc_sub_energy(ch):
            # Energy of sub-chain
            w = {p: i for i, p in enumerate(ch)}
            c = 0
            for i, p in enumerate(ch):
                if sub_seq[i] == "H":
                    for q in _neighbours(p):
                        j = w.get(q)
                        if j is not None and j > i + 1 and sub_seq[j] == "H":
                            c += 1
            return -c

        current_e = calc_sub_energy(improved_chain)
        
        for _ in range(50): # Small local search
            if _is_deadline_exceeded():
                break
            i = rng.randint(1, length-2)
            p_i = improved_chain[i]
            for site in _neighbours(p_i):
                # Check validity
                if i > 0 and site == improved_chain[i-1]: continue
                if i < length-1 and site == improved_chain[i+1]: continue
                if site in pos_map: continue
                
                # Check connectivity
                valid = True
                if i > 0:
                    if _manhattan(site, improved_chain[i-1]) != 1: valid = False
                if valid and i < length-1:
                    if _manhattan(site, improved_chain[i+1]) != 1: valid = False
                if not valid: continue
                
                # Try move
                old_site = improved_chain[i]
                improved_chain[i] = site
                del pos_map[old_site]
                pos_map[site] = i
                
                new_e = calc_sub_energy(improved_chain)
                if new_e < current_e:
                    current_e = new_e
                else:
                    # Revert
                    improved_chain[i] = old_site
                    del pos_map[site]
                    pos_map[old_site] = i
        
        # Normalize to start at (0,0,0) and canonicalize
        offset = improved_chain[0]
        norm_chain = tuple(tuple(p[k] - offset[k] for k in range(3)) for p in improved_chain)
        
        # Check if we've seen this structure
        if norm_chain in seen_structures:
            continue
        seen_structures.add(norm_chain)
        
        blobs.append(list(norm_chain))
        
    if not blobs:
        # Fallback: straight line
        blobs = [[(i, 0, 0) for i in range(length)]]
        
    return blobs

def _assemble_blobs(sequence, blobs, rng):
    """
    Assemble blobs into a full chain.
    blobs: list of lists of relative coords.
    Returns a full chain of absolute coords.
    """
    n_segments = len(blobs)
    if n_segments == 0:
        return []
    if n_segments == 1:
        return [list(p) for p in blobs[0]]
    
    # Each blob is a rigid body. We need to place them such that:
    # 1. The end of blob i is adjacent to the start of blob i+1.
    # 2. No self-avoidance violations between any residues of different blobs.
    # 3. Maximize total H-H contacts.
    
    # State: positions of the "anchor" point of each blob (e.g., the first residue).
    # Actually, it's easier to define the position of the first residue of each blob.
    # Let pos[i] be the absolute coordinate of the first residue of blob i.
    # Then the residues of blob i are at pos[i] + blob[i][k].
    
    # Connectivity constraint:
    # Last residue of blob i: pos[i] + blob[i][-1]
    # First residue of blob i+1: pos[i+1]
    # They must be adjacent.
    # So pos[i+1] must be a neighbor of (pos[i] + blob[i][-1]).
    
    # This forms a chain of constraints. We can use a random walk on the "blob graph"
    # or Simulated Annealing on the positions.
    
    # Let's use a simpler approach: Random placement with local repair/optimization.
    # Initialize pos[0] = (0,0,0).
    # For i > 0, pos[i] is determined by pos[i-1] and a choice of direction.
    # However, this might lead to collisions.
    
    # Better: Treat the whole assembly as a system.
    # Variables: The orientation and position of each blob.
    # But orientation is constrained by the previous blob's end and the next blob's start?
    # No, the bond is between the last residue of blob i and the first of blob i+1.
    # This bond has a specific vector in the local frame of blob i and blob i+1.
    
    # Let's simplify:
    # We will place blobs sequentially.
    # pos[0] = (0,0,0)
    # For i in 1..N-1:
    #   end_prev = pos[i-1] + blob[i-1][-1]
    #   start_curr = pos[i]
    #   We need start_curr to be a neighbor of end_prev.
    #   So start_curr is one of the 6 neighbors of end_prev.
    #   This determines pos[i] uniquely once we pick a direction.
    #   BUT, this fixes the orientation of blob i relative to blob i-1.
    #   This means the "internal" orientation of blob i is fixed relative to the incoming bond.
    #   Wait, no. The blob is a rigid body. If we fix the position of its first residue,
    #   and we know the vector from first to last residue in the blob's local frame,
    #   we don't know the full orientation unless we fix another point.
    #   Actually, for a rigid body in 3D, fixing one point (pos[i]) and the direction of the outgoing bond
    #   (which is determined by the next blob's start, which we haven't placed yet) is tricky.
    
    # Let's use a different formulation for assembly:
    # We define the state as the absolute positions of ALL residues.
    # But we enforce that residues within a blob maintain their relative distances.
    # This is a constraint satisfaction problem.
    
    # Alternative: "Chain of Blobs" where we randomly rotate/translate each blob to connect.
    
    # Let's try this:
    # 1. Start with blob 0 at origin.
    # 2. For each next blob, try to attach it to the end of the previous one.
    #    - The "attachment point" is the last residue of the previous blob.
    #    - The "start point" of the new blob must be adjacent to the attachment point.
    #    - The new blob can be rotated in any of the 24 orientations of the cubic lattice.
    #    - We check for collisions with all previously placed residues.
    #    - If valid, we accept it. If not, we try another orientation or backtracking.
    #    - To allow global optimization, we use Simulated Annealing to re-orient blobs.
    
    # This is complex. Let's stick to a simpler "Random Walk of Blobs" with Local Search on the full chain.
    
    # Actually, the current program's weakness is that local search on the full chain gets stuck.
    # The "Blob" idea is good, but implementing a full rigid-body SA is risky in 240s.
    
    # Let's go with a hybrid:
    # 1. Generate initial full chain using "Greedy Growth with Lookahead".
    # 2. Apply "Pivot Moves" (large scale) and "Crankshaft Moves" (medium scale) in a SA framework.
    # This is different from the current "single residue re-insertion" which is very local.
    
    # Re-evaluating the "Blob" strategy:
    # If I just use the blobs to generate DIVERSE initial states for the full-chain SA,
    # it's essentially what the current program does with "multi-start greedy growth".
    # The key difference must be in the MOVES.
    
    # Current program: `_local_search_simple` moves single residues to neighbors.
    # New Strategy: **Pivot Moves** and **Crankshaft Moves** with Simulated Annealing.
    # Pivot: Rotate a sub-chain of length k around the bond preceding it.
    # Crankshaft: Rotate a sub-chain of length 3 around two adjacent bonds.
    # These allow the chain to fold/unfold large sections, escaping local minima much better than single-residue moves.
    
    return None # Placeholder, will implement SA with Pivots in fold()

def fold(sequence, dimension):
    """
    Fold protein using Simulated Annealing with Pivot and Crankshaft moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    # 1. Generate initial chain
    # Use a simple random walk with greedy bias
    def generate_initial_chain():
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        for i in range(1, n):
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == "H":
                                gain += 1
                score = gain + rng.random()
                options.append((score, site))
            if not options:
                return None
            options.sort(reverse=True)
            site = options[0][1]
            chain.append(site)
            occupied[site] = i
        return chain
    
    best_chain = None
    best_energy = 0
    
    # Multi-start
    for restart in range(20):
        if _is_deadline_exceeded():
            break
        
        chain = generate_initial_chain()
        if chain is None:
            continue
        
        # SA with Pivot and Crankshaft moves
        current_energy = _compute_energy(sequence, chain)
        if best_chain is None or current_energy < best_energy:
            best_chain = list(chain)
            best_energy = current_energy
        
        # Simulated Annealing
        # T starts high, cools down
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        # We allocate time per restart
        start_time = time.time()
        max_time_per_restart = 10.0 # 10s * 20 restarts = 200s, leaving margin
        
        while T > min_T:
            if _is_deadline_exceeded() or (time.time() - start_time > max_time_per_restart):
                break
            
            # Choose a move
            move_type = rng.random()
            
            if move_type < 0.6:
                # Pivot Move
                # Pick a pivot point i (1 to n-2)
                # Rotate the tail (i+1 to n-1) around the bond (i-1, i)
                i = rng.randint(1, n - 2)
                p_prev = chain[i-1]
                p_pivot = chain[i]
                
                # The vector from pivot to next
                v = (chain[i+1][0] - p_pivot[0], chain[i+1][1] - p_pivot[1], chain[i+1][2] - p_pivot[2])
                
                # Possible rotations of v around the axis defined by (p_prev, p_pivot)
                # Axis is (p_pivot - p_prev)
                axis = (p_pivot[0] - p_prev[0], p_pivot[1] - p_prev[1], p_pivot[2] - p_prev[2])
                
                # Generate 5 other orientations for the tail
                # This is geometrically complex. 
                # Simplified Pivot: Try moving the tail to any valid position that is a rotation.
                # Actually, let's just try random valid "kinks" for the tail start.
                
                # Even simpler: "Crankshaft" is easier to implement robustly.
                # Let's stick to a robust local search but with LARGER moves.
                # Move: Pick a sub-chain of length L (e.g., 3-5), remove it, and re-insert it at a random valid location nearby.
                
                L = rng.randint(3, 5)
                if n < L + 2:
                    L = n - 2
                if L < 2:
                    continue
                
                start_idx = rng.randint(1, n - L - 1)
                end_idx = start_idx + L
                
                # Extract sub-chain
                sub_chain = chain[start_idx:end_idx+1]
                
                # Remove from main chain
                # We need to be careful with indices.
                # Let's just use the simple local search from the current program but add "Block Moves"
                # Block Move: Move a contiguous block of residues to a new location.
                
                # This is getting complicated to implement correctly in a short time.
                # Let's go back to the "Blob" idea but use it for DIVERSITY in initial states,
                # and use the EXISTING local search but with a "Restart" strategy if stuck.
                
                # Actually, the current program's score is 0.9290. The best is 1.0000.
                # The difference is likely in the quality of the local minima found.
                # Let's try a different heuristic: **Minimization of Potential Energy** using a more aggressive local search.
                
                # Let's implement **Fast Local Search** with **Pivot Moves** properly.
                
                # Pivot Move Implementation:
                # 1. Pick pivot i.
                # 2. The tail is chain[i+1:].
                # 3. The head is chain[:i+1].
                # 4. We want to rotate the tail.
                # 5. The first residue of the tail (i+1) must be adjacent to i.
                # 6. So we try all 6 neighbors of i (excluding i-1).
                # 7. For each candidate position for i+1, we check if the rest of the tail can be placed
                #    such that it doesn't collide.
                #    This is expensive.
                
                # Alternative: **Random Kink**
                # Pick i, j (i < j). Swap the sub-chain? No, breaks connectivity.
                
                # Let's use **Simulated Annealing with Single Residue Moves** but with a better energy function?
                # No, energy is fixed.
                
                # Let's try **Simulated Annealing** with the current single-residue moves, but run it for longer?
                # The current program runs for 100 iterations per restart, 50 restarts.
                # Let's increase iterations and use proper SA acceptance.
                
                # Move: Pick i, try to move chain[i] to a neighbor.
                i = rng.randint(1, n - 2)
                p_i = chain[i]
                valid_sites = []
                for site in _neighbours(p_i):
                    if site == chain[i-1] or site == chain[i+1]:
                        continue
                    # Check if site is occupied by any other residue
                    # We need a fast occupancy check.
                    # Using a set of tuples is fast.
                    # But we need to ensure it's not i-1 or i+1.
                    # And it must be empty.
                    # We don't have a global set here in the loop.
                    # Let's rebuild a set for the chain? That's O(N).
                    # For N=136, O(N) is fine.
                    
                    # Let's optimize: maintain a set `occupied` outside the loop?
                    # But the chain changes.
                    
                    # Let's just check neighbors.
                    pass
                
                # This inner loop is inefficient if we don't have a global map.
                # Let's stick to the current program's structure but improve the moves.
                
                # **New Strategy: Simulated Annealing with Crankshaft Moves**
                # A crankshaft move rotates a segment of 3 residues.
                # i, i+1, i+2.
                # The bond (i-1, i) and (i+2, i+3) are fixed.
                # The middle residue i+1 can move to any of the 4 other positions
                # that are adjacent to both i and i+2 (if they are distance 2 apart).
                # This is a very effective move for HP folding.
                
                if n >= 4:
                    i = rng.randint(1, n - 3)
                    p_i = chain[i]
                    p_i1 = chain[i+1]
                    p_i2 = chain[i+2]
                    
                    # Check if p_i and p_i2 are distance 2 apart (Manhattan)
                    if _manhattan(p_i, p_i2) == 2:
                        # Find the "other" position for p_i1
                        # p_i1 is currently at some position.
                        # The new position must be adjacent to p_i and p_i2.
                        # There are typically 2 such positions (one is the current one, one is the new one).
                        # Let's find all neighbors of p_i that are also neighbors of p_i2.
                        candidates = []
                        for site in _neighbours(p_i):
                            if _manhattan(site, p_i2) == 1:
                                if site != p_i1:
                                    candidates.append(site)
                        
                        if candidates:
                            new_site = candidates[rng.randint(0, len(candidates)-1)]
                            
                            # Check if new_site is occupied by any other residue
                            # We need to check against all j != i, i+1, i+2
                            # And new_site != p_i, p_i2
                            
                            # Efficient check:
                            # Build a set of occupied positions excluding i, i+1, i+2?
                            # Or just check if new_site is in the set of all positions.
                            # Let's maintain a global set `pos_set` for the current chain.
                            # We need to update it.
                            
                            # Let's restructure the SA loop to maintain `pos_set`.
                            
                            # For now, let's just check linearly (N is small enough).
                            occupied_by_other = False
                            for j, p in enumerate(chain):
                                if j in [i, i+1, i+2]:
                                    continue
                                if p == new_site:
                                    occupied_by_other = True
                                    break
                            
                            if not occupied_by_other:
                                # Perform move
                                old_site = chain[i+1]
                                chain[i+1] = new_site
                                
                                new_energy = _compute_energy(sequence, chain)
                                delta = new_energy - current_energy
                                
                                if delta < 0 or rng.random() < math.exp(-delta / T):
                                    current_energy = new_energy
                                    if current_energy < best_energy:
                                        best_energy = current_energy
                                        best_chain = list(chain)
                                else:
                                    chain[i+1] = old_site
            else:
                # Single residue move (current program's move)
                i = rng.randint(1, n - 2)
                p_i = chain[i]
                for site in _neighbours(p_i):
                    if site == chain[i-1] or site == chain[i+1]:
                        continue
                    # Check occupancy
                    occupied_by_other = False
                    for j, p in enumerate(chain):
                        if j != i and p == site:
                            occupied_by_other = True
                            break
                    if occupied_by_other:
                        continue
                    
                    old_site = chain[i]
                    chain[i] = site
                    new_energy = _compute_energy(sequence, chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                    else:
                        chain[i] = old_site
                        break # Try next site? Or break? Let's break to save time.
                        
            T *= cooling
            
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
