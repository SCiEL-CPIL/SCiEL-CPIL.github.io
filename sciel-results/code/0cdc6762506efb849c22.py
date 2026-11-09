"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow_csp(sequence, dimension, rng):
    """
    Constraint-Satisfaction-Problem framed growth.
    Instead of pure greedy, it looks ahead to ensure the chain doesn't trap itself
    and prioritizes H-H contact formation while maintaining connectivity options.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    N = len(sequence)
    
    for i in range(1, N):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Heuristic 1: Immediate H-H contact gain
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Heuristic 2: Connectivity / Trapping check (CSP constraint)
            # Count free neighbors. If 0, it's a dead end (unless it's the last residue).
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # If we are not at the end, we must have at least 1 free neighbor to continue
            if i < N - 1 and free == 0:
                continue
            
            # Heuristic 3: Prefer sites that are "close" to other H residues for future contacts
            # (Approximated by checking if the site is near existing H's, even if not immediate contact)
            # This is a soft constraint to guide towards compactness
            density = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == "H":
                        density += 1 # Already counted in gain, but reinforces compactness
            elif sequence[i] == "P":
                # P residues should bridge H's. Prefer sites between H's if possible?
                # Simple proxy: prefer sites with more occupied neighbors (compactness)
                for q in _neighbours(site):
                    if q in occupied:
                        density += 0.5

            # Randomization to escape local minima in growth
            score = gain * 10.0 + density + free * 0.1 + rng.random()
            options.append((score, site))
            
        if not options:
            return None
        
        # Pick the best option, but with some randomness to allow exploration
        # Sort by score descending
        options.sort(key=lambda x: x[0], reverse=True)
        
        # With 80% probability pick the best, otherwise pick randomly from top 3
        if len(options) > 1 and rng.random() < 0.8:
            site = options[0][1]
        else:
            top_k = min(3, len(options))
            site = options[rng.randint(0, top_k - 1)][1]
            
        occupied[site] = i
        chain.append(site)
        
    return chain

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

def _rotate_point(point, center, rot):
    x, y = point[0] - center[0], point[1] - center[1]
    if rot == 0:
        nx, ny = x, y
    elif rot == 90:
        nx, ny = -y, x
    elif rot == 180:
        nx, ny = -x, -y
    else: # 270
        nx, ny = y, -x
    return (center[0] + nx, center[1] + ny)

def _incremental_energy_update(sequence, chain, pivot_idx, new_segment, current_energy):
    """
    Fast energy calculation for a pivot move.
    Only re-evaluates contacts involving the moved segment and its boundary with the fixed part.
    """
    N = len(chain)
    k = pivot_idx
    
    # Contacts to remove:
    # 1. H-H contacts between segment [k..N-1] and fixed part [0..k-2]
    # 2. H-H contacts within the segment that are broken (actually, internal contacts remain if relative positions don't change? 
    #    Wait, pivot rotation preserves relative distances within the segment. So internal H-H contacts are INVARIANT.
    #    Only contacts crossing the boundary (segment vs fixed) change.
    #    Also, the contact between chain[k-1] and chain[k] is preserved? No, the angle changes, but they are still adjacent.
    #    Actually, the set of pairs (i,j) where |i-j|>1 and dist(i,j)=1 changes only for pairs where one is in fixed and one in segment.
    #    Pairs within fixed: invariant.
    #    Pairs within segment: invariant (rigid body rotation).
    #    Pairs crossing boundary: must be recalculated.
    
    # Let's verify:
    # Energy = - ( # of H-H pairs (i,j) with |i-j|>1 and Manhattan dist 1 )
    # If we rotate segment k..N-1 around k-1:
    # - Pairs (i,j) with i,j < k: Unchanged.
    # - Pairs (i,j) with i,j >= k: Unchanged (relative positions preserved).
    # - Pairs (i,j) with i < k and j >= k: Changed.
    
    # So we only need to subtract old crossing contacts and add new crossing contacts.
    
    # Identify H positions in fixed part and segment part
    # Fixed part indices: 0 .. k-1
    # Segment part indices: k .. N-1
    
    # Note: The pivot is k-1. The segment starts at k.
    # The pair (k-1, k) is adjacent along the chain, so it's never counted in energy (|i-j|=1).
    # So we only care about pairs (i, j) where i <= k-2 and j >= k.
    
    # Build set of H positions in fixed part (indices < k)
    # And H positions in new segment
    
    # To be safe and simple, let's just recalculate energy for the whole chain if N is small, 
    # but for N up to 100, incremental is faster.
    
    # Let's do a hybrid: 
    # 1. Identify all H indices in fixed part (0 to k-1)
    # 2. Identify all H indices in segment (k to N-1)
    
    # Old crossing contacts:
    # For each H in fixed part, check its neighbors in the OLD chain.
    # For each H in segment, check its neighbors in the OLD chain.
    
    # This is still O(N) in worst case but with small constant.
    # Given N<=100, full recalculation is O(N^2) ~ 10000 ops.
    # Incremental is O(k + (N-k)) ~ 100 ops.
    
    # Let's implement incremental properly.
    
    # 1. Remove old crossing contacts
    # Old chain is `chain`.
    # Fixed part is chain[0:k]. Segment is chain[k:N].
    # We need to count H-H pairs (i,j) with i < k, j >= k, |i-j|>1, dist(chain[i], chain[j])=1.
    
    # Since N is small (<=100), and we are doing this many times, 
    # a full recalculation of _energy might be surprisingly fast in Python if optimized?
    # No, _energy is O(N^2). 100^2 = 10000.
    # If we do 100,000 moves, that's 10^9 ops. Too slow.
    # Incremental is necessary.
    
    # Let's use a set of occupied positions for quick lookup?
    # We already have `where` in _energy, but we need it here.
    
    # Let's precompute the set of H positions in the fixed part.
    fixed_h_positions = set()
    for i in range(k):
        if sequence[i] == 'H':
            fixed_h_positions.add(chain[i])
            
    # Old crossing contacts:
    # For each position in the old segment (chain[k:N]), if it's H, check if any of its neighbors are in fixed_h_positions.
    # But we must ensure the index difference > 1.
    # Since i < k and j >= k, the max index in fixed is k-1. The min index in segment is k.
    # So |i-j| >= 1.
    # The only case where |i-j|=1 is when i=k-1 and j=k.
    # So we just need to exclude the pair (k-1, k) if both are H? No, (k-1,k) is adjacent in chain, so dist=1.
    # But energy definition excludes adjacent residues.
    # So if chain[k-1] is H and chain[k] is H, they are neighbors in lattice but adjacent in chain -> NOT a contact.
    # My check `j > i + 1` in _energy handles this.
    # In incremental:
    # If i = k-1 and j = k, they are adjacent in chain. Exclude.
    # All other pairs (i < k-1, j >= k) have |i-j| >= 2.
    
    old_crossing = 0
    # Iterate over segment positions
    for j in range(k, N):
        if sequence[j] == 'H':
            pos = chain[j]
            for nb in _neighbours(pos):
                if nb in fixed_h_positions:
                    # Check if this neighbor is chain[k-1] and j==k
                    # If nb == chain[k-1] and j == k:
                    #   i = k-1, j = k. |i-j|=1. Exclude.
                    if nb == chain[k-1] and j == k:
                        continue
                    old_crossing += 1
                    # Note: Could a fixed H neighbor be counted twice? 
                    # No, we are iterating over segment H's. Each contact is counted once from the segment side.
                    # Wait, if two fixed H's are neighbors to one segment H? 
                    # Yes, that's 2 contacts.
                    # If one fixed H is neighbor to two segment H's?
                    # Yes, that's 2 contacts.
                    
    # New crossing contacts:
    # The segment has moved to `new_segment`.
    # Fixed part is unchanged.
    new_crossing = 0
    for idx, pos in enumerate(new_segment):
        j = k + idx
        if sequence[j] == 'H':
            for nb in _neighbours(pos):
                if nb in fixed_h_positions:
                    if nb == chain[k-1] and j == k:
                        continue
                    new_crossing += 1
                    
    delta_e = -(new_crossing - old_crossing)
    return current_energy + delta_e

def fold(sequence, dimension):
    """
    Best conformation using CSP-guided initialization and Incremental Simulated Annealing.
    """
    rng = random.Random(12345 + len(sequence))
    N = len(sequence)
    
    # 1. Generate initial valid chain using CSP-guided growth
    best_chain = None
    best_energy = 0
    
    # Try a few initializations
    for _ in range(20):
        chain = _grow_csp(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
        if best_energy <= -len(sequence) * 0.6: # Good enough
            break

    if best_chain is None:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(N)]]

    chain = best_chain
    current_energy = best_energy
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.995 # Slower cooling for more exploration
    
    # Time check
    while time.time() < _DEADLINE:
        # Pick a pivot
        # Bias towards shorter segments? Or random?
        # Random is standard.
        k = rng.randint(1, N - 2)
        rot = rng.choice([90, 180, 270])
        
        # Rotate segment k..N-1 around chain[k-1]
        pivot = chain[k-1]
        new_segment = []
        for i in range(k, N):
            new_segment.append(_rotate_point(chain[i], pivot, rot))
        
        # Check collision with fixed part (0..k-2)
        # fixed_set = set(chain[:k-1])
        # To speed up, we can check only the new_segment against the fixed part.
        # Since N is small, building the set is O(k).
        fixed_set = set(chain[:k-1])
        
        valid = True
        for p in new_segment:
            if p in fixed_set:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate new energy incrementally
        # Note: _incremental_energy_update expects the OLD chain to compute old_crossing.
        # It uses `chain` for the old positions.
        new_energy = _incremental_energy_update(sequence, chain, k, new_segment, current_energy)
        
        delta_e = new_energy - current_energy
        
        # Acceptance
        if delta_e < 0 or (T > 0 and rng.random() < exp(-delta_e / T)):
            chain = chain[:k] + new_segment
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain
        
        T *= cooling
        if T < T_min:
            T = 5.0 # Restart temperature (Re-annealing)
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
