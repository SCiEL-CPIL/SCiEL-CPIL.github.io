# EVOLVE-BLOCK-START
import random
import time
import math

def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 4 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (p[0] + dx, p[1] + dy)
                j = where.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _delta_energy(sequence, chain, i, new_pos):
    """Calculate change in energy if residue i moves to new_pos.
    Assumes chain[i] is the old pos.
    """
    old_pos = chain[i]
    # Remove contacts involving i
    # A contact is between i and j if |i-j|>1 and dist(chain[i], chain[j])==1 and both H.
    # We need to subtract contacts that are lost and add contacts that are gained.
    
    # This is complex to do incrementally without a global structure.
    # For N<=100, full recompute is ~ 100 * 4 lookups = 400 ops.
    # 120s / 10 seq = 12s/seq.
    # 12s / 400 ops = 30,000 full recomputes.
    # This is too slow for a tight loop if we do many moves.
    # But we can optimize _compute_energy.
    
    # Let's stick to full recompute but make it fast.
    pass

def _compute_energy_fast(sequence, chain):
    """Optimized energy calculation."""
    if len(chain) < 2:
        return 0
    # Use a set of H positions? No, we need indices to check adjacency.
    # Map position to index.
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            # Check right and down to avoid double counting?
            # No, contact is between pair (i, j).
            # Just check 4 neighbors, count if neighbor is H and not adjacent in chain.
            # To avoid double counting, we can just count all and divide by 2?
            # Or check only neighbors with index > i.
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x + dx, y + dy)
                j = pos_to_idx.get(q)
                if j is not None and j > i and sequence[j] == 'H':
                    # Check if they are adjacent in chain
                    if abs(i - j) > 1:
                        contacts += 1
    return -contacts

def _constructive_init(sequence, rng):
    """Greedy constructive heuristic."""
    n = len(sequence)
    if n == 0: return []
    if n == 1: return [(0,0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                # Calculate potential contacts
                contacts = 0
                if sequence[i] == 'H':
                    # Check against all previous H's
                    # Optimization: only check H's
                    for j in range(i):
                        if sequence[j] == 'H' and j != i - 1:
                            if _manhattan(new_pos, chain[j]) == 1:
                                contacts += 1
                # Heuristic: prefer more contacts. Add small random noise to break ties.
                score = -contacts + rng.uniform(0, 0.1)
                candidates.append((score, new_pos))
        
        if not candidates:
            return None # Failed
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Pick best, or random from top 3 to add diversity?
        # Let's pick best to be greedy.
        best_pos = candidates[0][1]
        chain.append(best_pos)
        occupied.add(best_pos)
        
    return chain

def _local_search(sequence, chain, time_budget, rng):
    """Iterated Local Search with Flip moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy_fast(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    start_time = time.time()
    
    # Parameters
    flip_min_len = 3
    flip_max_len = min(10, n - 1)
    
    while time.time() - start_time < time_budget:
        # Perform a local search step: try a few flips, take the best improvement
        improved = False
        # Try a random number of flips
        for _ in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Choose a segment to reverse
            l = rng.randint(0, n - flip_min_len - 1)
            r = rng.randint(l + flip_min_len - 1, n - 1)
            
            # Check if reversal is valid
            # The segment chain[l:r+1] is reversed.
            # We need to check:
            # 1. Self-avoidance of the new segment against the rest.
            # 2. Connectivity at l-1/l and r/r+1.
            
            new_chain = list(current_chain)
            # Reverse segment
            new_chain[l:r+1] = reversed(new_chain[l:r+1])
            
            # Validate
            valid = True
            # Check self-avoidance
            # This is O(N).
            seen = set()
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            
            if valid:
                # Check connectivity
                for i in range(1, n):
                    if _manhattan(new_chain[i], new_chain[i-1]) != 1:
                        valid = False
                        break
            
            if valid:
                new_energy = _compute_energy_fast(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        if not improved:
            # Perturb: reverse a random segment
            l = rng.randint(0, n - 2)
            r = rng.randint(l + 1, n - 1)
            current_chain[l:r+1] = reversed(current_chain[l:r+1])
            # Check validity of perturbation
            valid = True
            seen = set()
            for p in current_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if valid:
                for i in range(1, n):
                    if _manhattan(current_chain[i], current_chain[i-1]) != 1:
                        valid = False
                        break
            if valid:
                current_energy = _compute_energy_fast(sequence, current_chain)
            else:
                # Revert perturbation if invalid
                current_chain[l:r+1] = reversed(current_chain[l:r+1])
                # Energy should be back to previous, but let's recompute to be safe
                current_energy = _compute_energy_fast(sequence, current_chain)

    return best_chain

def fold(sequence, dimension):
    """Fold HP sequence using Constructive Heuristic + Iterated Local Search."""
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    rng = random.Random(42 + n)
    
    # Time distribution
    # Total budget 120s for 10 sequences. Assume we get ~10-15s per sequence in average case.
    # But we don't know how many times fold is called.
    # The prompt says "fold is called once per benchmark sequence... with a 120s CPU budget for all sequences together".
    # This implies the *total* time for all 10 calls is 120s.
    # So we should manage a global time budget?
    # No, the sandbox kills the process. We should just run as fast as possible.
    # However, if we run for 120s in the first sequence, we have 0 for the rest.
    # We should estimate time per sequence based on N.
    # N=20 is fast, N=100 is slow.
    # Let's allocate time proportional to N^2 or N.
    # N=100 is 25x N=20.
    # Let's try: time_budget = 1.0 + n * 0.1
    # N=20 -> 3s. N=100 -> 11s. Total ~ 10 * 5s = 50s. Safe.
    
    time_budget = 1.0 + n * 0.08
    
    best_chain = None
    best_energy = float('inf')
    
    # Multi-start
    num_starts = max(5, int(time_budget / 0.5))
    time_per_start = time_budget / num_starts
    
    for start in range(num_starts):
        # Constructive init
        initial = _constructive_init(sequence, rng)
        if initial is None:
            # Fallback: random walk
            initial = []
            occupied = set()
            cur = (0,0)
            initial.append(cur)
            occupied.add(cur)
            for _ in range(n-1):
                last = initial[-1]
                cands = [(last[0]+dx, last[1]+dy) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)] if (last[0]+dx, last[1]+dy) not in occupied]
                if not cands: break
                nxt = rng.choice(cands)
                initial.append(nxt)
                occupied.add(nxt)
            if len(initial) < n:
                # Fill with straight line?
                # Just use straight line if fail
                initial = [(i,0) for i in range(n)]
        
        # Local search
        improved = _local_search(sequence, initial, time_per_start, rng)
        
        energy = _compute_energy_fast(sequence, improved)
        if energy < best_energy:
            best_energy = energy
            best_chain = improved
            
        # Early stop if we found a very good solution?
        # Max possible contacts for N residues is roughly N/2 * 2? No.
        # Max contacts is bounded.
        
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
