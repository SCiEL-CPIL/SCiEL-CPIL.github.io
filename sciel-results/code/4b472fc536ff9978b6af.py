"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    """Yield the 4 neighbors of a point on the 2D lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation. Energy = -number of H-H contacts."""
    n = len(chain)
    # Map position to index for O(1) lookup
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = pos_to_idx.get((nx, ny))
                if j is not None and j > i + 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _grow_guided(sequence, dimension, rng):
    """
    Grow the chain using a heuristic that strongly prefers creating H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Heuristic score
            score = rng.random() # Base randomness to avoid getting stuck
            
            if sequence[i] == 'H':
                # Check potential H-H contacts with previous H's
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10.0 # Strong weight for new contacts
            
            # Small bonus for free space to avoid trapping
            free_space = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free_space
            
            candidates.append((score, site))
        
        if not candidates:
            return None # Dead end
        
        # Pick the best candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_search(sequence, chain, rng, max_iters=1000):
    """
    Improve a conformation using greedy local search (single residue moves).
    Returns the improved chain.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try to move each residue
        # Use random order to break symmetry
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if i == 0 or i == n - 1:
                # End residues are harder to move without breaking chain connectivity?
                # Actually, we can move any residue as long as we maintain unit steps.
                # Moving residue i requires checking neighbors i-1 and i+1.
                pass
            
            x, y = current_chain[i]
            
            # Try 4 directions
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                new_pos = (nx, ny)
                
                # Check if new position is occupied by another part of the chain
                # We need a fast way to check occupancy.
                # For local search, rebuilding the set every time is O(N^2) per move, which is slow.
                # Let's use a set for current positions.
                
                # Optimization: Maintain a set of occupied positions
                # This implementation below is O(N) per move check due to set lookup if we had one.
                # Let's optimize by keeping a set.
                
                # To keep it simple and robust within time limits, we will use a set
                # but we must update it carefully.
                
                # Let's re-implement local search with a maintained set for speed.
                # This specific function structure is getting complex to optimize inline.
                # Let's use a simpler, slightly slower but correct approach for the first pass,
                # or just limit the local search depth.
                
                # Actually, let's write a dedicated local search that maintains state.
                pass
                
    return current_chain

def _local_search_optimized(sequence, chain, rng, max_passes=50):
    """
    Optimized local search maintaining a set of occupied positions.
    """
    n = len(chain)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_passes):
        improved = False
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            x, y = current_chain[i]
            
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                new_pos = (nx, ny)
                
                # Check if new_pos is free
                # It must not be in occupied, unless it is the position of i itself (which it isn't, since we're moving)
                # But we must ensure it doesn't collide with neighbors i-1 or i+1?
                # No, if we move i to new_pos, we must ensure:
                # 1. new_pos is not occupied by any j != i.
                # 2. The step from i-1 to new_pos is a unit step? No, i-1 is fixed.
                #    Wait, if we move i, the distance from i-1 to new_pos must be 1.
                #    And the distance from new_pos to i+1 must be 1.
                
                # Check unit step constraint with neighbors
                if i > 0:
                    prev_x, prev_y = current_chain[i-1]
                    if abs(prev_x - nx) + abs(prev_y - ny) != 1:
                        continue
                
                if i < n - 1:
                    next_x, next_y = current_chain[i+1]
                    if abs(next_x - nx) + abs(next_y - ny) != 1:
                        continue
                
                # Check collision
                if new_pos in occupied and occupied[new_pos] != i:
                    continue
                
                # If valid, calculate energy change
                # Remove i from old position, add to new
                # Update occupied
                old_pos = current_chain[i]
                
                # Temporarily update
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                
                new_energy = _energy(sequence, current_chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert
                    current_chain[i] = old_pos
                    del occupied[new_pos]
                    occupied[old_pos] = i
                    
        if not improved:
            break
            
    return current_chain

def fold(sequence, dimension):
    """
    Best conformation using multiple guided growths followed by local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than -1
    
    n = len(sequence)
    
    # If sequence is very short, just return a simple valid conformation
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    # Number of restarts depends on time
    # Estimate cost per iteration.
    # Growth is O(N^2) roughly due to neighbor checks.
    # Local search is O(Passes * N * 4 * Cost_Energy).
    # Cost_Energy is O(N) if we rebuild the map.
    # So one iteration is roughly O(N^2 + Passes * N^2).
    
    # Let's try a fixed number of restarts first, then use remaining time for more if fast.
    # Or just run until time runs out.
    
    attempts = 0
    while time.time() < deadline:
        # 1. Grow
        chain = _grow_guided(sequence, dimension, rng)
        
        if chain is None:
            continue
            
        # 2. Local Search
        # Limit local search passes based on time remaining
        time_left = deadline - time.time()
        if time_left < 1.0:
            # Not enough time for a full polish, just take the raw grown chain
            polished_chain = chain
        else:
            # Estimate max passes
            # For N=100, Energy calc is ~100 ops. Local search pass is ~100*4*100 = 40000 ops.
            # We can afford maybe 10-20 passes in 1 second?
            # Let's cap passes to 20 to be safe.
            polished_chain = _local_search_optimized(sequence, chain, rng, max_passes=20)
            
        # 3. Evaluate
        energy = _energy(sequence, polished_chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = polished_chain
            best_energy = energy
            
        attempts += 1
        
        # If we've found a very good energy (e.g., close to theoretical min), we could stop early?
        # Theoretical min for HP is roughly -0.5 * N_H^2 ? No, max contacts is limited by lattice.
        # For now, just keep searching.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
