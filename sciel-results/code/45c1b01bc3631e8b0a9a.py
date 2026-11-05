"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    """Calculate energy: -number of non-consecutive H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, dimension, rng, time_limit, start_time):
    """
    Perform Monte Carlo local search with simulated annealing.
    Uses single-residue moves (move a residue to an adjacent empty site).
    """
    current_chain = [list(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    
    # Annealing parameters
    # Start with a high temperature to allow exploration, cool down
    T = 2.0 
    T_min = 0.01
    cooling_rate = 0.999
    
    # Pre-compute neighbor directions for efficiency
    # We will iterate through indices and try to move them
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            # If cooled too much, we might try a quick restart or just stop
            # For simplicity, we just stop or try a few more steps at T_min
            if rng.random() > 0.5:
                break
        
        # Pick a random index to move. Avoid ends to keep it simple? 
        # Moving ends is safer but less flexible. Moving middle is more powerful.
        # Let's pick any index 1..L-2 for "crank-like" single moves, 
        # or actually, just any index. If we move index i, we must ensure 
        # it doesn't collide with i-1 or i+1.
        
        i = rng.randint(1, len(sequence) - 2)
        
        orig_pos = tuple(current_chain[i])
        
        # Get neighbors of orig_pos
        moves = list(_neighbours(orig_pos))
        # Shuffle to be unbiased
        rng.shuffle(moves)
        
        for new_pos in moves:
            # Check if new_pos is occupied by any other residue
            # We need to check against all residues except i-1, i, i+1
            # Actually, if new_pos is occupied by i-1 or i+1, it's a collision.
            # If occupied by any other, it's a collision.
            
            # Efficient check: 
            # 1. Is new_pos in the set of all occupied positions?
            # 2. If yes, is it one of the neighbors in the chain (i-1 or i+1)?
            
            # We don't have a global set of occupied positions updated dynamically in this loop easily 
            # without rebuilding it. Rebuilding is O(N). 
            # For N=100, O(N) is 100 operations. 100 * 1000 steps = 100k ops. This is fine in Python if optimized.
            
            # Let's use a set for occupied positions for O(1) lookup.
            # But we are modifying the chain. We should maintain an occupied set.
            
            # Let's refactor to maintain an occupied set.
            # This function currently takes chain. Let's assume we maintain it internally.
            
            # To keep the interface simple, I'll implement the occupied set inside the loop or pass it.
            # For now, let's just check against the list. It's O(N).
            # Given N <= 136, O(N) check is ~136 comparisons. 
            # 1000 steps * 136 = 136,000 comparisons. Very fast.
            
            is_occupied = False
            for j, pos in enumerate(current_chain):
                if pos == new_pos:
                    # It's occupied.
                    # Check if it's the immediate neighbor in the chain
                    if j == i - 1 or j == i + 1:
                        # Collision with chain neighbor
                        is_occupied = True
                        break
                    else:
                        # Collision with distant part of chain
                        is_occupied = True
                        break
            
            if is_occupied:
                continue
            
            # Valid move!
            # Calculate energy change
            # Remove i from orig_pos, add i to new_pos
            # We can compute delta energy by checking contacts involving i.
            
            # Contacts lost:
            # H at i interacting with H at j (j != i-1, i+1) at orig_pos
            # H at j interacting with H at i at orig_pos
            
            # This is complex to do incrementally without the global map.
            # Given the small N, let's just recompute energy if we accept? 
            # No, we need delta to decide acceptance.
            
            # Let's build a position map for the current chain for this step? 
            # That's O(N).
            # Total cost per move attempt: O(N) for collision check + O(N) for energy map + O(N) for energy calc?
            # That might be too slow if we do many steps.
            
            # Better approach: Maintain the `occupied` dict and `energy` incrementally.
            
            # Let's rewrite _local_search to maintain state.
            
            # Since I'm writing the code from scratch in the block, I will implement a version that maintains state.
            
            # Break out of the move loop if we found a valid move to process
            break
        
        # If no valid move found for index i, continue to next iteration
        # (This is inefficient, we should try other indices)
        
        # Actually, the above logic is flawed. We should try to find a valid move first.
        
        # Let's restart the local search function with a cleaner state management.
        
        # I will implement `_local_search` properly below.
        
        # For now, this placeholder logic is incomplete.
        pass
    
    return current_chain, current_energy

def _local_search_efficient(sequence, chain, dimension, rng, time_limit, start_time):
    current_chain = [list(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    
    # Maintain occupied set
    occupied = {tuple(p): i for i, p in enumerate(current_chain)}
    
    T = 2.0
    T_min = 0.05
    cooling_rate = 0.9995
    
    L = len(sequence)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random index
        i = rng.randint(1, L - 2)
        
        orig_pos = tuple(current_chain[i])
        neighbors = list(_neighbours(orig_pos))
        rng.shuffle(neighbors)
        
        move_made = False
        for new_pos in neighbors:
            # Check collision
            if new_pos in occupied:
                j = occupied[new_pos]
                # If it's the chain neighbor, it's a collision
                if abs(j - i) == 1:
                    continue
                else:
                    # Collision with distant part
                    continue
            
            # Valid move candidate
            # Calculate delta energy
            # We need to know which H-H contacts are broken and formed.
            # Contacts involving index i.
            
            # Old contacts for i:
            # If sequence[i] == 'H':
            #   For each neighbor q of orig_pos:
            #     if q in occupied and sequence[occupied[q]] == 'H' and abs(occupied[q]-i) > 1:
            #        delta_E += 1 (contact lost, energy increases by 1)
            
            # New contacts for i:
            # If sequence[i] == 'H':
            #   For each neighbor q of new_pos:
            #     if q in occupied and sequence[occupied[q]] == 'H' and abs(occupied[q]-i) > 1:
            #        delta_E -= 1 (contact gained, energy decreases by 1)
            
            # Note: We must be careful not to count contacts with i-1 and i+1.
            
            delta_E = 0
            
            if sequence[i] == 'H':
                # Loss of contacts at orig_pos
                for q in _neighbours(orig_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            delta_E += 1
                
                # Gain of contacts at new_pos
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            delta_E -= 1
            
            # Acceptance criterion
            if delta_E < 0:
                accept = True
            else:
                # Metropolis criterion
                # P = exp(-delta_E / T)
                # delta_E is positive here (energy increase)
                accept = rng.random() < math.exp(-delta_E / T)
            
            if accept:
                # Perform move
                # Update occupied
                del occupied[orig_pos]
                occupied[new_pos] = i
                
                # Update chain
                current_chain[i] = list(new_pos)
                
                # Update energy
                current_energy += delta_E
                
                move_made = True
                break
        
        if not move_made:
            # No valid move for this i, try next i
            pass
            
        T *= cooling_rate
    
    return current_chain, current_energy

def fold(sequence, dimension):
    """
    Best conformation using greedy growth + local search.
    """
    start_time = time.time()
    # Reserve 10% of time for final checks/overhead
    total_budget = 240.0
    safety_margin = 20.0
    effective_budget = total_budget - safety_margin
    
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Greedy Growth
    # Generate a set of candidate chains
    candidates = []
    # Use 50% of budget for growth
    growth_budget = effective_budget * 0.2
    
    num_growths = 100 # Fixed number, growth is fast
    
    for _ in range(num_growths):
        if time.time() - start_time > start_time + growth_budget:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            candidates.append(chain)
    
    if not candidates:
        # Fallback: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    # Phase 2: Local Search
    # Sort candidates by energy and pick the best few
    # Or just pick the best one and search it hard?
    # Better: Pick top K candidates, run local search on each, keep best result.
    
    # Evaluate energies of candidates
    scored_candidates = []
    for chain in candidates:
        e = _energy(sequence, chain)
        scored_candidates.append((e, chain))
    
    scored_candidates.sort(key=lambda x: x[0]) # Sort by energy (most negative first)
    
    # Take top 3 candidates for local search
    top_candidates = [c[1] for c in scored_candidates[:3]]
    
    best_chain = top_candidates[0]
    best_energy = scored_candidates[0][0]
    
    # Allocate remaining time to local search
    # Split time among top candidates? Or just give all to the best one?
    # Giving all to the best one is usually safer.
    
    search_budget = effective_budget - (time.time() - start_time)
    
    # Run local search on the best candidate
    if search_budget > 1.0:
        improved_chain, improved_energy = _local_search_efficient(
            sequence, best_chain, dimension, rng, search_budget * 0.9, time.time()
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # If we have time left, try the second best candidate?
    # If the first one didn't improve much, maybe the second one has a different basin.
    # But for simplicity and robustness, stick to the best one.
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
