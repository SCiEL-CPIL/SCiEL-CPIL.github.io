"""HP lattice protein folding using hybrid pivot/flip search with greedy polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time tracking
_start_time = time.time()
_TOTAL_BUDGET = 120.0
_SAFETY_MARGIN = 5.0

def _get_time_remaining():
    return _TOTAL_BUDGET - (time.time() - _start_time) - _SAFETY_MARGIN

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
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
            # Bias towards gains, but keep some randomness to explore
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick best with some randomness (softmax-like or just top-k)
        # For simplicity, pick max, but add noise to score
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_flip_move(chain, i, j, occupied_set):
    """
    Attempt to reverse the segment chain[i..j].
    Returns new_chain if valid, else None.
    """
    n = len(chain)
    if i >= j:
        return None
    
    # The segment to flip is indices i to j.
    # The rest of the chain remains fixed.
    # We need to check if the new positions of i..j collide with the rest.
    
    # Optimization: Only check collisions with the "outside" parts (0..i-1 and j+1..n-1)
    # The internal collisions within i..j are preserved by the reversal property?
    # No, if we reverse positions, the set of occupied sites in the segment is the same.
    # So internal collisions are not an issue.
    # We only need to check if any new position in i..j collides with 0..i-1 or j+1..n-1.
    
    # However, the positions themselves change.
    # Let P_k be the position of residue k.
    # New position of residue k (for i<=k<=j) is P_{i+j-k}.
    # So the set of positions occupied by the segment is invariant.
    # Therefore, we only need to check if the "boundary" of the segment causes collisions?
    # Actually, since the set of positions is the same, if it was valid before, the set of positions is still valid.
    # BUT, the chain connectivity must be maintained.
    # Reversing a segment in a self-avoiding walk is always a valid self-avoiding walk IF the original was valid?
    # Yes, because the set of vertices is the same. The edges are reversed.
    # Wait, does the reversal change the positions?
    # NO. A flip move in SA for SAWs usually means **reversing the order of vertices in the list**, 
    # which implies the **positions** are assigned to the new indices.
    # If we just reverse the list `chain[i:j+1]`, the positions associated with those indices change.
    # Example: chain = [A, B, C, D]. Flip 1..2.
    # Old: idx1=B, idx2=C.
    # New: idx1=C, idx2=B.
    # The position of idx1 is now C's old position.
    # So the set of occupied positions for the segment is the same.
    # Therefore, no new collisions with the outside are possible!
    # And internal collisions are impossible because it was a valid SAW before.
    # So a Flip move is ALWAYS valid for a SAW?
    # Yes, reversing a sub-chain of a self-avoiding walk results in another self-avoiding walk.
    # This is a known property.
    
    new_chain = chain[:]
    new_chain[i:j+1] = reversed(new_chain[i:j+1])
    return new_chain

def _hybrid_search(sequence, chain, max_iters, rng, time_limit):
    """
    Simulated Annealing with Pivot and Flip moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Pivot Move
            k = rng.randint(0, n - 2)
            v_x = current_chain[k+1][0] - current_chain[k][0]
            v_y = current_chain[k+1][1] - current_chain[k][1]
            
            rot_choice = rng.choice([1, 2, 3])
            if rot_choice == 1:
                new_dx, new_dy = -v_y, v_x
            elif rot_choice == 2:
                new_dx, new_dy = -v_x, -v_y
            else:
                new_dx, new_dy = v_y, -v_x
                
            # Apply rotation
            if rot_choice == 1:
                def rot(x, y): return (-y, x)
            elif rot_choice == 2:
                def rot(x, y): return (-x, -y)
            else:
                def rot(x, y): return (y, -x)
                
            px, py = current_chain[k]
            prefix_occupied = set(current_chain[:k])
            
            valid = True
            new_chain = current_chain[:k+1] + [None] * (n - k - 1)
            
            for j in range(k + 1, n):
                ox = current_chain[j][0] - px
                oy = current_chain[j][1] - py
                nx, ny = rot(ox, oy)
                new_pos = (px + nx, py + ny)
                
                if new_pos in prefix_occupied:
                    valid = False
                    break
                new_chain[j] = new_pos
            
            if not valid:
                continue
                
        else:
            # Flip Move
            # Pick a segment length
            seg_len = rng.randint(2, min(n-1, 10)) # Small flips are more likely to be accepted?
            # Actually, large flips explore more. Let's mix.
            if rng.random() < 0.5:
                seg_len = rng.randint(2, n-1)
            else:
                seg_len = rng.randint(2, 5)
                
            i = rng.randint(0, n - seg_len - 1)
            j = i + seg_len
            
            new_chain = _try_flip_move(current_chain, i, j, None)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
        
    return best_chain, best_energy

def _greedy_polish(sequence, chain, max_passes=10):
    """
    Deterministic local search.
    Tries to improve energy by flipping small segments or pivoting single bonds if it reduces energy.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_passes):
        improved = False
        # Try small flips
        for length in [2, 3, 4]:
            for i in range(0, n - length - 1):
                j = i + length
                new_chain = _try_flip_move(current_chain, i, j, None)
                if new_chain:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
        if not improved:
            break
            
    return current_chain, current_energy

def fold(sequence, dimension):
    """Best conformation using hybrid search and polish."""
    remaining_time = _get_time_remaining()
    
    # Fallback if no time
    if remaining_time < 0.5:
        n = len(sequence)
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    n = len(sequence)
    # Allocate time based on sequence length
    # Longer sequences need more time, but we have a fixed total budget.
    # Let's assume uniform allocation for simplicity, or slightly more for longer ones.
    # 10 sequences. ~10s each.
    time_limit = max(0.5, remaining_time * 0.1) # Use 10% of remaining budget? 
    # Better: If it's the first sequence, use 10s. If last, use whatever is left.
    # Since we don't know which sequence it is, we just use a fraction of the total remaining.
    # If we've used 50% of the budget, remaining is 60s. 10% is 6s.
    # This is safe.
    
    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate initial chains
    num_grow = 5 if n > 50 else 10
    
    initial_chains = []
    for _ in range(num_grow):
        if time.time() - _start_time > _TOTAL_BUDGET - _SAFETY_MARGIN:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain:
            initial_chains.append(chain)
            
    if not initial_chains:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Hybrid Search on each initial chain
    # We will run the search for a portion of the time_limit for each, or just one?
    # Running on multiple is better.
    # Let's run search on the best 2-3 initial chains.
    # Sort initial chains by energy
    scored_chains = [( _energy(sequence, c), c) for c in initial_chains]
    scored_chains.sort()
    
    # Take top 2
    top_chains = [c for e, c in scored_chains[:2]]
    
    for chain in top_chains:
        if time.time() - _start_time > _TOTAL_BUDGET - _SAFETY_MARGIN:
            break
            
        # Allocate half the time_limit to each top chain
        sub_limit = time_limit / len(top_chains)
        
        refined_chain, refined_energy = _hybrid_search(
            sequence, chain, 
            max_iters=5000, 
            rng=rng, 
            time_limit=sub_limit
        )
        
        # Phase 3: Polish
        polished_chain, polished_energy = _greedy_polish(sequence, refined_chain)
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_chain = polished_chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
