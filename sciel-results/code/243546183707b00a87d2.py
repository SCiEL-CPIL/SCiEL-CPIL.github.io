"""HP lattice protein folding: Randomized Greedy Growth + Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin for 120s budget


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Weighted random selection: gain is strong, free space is weak
            weight = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            options.append((weight, site))
        if not options:
            return None
        # Select best option with some randomness
        # Sort by weight descending, pick from top 3 with probability
        options.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(options))
        # Exponential weighting for selection
        weights = [math.exp(options[i][0] / 2.0) for i in range(top_k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        selected_site = options[0][1]
        for i in range(top_k):
            cumulative += weights[i]
            if r <= cumulative:
                selected_site = options[i][1]
                break
        
        occupied[selected_site] = i
        chain.append(selected_site)
    return chain


def _apply_move_reversal(chain, a, b):
    """Apply a reversal move from index a to b. Returns None if invalid."""
    # Check if the reversal creates valid connections
    # Connection a-1 <-> b
    # Connection a <-> b+1
    if a == 0 or b == len(chain) - 1:
        return False
    
    p_prev = chain[a-1]
    p_end = chain[b]
    p_start = chain[a]
    p_next = chain[b+1]
    
    # Distance must be 1
    if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
        return False
    if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
        return False
    
    # Check self-avoidance for the reversed segment
    # The segment [a..b] is reversed.
    # The new positions are the same set of points, so no new collisions with outside.
    # We just need to ensure the internal structure doesn't collide with itself?
    # Reversal preserves the set of points occupied by the segment.
    # So if it was self-avoiding before, it is self-avoiding after, EXCEPT 
    # for the new connections to the outside which we checked above.
    # Wait, does the reversal change the relative positions? No.
    # So self-avoidance is maintained automatically for the internal points.
    
    # Perform the reversal
    chain[a:b+1] = reversed(chain[a:b+1])
    return True


def _apply_move_pivot(chain, i, k):
    """Apply a pivot move at index i, rotating the tail by 90 degrees k times.
    Returns True if valid and applied, False otherwise."""
    if i == 0 or i == len(chain) - 1:
        return False
    
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    prefix_set = set(prefix)
    
    # Rotate tail points relative to pivot
    rotated = []
    valid = True
    for p in tail:
        x = p[0] - pivot[0]
        y = p[1] - pivot[1]
        for _ in range(k):
            x, y = -y, x
        new_p = (pivot[0] + x, pivot[1] + y)
        rotated.append(new_p)
        if new_p in prefix_set:
            valid = False
            break
    
    if not valid:
        return False
    
    # Check connection between i-1 and i+1 (which is now rotated)
    # chain[i-1] must be adjacent to rotated[0]
    p_prev = chain[i-1]
    p_new_start = rotated[0]
    if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
        return False
    
    # Apply
    chain[i+1:] = rotated
    return True


def _simulated_annealing(sequence, chain, time_limit, rng):
    """Simulated Annealing to optimize the chain."""
    n = len(chain)
    if n < 4:
        return _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Temperature schedule
    # Start high, end low
    T_start = 5.0
    T_end = 0.1
    # Estimate number of steps we can take
    # Each step takes ~1ms? Let's assume 1000 steps per second.
    # We will just run until time runs out.
    
    deadline = time.time() + time_limit
    T = T_start
    
    # Cooling rate: we want to reach T_end roughly at the end.
    # But we don't know the exact number of steps.
    # We can use a dynamic cooling rate or just a fixed small rate.
    # Let's use a fixed rate that assumes ~5000 steps.
    alpha = 0.999
    
    while time.time() < deadline:
        # Pick a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Reversal
            if n < 4:
                continue
            a = rng.randint(1, n - 3)
            b = rng.randint(a + 1, n - 2)
            
            # Save state
            old_seg = chain[a:b+1]
            
            if _apply_move_reversal(chain, a, b):
                new_energy = _energy(sequence, chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # Revert
                    chain[a:b+1] = old_seg
            else:
                # Invalid move, revert nothing (we didn't change it)
                pass
        
        else:
            # Pivot
            if n < 4:
                continue
            i = rng.randint(1, n - 2)
            k = rng.randint(1, 3)
            
            # Save state
            old_tail = chain[i+1:]
            
            if _apply_move_pivot(chain, i, k):
                new_energy = _energy(sequence, chain)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # Revert
                    chain[i+1:] = old_tail
            else:
                pass
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            
    # Restore best chain
    chain[:] = best_chain
    return best_energy


def fold(sequence, dimension):
    """Best conformation from randomized greedy growths followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worse than any valid fold
    
    n = len(sequence)
    
    # Phase 1: Generate diverse initial chains
    # We will keep the top K chains based on energy
    top_chains = []
    K = 5
    
    # Time budget allocation:
    # 20% for growth, 80% for SA
    growth_time_limit = _remaining_time() * 0.2
    sa_time_limit = _remaining_time() * 0.8
    
    growth_deadline = time.time() + growth_time_limit
    
    while time.time() < growth_deadline:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        # Insert into top K
        if len(top_chains) < K:
            top_chains.append((energy, chain))
            top_chains.sort(key=lambda x: x[0])
        else:
            if energy < top_chains[-1][0]:
                top_chains[-1] = (energy, chain)
                top_chains.sort(key=lambda x: x[0])
        
        # Early exit if we find a very good solution
        # Heuristic: if energy is very low, we can stop growing
        if energy <= -n // 2:
            break

    if not top_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Simulated Annealing on the best chains
    # Allocate SA time proportionally to the quality of the initial chain?
    # Or just give equal time to the best 2-3?
    # Let's give 50% of SA time to the best, 30% to the second, 20% to the third.
    
    num_sa_runs = min(3, len(top_chains))
    if num_sa_runs == 0:
        return [list(p) for p in top_chains[0][1]]
    
    # Define time fractions
    if num_sa_runs == 1:
        fractions = [1.0]
    elif num_sa_runs == 2:
        fractions = [0.6, 0.4]
    else:
        fractions = [0.5, 0.3, 0.2]
    
    for i in range(num_sa_runs):
        if _remaining_time() < 1.0:
            break
            
        e, c = top_chains[i]
        # Make a copy
        c_copy = c[:]
        
        time_for_this = sa_time_limit * fractions[i]
        # Ensure we don't overshoot global deadline
        time_for_this = min(time_for_this, _remaining_time() - 0.5)
        
        if time_for_this <= 0:
            continue
            
        _simulated_annealing(sequence, c_copy, time_for_this, rng)
        
        new_e = _energy(sequence, c_copy)
        
        if new_e < best_energy:
            best_energy = new_e
            best_chain = c_copy
    
    if best_chain is None:
        best_chain = top_chains[0][1]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
