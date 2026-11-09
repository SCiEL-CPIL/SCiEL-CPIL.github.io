"""Rewritten program for HP lattice protein folding with Backtracking Search and Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Pre-compute neighbor offsets for 3D cubic lattice
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _energy(sequence, chain):
    """Calculate energy of a chain."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            # Check 6 neighbors
            for nx, ny, nz in [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]:
                j = where.get((nx, ny, nz))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, prefix_chain, start_idx, occupied, T):
    """
    Grow the chain from start_idx to the end using a heuristic with noise.
    Returns None if stuck.
    """
    n = len(sequence)
    chain = list(prefix_chain)
    # occupied is a dict pos -> idx, already contains prefix
    # We need to copy occupied to avoid modifying the original during exploration?
    # Actually, for SA, we want to see if the move is good. 
    # But regrowth is stochastic. 
    # Standard approach: 
    # 1. Remove tail from current state.
    # 2. Regrow tail.
    # 3. If regrowth fails, revert.
    # 4. If regrowth succeeds, calculate energy.
    
    # To speed up, we can pass a copy of occupied? 
    # Copying a dict of size N is O(N). N is up to 136. 
    # Number of steps ~ 10^5. 10^5 * 136 ~ 1.3 * 10^7 ops. Acceptable.
    
    occ = dict(occupied)
    
    for i in range(start_idx, n):
        last_pos = chain[-1]
        x, y, z = last_pos
        
        candidates = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        valid_candidates = []
        for site in candidates:
            if site in occ:
                continue
            
            # Calculate heuristic score
            # Gain: number of new H-H contacts
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,sz+1), (sx,sy,sz-1)]:
                    j = occ.get((nx, ny, nz))
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Free space: number of unoccupied neighbors
            free = 0
            sx, sy, sz = site
            for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,sz+1), (sx,sy,sz-1)]:
                if (nx, ny, nz) not in occ:
                    free += 1
            
            # Heuristic: Gain is dominant. Free space prevents trapping.
            # Noise: Add random noise scaled by T to allow exploration.
            # If T is high, noise is large, allowing bad moves.
            # If T is low, noise is small, greedy choice dominates.
            
            # Base score
            score = gain * 10.0 + free * 0.5
            
            # Add noise
            # We want to select based on score + noise.
            # Or use Boltzmann distribution? 
            # Simpler: score + rng.gauss(0, T)
            # If T is high, we might pick a low gain move.
            
            noisy_score = score + rng.gauss(0, T)
            valid_candidates.append((noisy_score, site))
        
        if not valid_candidates:
            return None
        
        # Select best noisy score
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = valid_candidates[0][1]
        
        chain.append(best_site)
        occ[best_site] = i
    
    return chain

def _backtrack_search(sequence, dimension, rng, deadline):
    """
    Main search loop using Backtracking (Regrowth) moves with Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial growth
    # Try a few greedy starts to get a good initial chain
    best_chain = None
    best_energy = 0
    
    # Simple initial greedy growth (T=0 essentially, or very low)
    # Actually, let's just run the main loop starting from a random or greedy start.
    # To ensure we have a valid chain, we start with a greedy one.
    
    # Greedy start
    for _ in range(10):
        chain = _greedy_grow(sequence, dimension, rng, [], 0, {}, 0.0)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
        if best_chain:
            break
            
    if not best_chain:
        # Fallback: linear chain (should not happen for valid inputs usually, but safe)
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    current_chain = best_chain
    current_energy = best_energy
    
    # Maintain occupied set for current_chain
    # We rebuild it if we accept a move, or we can maintain it.
    # Since we are regrowing the tail, we can just rebuild the occupied set for the prefix?
    # No, we need the full occupied set to check validity during regrowth.
    # But _greedy_grow builds its own `occ` from the passed `occupied`.
    # So we need to pass the `occupied` of the *prefix* only? 
    # Wait, _greedy_grow takes `occupied` which should contain the prefix.
    # It adds the tail to a *copy* of that occupied set.
    # So we need to provide the occupied set of the prefix `current_chain[:k]`.
    
    # To make this fast, we can pre-compute occupied sets for all prefixes? 
    # No, that's O(N^2) memory and time.
    
    # Alternative: 
    # 1. Pick k.
    # 2. Create prefix = current_chain[:k]
    # 3. Create occupied_prefix = {p: i for i, p in enumerate(prefix)}
    # 4. Call _greedy_grow(..., prefix, k, occupied_prefix, T)
    # 5. If result, calculate energy.
    
    # This is O(N) to build occupied_prefix. 
    # Total cost per step: O(N) for prefix map + O(N) for regrowth + O(N) for energy.
    # ~ 3N operations.
    # N=136. 400 ops per step.
    # 10^6 steps -> 4*10^8 ops. Might be too slow in Python?
    # Let's optimize.
    
    # We can maintain a global `pos_to_idx` for current_chain.
    # When we pick k, we need to "remove" k..N-1 from pos_to_idx?
    # Or just pass the subset?
    
    # Let's stick to the simpler approach first and see if it's fast enough.
    # If not, we can optimize by maintaining a list of positions and a dict, 
    # and updating the dict when we accept a move.
    
    # Let's optimize the energy calculation in _greedy_grow? 
    # No, _greedy_grow is just generating the chain.
    
    # Let's try to run as many steps as possible.
    # Estimate: 1000 steps/sec? 
    # If we have 300s, 300,000 steps.
    # 300,000 * 400 ops = 1.2 * 10^8 ops. Should be okay in 300s? 
    # Python is slow. 10^8 simple ops can take 10-20s. 
    # So 300,000 steps might take 30-60s. 
    # We have 300s. So we can do ~1-2 million steps.
    
    # SA Parameters
    T0 = 2.0
    T_min = 0.01
    # We want T to drop from T0 to T_min over the course of the run.
    # If we run M steps, rate = (T_min/T0)^(1/M)
    # But M is unknown. 
    # Use a fixed rate per step? 
    # Or cool based on time?
    # Let's use a fixed rate that drops T significantly.
    # e.g. T *= 0.999 every step?
    # After 1000 steps, T = 2 * 0.999^1000 = 2 * 0.36 = 0.72.
    # After 4000 steps, T = 2 * 0.999^4000 = 2 * 0.018 = 0.036.
    # This seems reasonable.
    
    cooling_rate = 0.999
    T = T0
    
    start_time = time.time()
    step = 0
    
    # Pre-calculate H positions for energy? 
    # _energy rebuilds the map every time. That's O(N^2) effectively if not careful?
    # _energy:
    # where = {p: i ...} -> O(N)
    # for i in N: check 6 neighbors -> O(1) lookup.
    # So _energy is O(N).
    
    while time.time() < deadline - 5.0: # Safety margin
        step += 1
        T = max(T * cooling_rate, T_min)
        
        # Pick cut point k
        # Avoid too small k (local moves) or too large k (global moves).
        # Uniform random from 1 to N-1?
        # Or biased towards middle?
        # Let's try uniform first.
        if n <= 2:
            break
        k = rng.randint(1, n - 1)
        
        prefix = current_chain[:k]
        # Build occupied map for prefix
        occ_prefix = {p: i for i, p in enumerate(prefix)}
        
        new_chain = _greedy_grow(sequence, dimension, rng, prefix, k, occ_prefix, T)
        
        if new_chain is None:
            # Regrowth failed. 
            # In SA, we usually reject this move.
            # Or we can try a different k? 
            # Let's just continue.
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance
        if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
        else:
            # Reject, keep current_chain
            pass
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Backtracking Search with Simulated Annealing."""
    start_time = time.time()
    # Deadline: 320s wall clock limit. 
    # Leave 5s for safety/overhead.
    deadline = start_time + 315.0
    
    rng = random.Random(12345 + len(sequence))
    
    result = _backtrack_search(sequence, dimension, rng, deadline)
    
    return [list(p) for p in result]
# EVOLVE-BLOCK-END
