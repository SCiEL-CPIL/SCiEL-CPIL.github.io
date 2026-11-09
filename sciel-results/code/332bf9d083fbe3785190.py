import time
import random
import math

_START_TIME = time.time()
# The task specifies a 240s CPU budget for all sequences together.
# The evaluator kills after 330s wall-clock.
# We assume ~26 sequences.
# To be safe, we target a total wall-clock time of ~260s, leaving a 70s buffer.
# 260s / 26 sequences ≈ 10s per sequence.
# However, to be conservative and account for Python startup/teardown overhead per call
# (if the evaluator spawns a new process per call, or just function call overhead),
# and to ensure we don't hit the 330s hard limit, we use a dynamic check.
# 
# Strategy:
# 1. Use a global start time.
# 2. In each fold() call, calculate remaining time.
# 3. If this is the first call, allocate a generous per-call budget (e.g., 9.5s).
# 4. As calls accumulate, the remaining global time shrinks.
# 5. If remaining time is insufficient for a full search, return the best found so far (or a quick greedy if none).
# 
# Actually, since we don't know the number of calls in advance inside a single process,
# and "fold is called once per benchmark sequence in a sandboxed subprocess", it's ambiguous
# if it's ONE subprocess for all 26 or 26 subprocesses.
# "240 s CPU budget for all sequences together" strongly implies a single process or a sum of CPU times.
# "Each evaluation must finish within 300 s" implies the total run.
# 
# Let's assume a single process handling all 26 calls.
# We will track elapsed time globally.
# Target total time: 280s (leaves 50s buffer for 330s kill).
# Average per sequence: 280 / 26 ≈ 10.7s.
# We will use a deadline of: _START_TIME + 280.0
# And in each fold(), we check if we are close to that deadline.
# If we are, we reduce the search time for that specific call.

_GLOBAL_DEADLINE = _START_TIME + 280.0

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y-1, z)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, bias=1.0):
    """
    Greedy growth to create an initial self-avoiding walk.
    Incorporates free-space heuristic to avoid traps.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            free = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
                    else:
                        free += 1
            else:
                # Even for P, prefer some free space to avoid dead ends
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1

            # Score: maximize gain, then free space, then noise
            # Using a higher weight for gain ensures we prioritize contacts
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _tail_regrowth_step(sequence, current_chain, rng, T):
    """
    Perform one tail regrowth move.
    Returns (new_chain, new_energy, accepted)
    """
    N = len(current_chain)
    if N < 2:
        return current_chain, _energy(sequence, current_chain), False

    # Pick a cut point
    # Bias towards the middle/early part to allow more restructuring?
    # Uniform is standard. Let's stick to uniform for robustness.
    i = rng.randint(1, N - 2)
    
    head = current_chain[:i]
    
    # Prepare data structures for regrowth
    head_map = {p: idx for idx, p in enumerate(head)}
    occupied = set(head_map.keys())
    
    new_tail = []
    tail_map = {} # pos -> global index
    current_pos = head[-1]
    
    valid = True
    
    for k in range(i, N):
        options = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            
            gain = 0
            free = 0
            if sequence[k] == "H":
                for q in _neighbours(site):
                    if q in head_map:
                        idx_h = head_map[q]
                        if sequence[idx_h] == "H":
                            gain += 1
                    elif q in tail_map:
                        idx_t = tail_map[q]
                        if sequence[idx_t] == "H":
                            gain += 1
                    else:
                        free += 1
            else:
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1

            # Score: maximize gain, then free space, then noise
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
            
        if not options:
            valid = False
            break
            
        best_score, best_site = max(options)
        new_tail.append(best_site)
        tail_map[best_site] = k
        occupied.add(best_site)
        current_pos = best_site
        
    if not valid:
        return current_chain, _energy(sequence, current_chain), False
        
    new_chain = head + new_tail
    new_energy = _energy(sequence, new_chain)
    
    # Metropolis Acceptance
    current_energy = _energy(sequence, current_chain) # Recompute to be safe? Or pass it?
    # For efficiency, we should pass current_energy. But _tail_regrowth_step is a helper.
    # Let's modify the main loop to handle acceptance to avoid recomputing energy twice.
    # Actually, the current implementation in the seed recomputes energy for the new chain.
    # We need the current energy for the delta.
    # Let's assume the caller provides current_energy or we compute it here.
    # To keep the interface clean, let's just return the new chain and energy.
    # The caller will decide on acceptance.
    
    return new_chain, new_energy, True

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    # Determine deadline for this specific call
    # If we are near the global deadline, we must hurry.
    now = time.time()
    remaining_global = _GLOBAL_DEADLINE - now
    
    # If we have very little time left globally, we should just return a quick greedy or best so far.
    # But we don't have "best so far" across calls in this stateless function design.
    # So we must do our best within the time we have.
    
    # Heuristic: If remaining_global < 5s, use a very short local deadline.
    # Otherwise, use a standard per-call budget, capped by remaining_global.
    
    # Standard per-call budget: ~9.5s
    # But if remaining_global is smaller, use that (with a small buffer).
    
    # Let's say we want to finish 1s before the global deadline.
    max_allowed = max(0.1, remaining_global - 1.0)
    
    # Standard budget is 9.5s.
    standard_budget = 9.5
    
    # If we are in the early calls, we can use the full standard budget.
    # If we are in the late calls, we might be forced to use less.
    
    # However, if we use 9.5s for every call, and there are 26 calls, total is 247s.
    # 247s is well within 280s. So we should be fine using a fixed budget per call,
    # as long as the total doesn't exceed the global deadline.
    
    # To be safe, let's use a fixed budget of 9.0s per call.
    # 26 * 9.0 = 234s.
    # This leaves 46s buffer for the 280s target (and 96s for 330s kill).
    # This is safe.
    
    deadline = now + 9.0
    
    # But if the global deadline is closer than this, use the global deadline minus a small buffer.
    if deadline > _GLOBAL_DEADLINE - 0.5:
        deadline = _GLOBAL_DEADLINE - 0.5
    
    # If the deadline is in the past or too short, return a quick greedy
    if deadline <= now:
        chain = _greedy_grow(sequence, random.Random(42))
        if chain:
            return [list(p) for p in chain]
        return [(i, 0, 0) for i in range(N)]
    
    rng = random.Random(42)
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    # Generate multiple diverse seeds
    # Use more seeds if we have more time?
    # 20 seeds is a good start.
    num_seeds = 20
    for _ in range(num_seeds):
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search with Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Perform a tail regrowth step
        new_chain, new_energy, valid = _tail_regrowth_step(sequence, current_chain, rng, T)
        
        if not valid:
            continue
            
        delta_e = new_energy - current_energy
        
        # Metropolis Acceptance
        if delta_e < 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Rejected
            pass
            
    return [list(p) for p in best_chain]
