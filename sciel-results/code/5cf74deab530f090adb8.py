"""HP lattice protein folding using Monte Carlo local search."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _valid_move(sequence, chain, occupied, i, new_site):
    # Check self-avoidance
    if new_site in occupied:
        return False
    
    # Check adjacency with neighbors in chain
    # If i > 0, new_site must be adjacent to chain[i-1]
    # If i < n-1, new_site must be adjacent to chain[i+1]
    # Note: chain[i-1] and chain[i+1] are fixed during the move of i
    
    prev_site = chain[i-1]
    next_site = chain[i+1]
    
    # Distance check
    if abs(new_site[0] - prev_site[0]) + abs(new_site[1] - prev_site[1]) != 1:
        return False
    if abs(new_site[0] - next_site[0]) + abs(new_site[1] - next_site[1]) != 1:
        return False
        
    return True

def _mc_search(sequence, chain, rng, time_limit):
    n = len(sequence)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        # Pick a random index to move (not ends)
        i = rng.randint(1, n - 2)
        old_site = current_chain[i]
        
        # Get neighbors of old_site
        # We want to try moving to a neighbor of old_site
        # But the new site must be adjacent to old_site's neighbors in the chain?
        # No, the new site must be adjacent to chain[i-1] and chain[i+1].
        # So we can just pick a random neighbor of chain[i-1] or chain[i+1]?
        # Or pick a random neighbor of old_site and check validity?
        # Picking a neighbor of old_site is efficient.
        
        moves = list(_neighbours(old_site))
        rng.shuffle(moves)
        
        moved = False
        for new_site in moves:
            if new_site == old_site:
                continue
            if not _valid_move(sequence, current_chain, occupied, i, new_site):
                continue
            
            # Calculate energy delta
            # This is complex to do incrementally without a full re-scan or careful bookkeeping.
            # Given N <= 100, _energy is O(N). 
            # Let's just recompute energy for simplicity and speed (Python overhead might be high, but N is small).
            # Actually, let's try to be a bit smarter or just accept O(N) per step.
            # 100 residues, O(N) energy is 100 ops. 10000 steps is 1M ops. Fast enough in Python?
            # Maybe. Let's optimize _energy slightly by using a set of H positions.
            
            # Update chain
            current_chain[i] = new_site
            del occupied[old_site]
            occupied[new_site] = i
            
            new_energy = _energy(sequence, current_chain)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.71828, -delta / temp):
                # Accept
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Reject
                current_chain[i] = old_site
                occupied[old_site] = i
                del occupied[new_site]
        
        if not moved:
            # Try a different index? Or just continue
            pass
            
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation using MC search with time management."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Initial conformation: simple spiral or line
    # Line is safe but bad energy. Spiral is better.
    # Let's use a simple greedy growth to get a decent start, then MC.
    # Or just start with a random walk?
    # Greedy growth from seed is good.
    
    # We have 180s total for all sequences.
    # Let's assume we are called once per sequence?
    # "fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together."
    # Wait, "120 s CPU budget for all sequences together".
    # So the total time for all 10 calls is 120s?
    # Or is it 120s per process? "sandboxed subprocess... 120s CPU budget for all sequences together" implies the subprocess runs all 10 folds and has 120s total.
    # But the interface is `fold(sequence, dimension)`.
    # Usually in these benchmarks, the harness calls `fold` 10 times.
    # If it's a single subprocess, I can use global state to track time.
    
    global _total_start_time
    global _total_time_limit
    if not hasattr(fold, '_initialized'):
        _total_start_time = time.time()
        _total_time_limit = 110.0  # Safety margin for 120s limit
        fold._initialized = True
    
    elapsed = time.time() - _total_start_time
    time_left = _total_time_limit - elapsed
    
    if time_left <= 0:
        # Return best effort immediately (or a valid trivial solution)
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    # Allocate time for this sequence
    # Heuristic: give more time to longer sequences? Or equal share?
    # 10 sequences. If equal, 11s each.
    # But shorter sequences converge faster.
    # Let's give a proportional share based on length, or just a fixed slice.
    # If we assume 10 calls, and we don't know which one we are, we can't divide perfectly.
    # However, we can just use the remaining time divided by estimated remaining calls?
    # We don't know remaining calls.
    # Safest: use a fraction of remaining time.
    # Or, since we don't know if this is the last call, assume we have 10 calls total.
    # If we track a call counter, we can do better.
    
    if not hasattr(fold, '_call_count'):
        fold._call_count = 0
    fold._call_count += 1
    calls_done = fold._call_count
    # Assume 10 total calls.
    # If calls_done > 10, something is wrong, but cap it.
    assumed_total = 10
    remaining_calls = max(1, assumed_total - calls_done + 1) # Including current
    time_per_call = time_left / remaining_calls
    
    # Cap time per call to avoid one sequence eating everything
    # But if we have time left, we should use it.
    # Let's just use time_per_call.
    
    # Generate initial chain
    rng = random.Random(12345 + len(sequence))
    
    # Greedy growth for start
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, n):
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
            # Dead end, restart growth? Or just break and let MC fix?
            # If we break, chain is incomplete.
            # For MC, we need a full chain.
            # If greedy fails, we need a fallback.
            # Let's try a different start or just use a straight line if we can't grow.
            # But straight line is valid.
            # Let's just extend with a straight line if stuck? No, must be self-avoiding.
            # If greedy fails, we might have a partial chain.
            # Let's just restart greedy a few times.
            break
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        
    if len(chain) < n:
        # Fallback: try to fill the rest with a simple pattern or just use what we have?
        # We MUST return len(sequence) points.
        # If we can't grow, we can't MC on a partial chain easily without completion.
        # Let's just generate a random self-avoiding walk if greedy fails?
        # Or just return a straight line if we fail completely.
        # Actually, greedy rarely fails for N<=100 on 2D lattice if we are lucky.
        # If it fails, let's try a few more times.
        success = False
        for _ in range(50):
            chain2, occ2 = [(0,0)], {(0,0):0}
            for i in range(1, n):
                opts = []
                for site in _neighbours(chain2[-1]):
                    if site in occ2: continue
                    opts.append(site)
                if not opts: break
                site = rng.choice(opts)
                occ2[site] = i
                chain2.append(site)
            if len(chain2) == n:
                chain = chain2
                occupied = occ2
                success = True
                break
        if not success:
            # Last resort: straight line
            chain = [(i, 0) for i in range(n)]
            occupied = {p: i for i, p in enumerate(chain)}
            
    # Run MC
    best_chain = _mc_search(sequence, chain, rng, time_per_call)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
