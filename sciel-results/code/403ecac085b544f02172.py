import time
import random
import math

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _greedy_construct():
    """
    Fast heuristic to get a decent starting point.
    Start with all numbers [0..N], and for each prime p <= N,
    if all p classes are covered, remove the least populated class.
    """
    points = list(range(N + 1))
    primes = _primes(N + 1)
    
    for p in primes:
        if p > len(points):
            break
        
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        if min(counts) > 0:
            min_idx = counts.index(min(counts))
            points = [v for v in points if v % p != min_idx]
            
    return points

def construct():
    start_time = time.time()
    
    # 1. Greedy Initial Solution
    points = _greedy_construct()
    in_set = set(points)
    current_len = len(points)
    best_len = current_len
    best_points = points[:]
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # We need to track the "state" efficiently.
    # State: set of points.
    # Objective: Maximize size, subject to admissibility.
    # Admissibility: For all p <= len(S), exists r s.t. count_p(r) == 0.
    
    # Let's use a "soft" constraint approach or Simulated Annealing on the set.
    # Moves:
    # 1. Insert a random candidate from [0, N] \ S.
    # 2. Remove a random element from S.
    # 3. Swap: Remove x from S, Insert y from [0, N] \ S.
    
    # To evaluate admissibility quickly, we maintain counts for relevant primes.
    # Relevant primes: p <= len(S) + 1 (since adding one might make p = len(S)+1 relevant).
    
    # Initialize counts for all primes in `primes` (up to N+1)
    # This is O(N * pi(N)) ~ 3M ops, acceptable.
    counts = {}
    missed = {}
    
    for p in primes:
        counts[p] = [0] * p
        for v in points:
            counts[p][v % p] += 1
        missed[p] = set()
        for r in range(p):
            if counts[p][r] == 0:
                missed[p].add(r)
                
    def is_valid():
        # Check if current set is admissible
        for p in primes:
            if p > current_len:
                break
            if len(missed[p]) == 0:
                return False
        return True
    
    # Simulated Annealing
    # We want to maximize size.
    # Let's define a "cost" function that penalizes invalidity heavily but allows exploration.
    # Actually, standard SA on combinatorial problems with hard constraints is tricky.
    # Alternative: Random Restart Local Search with more aggressive moves.
    
    # Let's try a "Swap" based local search with random restarts.
    # Current best is from greedy.
    
    # Candidates for insertion
    candidates = [i for i in range(N + 1) if i not in in_set]
    
    # We will run multiple passes of local search.
    # In each pass, we try to improve the set.
    # If stuck, we might "shake" the set by removing some elements and re-optimizing, 
    # or just start a new random restart from a perturbed state.
    
    # Given the time budget, let's do a robust local search with randomization.
    # The current seed does simple insertion. 
    # A key insight: sometimes removing a "bad" element allows adding two "good" ones.
    # So we should allow removals.
    
    improved = True
    restarts = 0
    max_restarts = 100 # Limit restarts to avoid infinite loops if time is tight
    
    while improved and (time.time() - start_time < DEADLINE - 5) and restarts < max_restarts:
        improved = False
        restarts += 1
        
        # Shuffle candidates to randomize order
        random.shuffle(candidates)
        
        # 1. Try to insert
        for cand in candidates:
            if cand in in_set:
                continue
            
            new_len = current_len + 1
            can_add = True
            
            # Check primes p <= new_len
            for p in primes:
                if p > new_len:
                    break
                
                m_set = missed[p]
                m_size = len(m_set)
                
                if m_size == 0:
                    can_add = False
                    break
                elif m_size == 1:
                    if cand % p in m_set:
                        can_add = False
                        break
            
            if can_add:
                # Add cand
                points.append(cand)
                in_set.add(cand)
                current_len = new_len
                improved = True
                best_len = max(best_len, current_len)
                if current_len > best_len:
                    best_points = points[:]
                    
                # Update counts and missed
                for p in primes:
                    r = cand % p
                    counts[p][r] += 1
                    if r in missed[p]:
                        missed[p].discard(r)
                
                # Remove from candidates list (or mark as used)
                # Since we iterate over a shuffled list, we can just skip if in_set
                # But candidates list contains indices. 
                # It's better to maintain a separate list of available candidates or check in_set.
                # Checking in_set is O(1).
                
                # Optimization: If we added a lot, we might have new primes to consider?
                # No, primes are fixed. But the constraint p <= len(S) becomes stricter as len(S) grows.
                
        # 2. If no insertions were made in this pass, try a "swap" move or restart
        if not improved:
            # Try to find a swap: remove one, add one (net 0 change) to reach a different local optimum?
            # Or just remove a random element to "unlock" potential insertions?
            # Let's try a "perturbation": remove k random elements, then try to re-insert.
            
            if current_len > 0:
                # Remove 1-3 random elements
                num_removals = random.randint(1, 3)
                removed = []
                for _ in range(num_removals):
                    if not points:
                        break
                    idx = random.randint(0, len(points) - 1)
                    val = points.pop(idx)
                    in_set.remove(val)
                    removed.append(val)
                    current_len -= 1
                    
                    # Update counts
                    for p in primes:
                        r = val % p
                        counts[p][r] -= 1
                        if counts[p][r] == 0:
                            missed[p].add(r)
                
                # Add removed elements back to candidates pool?
                # They are not in in_set, so they are effectively candidates.
                # But our `candidates` list might not contain them if it was pre-computed.
                # Let's just rebuild candidates or accept that we iterate over the static list 
                # and check `if cand in in_set`.
                # Wait, if we removed them, they are NOT in in_set. 
                # But they might not be in the `candidates` list if that list was 
                # [i for i in range(N+1) if i not in in_set_initial].
                # So we need to ensure we consider them.
                
                # Simplest: Just loop through ALL possible candidates [0..N] and check if not in set.
                # This is O(N) per pass, which is 5000. 
                # Number of passes? If we do 100 passes, 500k checks. Fast.
                
                # Let's do a full scan pass for insertion.
                for cand in range(N + 1):
                    if cand in in_set:
                        continue
                    
                    new_len = current_len + 1
                    can_add = True
                    
                    for p in primes:
                        if p > new_len:
                            break
                        
                        m_set = missed[p]
                        m_size = len(m_set)
                        
                        if m_size == 0:
                            can_add = False
                            break
                        elif m_size == 1:
                            if cand % p in m_set:
                                can_add = False
                                break
                    
                    if can_add:
                        points.append(cand)
                        in_set.add(cand)
                        current_len = new_len
                        improved = True
                        if current_len > best_len:
                            best_len = current_len
                            best_points = points[:]
                        
                        for p in primes:
                            r = cand % p
                            counts[p][r] += 1
                            if r in missed[p]:
                                missed[p].discard(r)
                
                # If we still didn't improve, we are likely stuck in a local optimum.
                # To escape, we could restart from a completely random valid set?
                # Or just accept the current best.
                # Given the time, restarting from scratch (greedy) might be safer 
                # than complex perturbations if we don't have a good model.
                
                # Let's try a different strategy if stuck: 
                # Random Restart with Greedy + Local Search
                # If we've tried several restarts, we might just return best.
                
                # For now, if no improvement after perturbation, break to next restart?
                # Actually, the loop condition `while improved` will break if `improved` is False.
                # But we set `improved = False` at the start of the loop.
                # If the perturbation+insertion didn't find anything, `improved` stays False.
                # The loop will terminate.
                # This means we only do ONE perturbation attempt per "stuck" state?
                # No, the `while improved` loop breaks.
                
                # We need a structure that allows multiple attempts.
                # Let's change the structure:
                # Outer loop: Restarts
                # Inner loop: Local Search until stuck
                
                pass # The logic above handles one "shake" then breaks if no improvement.
                # This might be too conservative. Let's allow the outer loop to continue 
                # by not breaking immediately, but the `while improved` condition will stop it.
                
                # To allow multiple shakes, we should not use `while improved` for the main loop, 
                # but rather a time-based loop.
                
                break # Break out of the "try to insert" phase, go to next restart iteration?
                # Wait, the code structure is:
                # while improved:
                #   improved = False
                #   shuffle candidates
                #   for cand in candidates: ...
                #   if not improved:
                #      ... perturbation ...
                #      break # This breaks the for loop? No, the 'break' above is inside the 'if not improved' block.
                #      It breaks the `for cand in candidates` loop? No, that loop already finished.
                #      It breaks the `while improved` loop? No, `break` inside `if` inside `while` breaks the `while`.
                
                # So if we get stuck, we do one perturbation, and then exit the main loop.
                # This is a single-shot perturbation.
                
                # Let's restructure to allow multiple perturbations within the time budget.
                
                # I will rewrite the main loop to be time-based and restart-based.

    # Let's restart the logic with a cleaner time-based loop for multiple restarts.
    
    # Reset time check
    time_remaining = DEADLINE - (time.time() - start_time)
    
    # We already have a good solution from the first pass.
    # Let's do a few more restarts if time permits.
    
    # Actually, the previous loop structure was flawed for multiple restarts.
    # Let's implement a clean Simulated Annealing / Random Restart structure.
    
    # Since I already ran the greedy and one pass of local search, 
    # and potentially one perturbation, let's just ensure we return the best found.
    
    # However, the code above has a `break` that exits the `while improved` loop.
    # So it only does ONE full local search pass, then ONE perturbation, then stops.
    # This is not utilizing the 300s budget well.
    
    # Let's replace the main loop with a proper time-based loop.
    
    # I will re-implement the search part to be robust.
    
    # The code above is already written. I need to fix the loop structure.
    
    # Let's discard the flawed loop and write a correct one.
    
    # --- REIMPLEMENTING MAIN SEARCH LOOP ---
    
    # Current state: points, in_set, counts, missed, current_len
    # Best state: best_points, best_len
    
    # We will run multiple "episodes".
    # Each episode:
    # 1. Try to insert any valid candidate (full scan or shuffled).
    # 2. If stuck, perform a perturbation (remove k elements).
    # 3. Try to insert again.
    # 4. Repeat until time runs out.
    
    # To avoid infinite loops, we track how many times we've perturbed.
    
    perturbation_count = 0
    max_perturbations = 50 # Arbitrary limit, time will stop it anyway.
    
    while (time.time() - start_time < DEADLINE - 5) and perturbation_count < max_perturbations:
        
        # Phase 1: Greedy Insertion
        # Scan all candidates to find any that can be added.
        # This is O(N * pi(N)) per scan, which is ~3M ops.
        # 3M ops is very fast (<0.1s in Python with optimizations, maybe 0.5s naive).
        # We can do many of these.
        
        any_added = False
        
        # To make it stochastic, we can shuffle the range? 
        # Or just iterate 0..N. 
        # Shuffling 5000 items is fast.
        
        # Let's iterate through a shuffled list of all possible values [0..N]
        # and check if they are not in set and can be added.
        
        # Optimization: Only check if not in set first.
        
        # We can maintain a list of "active" candidates? 
        # Or just iterate range(N+1) and check `if i in in_set`.
        
        for i in range(N + 1):
            if i in in_set:
                continue
            
            new_len = current_len + 1
            can_add = True
            
            for p in primes:
                if p > new_len:
                    break
                m_size = len(missed[p])
                if m_size == 0:
                    can_add = False
                    break
                elif m_size == 1:
                    if i % p in missed[p]:
                        can_add = False
                        break
            
            if can_add:
                # Add
                points.append(i)
                in_set.add(i)
                current_len = new_len
                any_added = True
                
                if current_len > best_len:
                    best_len = current_len
                    best_points = points[:]
                
                for p in primes:
                    r = i % p
                    counts[p][r] += 1
                    if r in missed[p]:
                        missed[p].discard(r)
                
                # Continue scanning to see if more can be added
                # Note: Adding i might make other candidates invalid? 
                # No, adding an element only fills more residue classes.
                # It can only REDUCE the set of missed residues.
                # So it can only make it HARDER to add others.
                # But it doesn't invalidate previously added ones.
                # So we can keep scanning.
        
        if any_added:
            # We made progress. Reset perturbation count? 
            # Or just continue.
            # If we added many, we might have hit a new local optimum.
            # Let's continue to next iteration to see if more can be added.
            # But if we added nothing in the next iteration, we perturb.
            continue
        
        # Phase 2: Stuck. Perturb.
        if current_len == 0:
            break
        
        # Remove 1-3 random elements
        num_removals = random.randint(1, 3)
        for _ in range(num_removals):
            if not points:
                break
            idx = random.randint(0, len(points) - 1)
            val = points.pop(idx)
            in_set.remove(val)
            current_len -= 1
            
            for p in primes:
                r = val % p
                counts[p][r] -= 1
                if counts[p][r] == 0:
                    missed[p].add(r)
        
        perturbation_count += 1
        
        # After perturbation, loop back to Phase 1 to try to fill again.
        
    return best_points

# EVOLVE-BLOCK-END
