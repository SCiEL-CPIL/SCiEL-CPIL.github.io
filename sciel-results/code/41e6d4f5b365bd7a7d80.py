import time
import random
import sys

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI].
    Returns None if A is empty.
    """
    if not A_set:
        return None
    A_sorted = sorted(A_set)
    n = len(A_sorted)
    counts = [0] * _RANGE
    
    for i in range(n):
        a = A_sorted[i]
        for j in range(i, n):
            b = A_sorted[j]
            s = a + b
            if _LO <= s <= _HI:
                counts[s - _LO] += 1
    return counts

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return max(counts)

def _greedy_peel(A_init, rng=None, time_limit=5.0):
    """
    Greedily remove elements from A_init to minimize peak while maintaining validity.
    If rng is provided, it uses stochastic selection among best candidates to escape local minima.
    """
    A = set(A_init)
    counts = _get_counts(A)
    
    if not _is_valid(counts):
        return A, _get_peak(counts)

    current_peak = _get_peak(counts)
    best_A = A.copy()
    best_peak = current_peak
    
    start_time = time.time()
    
    # We iterate until we can't remove anything or time runs out
    # To avoid infinite loops if we only swap size without reducing peak, we track size
    # But here we only remove. So size strictly decreases. Max 129 iterations.
    
    while time.time() - start_time < time_limit:
        # Find candidates for removal
        A_list = sorted(A)
        
        best_candidate = None
        best_candidate_peak = current_peak
        best_candidate_size = len(A)
        
        # List of all valid removals that either reduce peak or keep peak same (to allow further exploration)
        # Actually, we want to prioritize peak reduction.
        # Let's collect all valid removals.
        valid_removals = []
        
        for x in A_list:
            new_A = A.copy()
            new_A.discard(x)
            if not new_A:
                continue
            
            new_counts = _get_counts(new_A)
            if not _is_valid(new_counts):
                continue
            
            new_peak = _get_peak(new_counts)
            valid_removals.append((new_peak, new_A, x))
            
        if not valid_removals:
            break
            
        # Sort by peak (ascending), then by size (ascending, though size is same for all single removals)
        # We want the lowest peak.
        valid_removals.sort(key=lambda item: item[0])
        
        # If stochastic, pick from the top few? Or just the best?
        # The inspiration used the best. Let's try picking the best, but if there are ties, pick randomly?
        # Or, more aggressively: if we have time, try a few random valid removals and see which leads to a better final peak?
        # That's expensive. Let's stick to the greedy best, but if there are multiple with the same peak, pick randomly to diversify paths in restarts.
        
        min_peak = valid_removals[0][0]
        best_options = [item for item in valid_removals if item[0] == min_peak]
        
        if rng:
            chosen = rng.choice(best_options)
        else:
            chosen = best_options[0]
            
        new_peak, new_A, removed_x = chosen
        
        # Accept the move
        A = new_A
        counts = _get_counts(A)
        current_peak = new_peak
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = A.copy()
            
        # If peak didn't decrease, we might be stuck in a plateau.
        # But since we removed an element, size decreased.
        # We continue to see if we can reduce peak in subsequent steps.
        
    return best_A, best_peak

def _local_search(A_init, time_limit, rng):
    """
    Perform local search starting from A_init.
    Returns the best set found and its peak.
    """
    A = set(A_init)
    counts = _get_counts(A)
    
    if not _is_valid(counts):
        return None, float('inf')

    current_peak = _get_peak(counts)
    best_A = A.copy()
    best_peak = current_peak
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 20 # Reduced to allow more restarts or faster cycling
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Generate moves
        in_A = list(A)
        not_in_A = [x for x in range(_N + 1) if x not in A]
        
        moves = []
        
        # 1. Swaps: Remove x, Add y
        # Limit number of swaps to keep it fast
        num_swaps = min(30, len(in_A) * len(not_in_A))
        if num_swaps > 0:
            for _ in range(num_swaps):
                x = in_A[rng.randint(0, len(in_A) - 1)]
                y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
                moves.append(('swap', x, y))
        
        # 2. Removes: Remove x
        # Only if it keeps validity.
        num_rems = min(10, len(in_A))
        if num_rems > 0:
            for _ in range(num_rems):
                x = in_A[rng.randint(0, len(in_A) - 1)]
                moves.append(('remove', x, None))
                
        # 3. Adds: Add y
        # Helps escape local minima where we need to grow to shrink better later?
        # Or just to find better configurations.
        num_adds = min(5, len(not_in_A))
        if num_adds > 0:
            for _ in range(num_adds):
                y = not_in_A[rng.randint(0, len(not_in_A) - 1)]
                moves.append(('add', None, y))
        
        rng.shuffle(moves)
        
        for move in moves:
            if time.time() - start_time > time_limit:
                break
            
            new_A = A.copy()
            
            if move[0] == 'swap':
                _, x, y = move
                new_A.discard(x)
                new_A.add(y)
            elif move[0] == 'add':
                _, _, y = move
                new_A.add(y)
            elif move[0] == 'remove':
                _, x, _ = move
                new_A.discard(x)
            
            if not new_A:
                continue
                
            new_counts = _get_counts(new_A)
            if not _is_valid(new_counts):
                continue
                
            new_peak = _get_peak(new_counts)
            
            if new_peak < current_peak:
                A = new_A
                counts = new_counts
                current_peak = new_peak
                improved = True
                no_improve_count = 0
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = A.copy()
                
                # Restart move generation with new state to exploit local structure
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
    return best_A, best_peak

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 90s limit, 5s margin
    
    rng = random.Random(42)
    
    best_A_global = None
    best_peak_global = float('inf')
    
    # 1. Primary Strategy: Greedy Peeling from Full Interval
    # This is the strongest known approach from inspiration 1.
    A_full = set(range(0, _N + 1))
    
    # Run greedy peeling with a time limit
    # We can run it multiple times with different rng seeds to diversify?
    # The greedy peel is deterministic if we always pick the best.
    # If we use rng in _greedy_peel, we can get different results.
    
    # Let's run one deterministic peel first.
    A_peel, peak_peel = _greedy_peel(A_full, rng=None, time_limit=10.0)
    
    if _is_valid(_get_counts(A_peel)):
        best_A_global = A_peel.copy()
        best_peak_global = peak_peel
    
    # 2. Refinement: Local Search on the Peeled Set
    # The peeled set is a good local minimum for removals.
    # Swaps can escape this.
    
    if time.time() < DEADLINE - 5:
        A_curr = best_A_global.copy()
        # Run local search on the best peeled set
        # Give it a generous time budget
        t_limit = min(20.0, DEADLINE - time.time() - 2)
        
        A_res, peak_res = _local_search(A_curr, t_limit, rng)
        
        if A_res is not None and peak_res < best_peak_global:
            best_peak_global = peak_res
            best_A_global = A_res
    
    # 3. Diversity: Try other starting points
    # If we have time, try peeling from other structures or random sets.
    
    if time.time() < DEADLINE - 10:
        # Try a few random sets + peel + local search
        # This might find a globally better structure that full interval peel misses.
        
        # Candidate 1: Random set of size ~40-60
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
                
            size = rng.randint(30, 50)
            A_rand = set(rng.sample(range(0, _N + 1), size))
            A_rand.add(0)
            A_rand.add(128)
            
            # Repair if invalid? Greedy peel assumes valid start.
            # Let's check validity first.
            counts = _get_counts(A_rand)
            if not _is_valid(counts):
                # Quick repair: add random elements until valid?
                # Or skip?
                # Skipping is safer to avoid wasting time on repair.
                # But random sets of size 40+ are often valid.
                # Let's try a few more sizes if invalid.
                continue
                
            # Peel it
            A_p, peak_p = _greedy_peel(A_rand, rng=rng, time_limit=5.0)
            
            if _is_valid(_get_counts(A_p)):
                if peak_p < best_peak_global:
                    best_peak_global = peak_p
                    best_A_global = A_p.copy()
                    
                # Local search on it
                if time.time() < DEADLINE - 5:
                    A_ls, peak_ls = _local_search(A_p, 5.0, rng)
                    if A_ls is not None and peak_ls < best_peak_global:
                        best_peak_global = peak_ls
                        best_A_global = A_ls.copy()
                        
    # 4. Final Polish: If we have time, run one more local search on the global best
    if time.time() < DEADLINE - 2 and best_A_global is not None:
        A_final, peak_final = _local_search(best_A_global, DEADLINE - time.time() - 1, rng)
        if A_final is not None and peak_final < best_peak_global:
            best_peak_global = peak_final
            best_A_global = A_final
    
    if best_A_global is None:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_A_global)
