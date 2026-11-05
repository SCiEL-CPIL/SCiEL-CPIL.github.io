import time
import random
import numpy as np
import itertools

# Constants for the specific problem instance K_3(7,1)
Q = 3
N = 7
R = 1
TOTAL_WORDS = Q ** N  # 2187

# Precompute neighbors for all possible words in {0,1,2}^7
# For a word w, its neighbors at distance <= 1 are:
# - w itself
# - w with one position changed to one of the other Q-1 values
# There are 1 + N * (Q-1) = 1 + 7 * 2 = 15 neighbors per word.

# We represent words as integers in [0, 3^7)
# We need a mapping from integer index to list of neighbors (indices)

def _get_neighbors(idx):
    """
    Return list of indices of all words within Hamming distance 1 from word with index idx.
    """
    # Convert index to base-3 representation
    # idx = d0 * 3^6 + d1 * 3^5 + ... + d6
    # We use little-endian for easier computation: idx = d6 * 3^0 + ... ? 
    # Let's stick to standard: pos 0 is most significant.
    # Actually, for neighbor calculation, it's easier to work with the tuple.
    
    # Convert idx to tuple of digits (length N)
    digits = []
    temp = idx
    for _ in range(N):
        digits.append(temp % Q)
        temp //= Q
    digits.reverse() # Now digits[0] is most significant
    
    neighbors = [idx]
    
    # Generate neighbors by changing one digit
    for i in range(N):
        original_digit = digits[i]
        for new_digit in range(Q):
            if new_digit == original_digit:
                continue
            # Create new digits
            new_digits = digits.copy()
            new_digits[i] = new_digit
            # Convert back to index
            new_idx = 0
            for d in new_digits:
                new_idx = new_idx * Q + d
            neighbors.append(new_idx)
            
    return neighbors

# Precompute neighbors for all 2187 words
# This is a memory trade-off: 2187 * 15 ints is small.
_NEIGHBORS = [None] * TOTAL_WORDS
for i in range(TOTAL_WORDS):
    _NEIGHBORS[i] = _get_neighbors(i)

def _idx_to_tuple(idx):
    """Convert integer index to tuple of digits (length N)."""
    digits = []
    temp = idx
    for _ in range(N):
        digits.append(temp % Q)
        temp //= Q
    digits.reverse()
    return tuple(digits)

def _tuple_to_idx(tup):
    """Convert tuple of digits to integer index."""
    idx = 0
    for d in tup:
        idx = idx * Q + d
    return idx

def _initial_tetracode():
    """
    Construct the known 186-word covering code (or close to it) using Tetracode structure.
    The Tetracode is a code of length 4 over {0,1,2} with 9 words, covering radius 1.
    The direct sum with a code of length 3 (all words) gives 9 * 27 = 243 words.
    However, we can do better.
    
    Actually, the known best code has 186 words.
    A common construction is based on the Tetracode.
    Let's start with a random valid solution or the tetracode direct sum and let SA optimize.
    
    For initialization, let's use a greedy construction to get a valid starting point.
    """
    # Greedy construction:
    # Start with empty set.
    # While there are uncovered words:
    #   Pick a random uncovered word w.
    #   Add a codeword c that covers w and maximizes the number of newly covered words.
    
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    codewords = []
    
    # Precompute coverage impact for each potential codeword? 
    # Since we add one by one, we can just check.
    
    # To speed up, we can just pick a random uncovered word and add a random neighbor, 
    # but greedy is better for a good start.
    
    # Let's do a simple greedy:
    # 1. Find an uncovered word u.
    # 2. For each potential codeword c in neighbors(u), calculate how many *new* words it covers.
    # 3. Add the best c.
    
    # This is O(TotalWords * NumCandidates * 15).
    # 2187 * 15 * 15 ~ 500k ops per step. Max 200 steps. 100M ops. Might be slow in Python.
    
    # Faster init: Just add random codewords until covered?
    # Or use the known 186 structure?
    
    # Let's try a slightly smarter random init:
    # Add 200 random codewords.
    # Then check coverage. If not covered, add more.
    
    # Actually, let's just use the Tetracode direct sum as a baseline valid solution.
    # Tetracode words:
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a+b)%3, (a+2*b)%3])
            
    # Direct sum with all 27 words of length 3
    init_cw = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            w = h + list(tail)
            init_cw.append(_tuple_to_idx(tuple(w)))
            
    # This gives 9 * 27 = 243 words.
    # We can reduce this with local search.
    return init_cw

def _check_coverage(codewords):
    """
    Return True if codewords cover all words.
    """
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in codewords:
        for nb in _NEIGHBORS[c]:
            cov[nb] += 1
    return np.all(cov >= 1)

def _count_uncovered(codewords):
    """
    Return number of uncovered words.
    """
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in codewords:
        for nb in _NEIGHBORS[c]:
            cov[nb] += 1
    return np.sum(cov == 0)

def _simulated_annealing(start_cw, time_limit, rng):
    """
    Run Simulated Annealing starting from start_cw.
    Returns the best codeword list found.
    """
    t0 = time.time()
    codewords = list(start_cw)
    best_cw = list(codewords)
    best_len = len(codewords)
    
    # Initial coverage calculation
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in codewords:
        for nb in _NEIGHBORS[c]:
            cov[nb] += 1
    
    # Initial state should be valid if start_cw is valid.
    # If not, we force validity first.
    if not np.all(cov >= 1):
        # Greedily add codewords to fix coverage
        while not np.all(cov >= 1):
            # Find an uncovered word
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            u = int(uncovered[rng.integers(0, len(uncovered))])
            # Pick a random neighbor of u to add
            candidates = _NEIGHBORS[u]
            c = int(candidates[rng.integers(0, len(candidates))])
            if c not in codewords:
                codewords.append(c)
                for nb in _NEIGHBORS[c]:
                    cov[nb] += 1
        best_cw = list(codewords)
        best_len = len(codewords)

    # SA Parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # We track the set of codewords for fast membership check
    cw_set = set(codewords)
    
    while time.time() - t0 < time_limit and T > min_T:
        # Pick a random move type
        # 1. Remove a codeword
        # 2. Add a codeword
        # 3. Move a codeword (replace)
        
        move_type = rng.integers(0, 3)
        
        if move_type == 0:
            # Try to remove a random codeword
            if len(codewords) == 0:
                T *= cooling_rate
                continue
                
            i = int(rng.integers(0, len(codewords)))
            c = codewords[i]
            
            # Check if removing c leaves any word uncovered
            # We need to check if cov[nb] > 1 for all nb in neighbors(c)
            # If cov[nb] == 1 for any nb, removing c makes it uncovered.
            
            removable = True
            for nb in _NEIGHBORS[c]:
                if cov[nb] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                codewords.pop(i)
                cw_set.remove(c)
                for nb in _NEIGHBORS[c]:
                    cov[nb] -= 1
                # Improvement: length decreased
                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best_cw = list(codewords)
            else:
                # Not removable. Could accept with probability exp(-delta/T) if we allowed invalid states?
                # But we want to keep validity or handle it carefully.
                # Let's stick to valid states only for simplicity and robustness.
                # Or, we can allow temporary invalid states if we add a penalty.
                # Given the time limit, let's try to stay valid.
                pass

        elif move_type == 1:
            # Try to add a codeword
            # This is usually bad (increases length), but might enable future removals.
            # Or, we can view it as "Swap": Remove one, Add one.
            # Let's implement Swap instead of pure Add.
            # If we add, we must remove later.
            # Let's do: Pick a random codeword c_old, pick a random new codeword c_new.
            # If c_new is not in set, try to replace c_old with c_new.
            
            if len(codewords) == 0:
                T *= cooling_rate
                continue
                
            i = int(rng.integers(0, len(codewords)))
            c_old = codewords[i]
            
            # Generate a random neighbor c_new
            # Pick a random position and change value
            # Or pick a random word from the whole space?
            # Picking a random word is too sparse.
            # Let's pick a word within distance 2 of c_old?
            # Or just a random word in the space.
            
            # Better: Pick a random word from the space.
            c_new = int(rng.integers(0, TOTAL_WORDS))
            
            if c_new == c_old:
                T *= cooling_rate
                continue
                
            if c_new in cw_set:
                T *= cooling_rate
                continue
                
            # Calculate delta in "coverage quality"
            # We want to maintain coverage.
            # If we remove c_old and add c_new, does coverage remain >= 1?
            
            # Words that become uncovered:
            # Those nb in neighbors(c_old) where cov[nb] == 1 AND cov[nb] is not incremented by c_new.
            # i.e., nb is not in neighbors(c_new).
            
            # Words that become covered (were 0, now 1):
            # Those nb in neighbors(c_new) where cov[nb] == 0.
            
            # We compute the change in the number of uncovered words.
            # Let U be the set of uncovered words.
            # delta_U = |U_new| - |U_old|
            
            # Instead of sets, we can compute:
            # lost_coverage = 0
            # gained_coverage = 0
            
            # For each nb in neighbors(c_old):
            #   if cov[nb] == 1:
            #       if nb not in neighbors(c_new):
            #           lost_coverage += 1
            #           # This word becomes uncovered
            
            # For each nb in neighbors(c_new):
            #   if cov[nb] == 0:
            #       # This word becomes covered (if it wasn't covered by c_old, which it isn't if cov==0)
            #       # But wait, if cov[nb]==0, it wasn't covered by c_old.
            #       # So it becomes covered.
            #       gained_coverage += 1
            #       # But we must ensure it's not already covered by others? 
            #       # cov[nb]==0 means it's NOT covered by others.
            #       # So adding c_new covers it.
            
            # Net change in uncovered count:
            # delta_uncovered = lost_coverage - gained_coverage
            
            # We accept if delta_uncovered <= 0 (improvement or neutral)
            # Or if delta_uncovered > 0 with probability exp(-delta/T)
            
            lost_coverage = 0
            gained_coverage = 0
            
            # Optimization: Use sets of neighbors?
            # Neighbors are lists.
            # For speed, we can precompute neighbor sets? 
            # Or just check.
            
            # Let's use a temporary array or just iterate.
            # Iterating 15 items is fast.
            
            # Check lost
            for nb in _NEIGHBORS[c_old]:
                if cov[nb] == 1:
                    # Check if nb is in neighbors of c_new
                    # We can check if nb is covered by c_new
                    # i.e., if Hamming distance(c_new, nb) <= 1
                    # Which is equivalent to nb in _NEIGHBORS[c_new]
                    # To avoid O(15) search, we can precompute a boolean array for c_new neighbors?
                    # Or just check.
                    # Let's just check.
                    if nb in _NEIGHBORS[c_new]:
                        pass
                    else:
                        lost_coverage += 1
            
            # Check gained
            for nb in _NEIGHBORS[c_new]:
                if cov[nb] == 0:
                    # It was uncovered. Now covered by c_new.
                    # But is it covered by c_old? No, because cov[nb]==0.
                    # So it becomes covered.
                    gained_coverage += 1
            
            delta_uncovered = lost_coverage - gained_coverage
            
            if delta_uncovered <= 0:
                # Accept
                # Update cov
                for nb in _NEIGHBORS[c_old]:
                    cov[nb] -= 1
                for nb in _NEIGHBORS[c_new]:
                    cov[nb] += 1
                
                # Update codewords
                codewords[i] = c_new
                cw_set.remove(c_old)
                cw_set.add(c_new)
                
                # Check if length is same, but maybe we can remove more later?
                # Length didn't change.
                # However, if we can now remove a codeword, that's a future step.
                
            else:
                # Accept with probability
                prob = np.exp(-delta_uncovered / T)
                if rng.random() < prob:
                    # Accept
                    for nb in _NEIGHBORS[c_old]:
                        cov[nb] -= 1
                    for nb in _NEIGHBORS[c_new]:
                        cov[nb] += 1
                    
                    codewords[i] = c_new
                    cw_set.remove(c_old)
                    cw_set.add(c_new)
            
        T *= cooling_rate
        
        # Periodic check for best
        # Since we maintain validity, len(codewords) is the metric.
        # We only update best if we removed a codeword (move_type 0).
        # Wait, in move_type 1 (swap), length doesn't change.
        # So best_len only improves in move_type 0.
        
        # To improve swap, we should look for swaps that enable removals.
        # This is complex.
        # Alternative: After a swap, try to remove a random codeword.
        # If successful, we have improved.
        
        # Let's add a "polish" step every 100 iterations.
        # Try to remove any removable codeword.
        if rng.integers(0, 100) == 0:
            for i in range(len(codewords)):
                c = codewords[i]
                removable = True
                for nb in _NEIGHBORS[c]:
                    if cov[nb] == 1:
                        removable = False
                        break
                if removable:
                    codewords.pop(i)
                    cw_set.remove(c)
                    for nb in _NEIGHBORS[c]:
                        cov[nb] -= 1
                    if len(codewords) < best_len:
                        best_len = len(codewords)
                        best_cw = list(codewords)
                    break # Restart loop? Or just continue.
    
    return best_cw

def construct(q, n, radius):
    global Q, N, R, TOTAL_WORDS, _NEIGHBORS
    
    # Verify parameters
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (though task specifies 3,7,1)
        # Simple greedy
        pass
    
    t0 = time.time()
    deadline = 120.0 # Safety margin
    
    # Precompute neighbors if not done
    if _NEIGHBORS is None or len(_NEIGHBORS) != TOTAL_WORDS:
        _NEIGHBORS = [None] * TOTAL_WORDS
        for i in range(TOTAL_WORDS):
            _NEIGHBORS[i] = _get_neighbors(i)
            
    rng = np.random.default_rng(42)
    
    # Start with Tetracode construction
    start_cw = _initial_tetracode()
    
    # Run multiple SA instances
    best_overall = start_cw
    best_len = len(start_cw)
    
    # Time budget per SA run
    # We have ~110s. Let's do 5 runs of 20s each?
    # Or 10 runs of 10s?
    # SA benefits from longer runs.
    
    num_runs = 5
    time_per_run = (deadline - (time.time() - t0)) / num_runs
    
    for i in range(num_runs):
        if time.time() - t0 > deadline - 5.0:
            break
            
        # Vary the start slightly
        if i == 0:
            current_start = start_cw
        else:
            # Perturb the best solution found so far
            current_start = list(best_overall)
            # Randomly swap 2-3 codewords
            for _ in range(3):
                if len(current_start) > 0:
                    idx = int(rng.integers(0, len(current_start)))
                    new_cw = int(rng.integers(0, TOTAL_WORDS))
                    current_start[idx] = new_cw
        
        # Run SA
        result = _simulated_annealing(current_start, time_per_run, rng)
        
        if len(result) < best_len:
            best_len = len(result)
            best_overall = result
            
        # Check time
        if time.time() - t0 > deadline - 10.0:
            break
            
    # Final polish: Try to remove any redundant codewords from best_overall
    cov = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in best_overall:
        for nb in _NEIGHBORS[c]:
            cov[nb] += 1
            
    changed = True
    while changed and (time.time() - t0) < deadline - 2.0:
        changed = False
        for i in range(len(best_overall)):
            c = best_overall[i]
            removable = True
            for nb in _NEIGHBORS[c]:
                if cov[nb] == 1:
                    removable = False
                    break
            if removable:
                best_overall.pop(i)
                for nb in _NEIGHBORS[c]:
                    cov[nb] -= 1
                changed = True
                break
                
    # Ensure validity
    cov_final = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in best_overall:
        for nb in _NEIGHBORS[c]:
            cov_final[nb] += 1
            
    if not np.all(cov_final >= 1):
        # Fallback to initial valid solution
        return start_cw
        
    # Convert indices back to lists of integers
    result = []
    for idx in best_overall:
        result.append(list(_idx_to_tuple(idx)))
        
    return result

# Ensure global variables are initialized
_NEIGHBORS = [None] * TOTAL_WORDS
for i in range(TOTAL_WORDS):
    _NEIGHBORS[i] = _get_neighbors(i)
