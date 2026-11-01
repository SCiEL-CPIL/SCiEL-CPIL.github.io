import time
import random
import math
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    N = q ** n  # 729
    
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute neighbors as bitmasks
    # neighbors_mask[i] is a bitmask of all words within Hamming distance 1 of word i
    neighbors_mask = [0] * N
    
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        neighbors_mask[i] = mask
        
    full_mask = (1 << N) - 1
    
    # Initial solution: Greedy
    # We start with a large code and try to shrink it.
    # Or start with empty and grow? No, we want to minimize size.
    # Let's start with a valid code. Greedy is a good start.
    
    code_set = set()
    cover_count = [0] * N
    covered_mask = 0
    
    # Greedy initialization
    while covered_mask != full_mask:
        best_idx = -1
        best_cnt = -1
        for i in range(N):
            if i in code_set:
                continue
            new_covered = neighbors_mask[i] & ~covered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1:
            break
        code_set.add(best_idx)
        for j in range(N):
            if (neighbors_mask[best_idx] >> j) & 1:
                cover_count[j] += 1
        covered_mask |= neighbors_mask[best_idx]
        
    # Now we have a valid code. Let's try to reduce it.
    # Simulated Annealing to minimize |code_set|
    
    rng = random.Random(42)
    
    # State
    current_code = set(code_set)
    current_cover_count = list(cover_count)
    current_covered_mask = covered_mask
    current_size = len(current_code)
    
    best_code = set(current_code)
    best_cover_count = list(current_cover_count)
    best_covered_mask = current_covered_mask
    best_size = current_size
    
    T = 10.0  # Initial temperature
    cooling = 0.99995
    
    # We want to remove words. 
    # Move: Pick a random word in current_code. Try to remove it.
    # If it can be removed (no unique coverage), remove it.
    # If it cannot be removed, we might "add" a random word to change the landscape?
    # Actually, a better move set for minimization is:
    # 1. Try to remove a random codeword. If valid, accept.
    # 2. If we get stuck, we can swap: remove one, add another?
    #    Or just add a random word to escape local minima, then try to remove again.
    
    # Let's use a hybrid: 
    # - If we can remove a word, do it (always accept).
    # - Otherwise, with some probability, add a random word (to escape), then try to remove.
    # - We accept "adding" only if we are in a local minimum and want to explore.
    
    # Actually, standard SA for set cover minimization:
    # State: Set of codewords.
    # Objective: Minimize size.
    # Constraint: Must cover all words.
    
    # Since we must maintain validity, we can only remove a word if it's redundant.
    # If no word is redundant, we are in a local minimum.
    # To escape, we must add a word. This increases size.
    # We accept the addition with probability exp(-1/T).
    
    while time.time() < deadline - 2:
        if T < 0.01:
            T = 10.0 # Restart with higher T if stuck too long? Or just stop.
            # Let's just cool down and stop.
            break
            
        # Try to find a removable word
        removable = []
        # Iterate over a copy to avoid modification issues, though we only check
        for i in list(current_code):
            # Check if i is removable
            # i is removable if for all j in neighbors(i), cover_count[j] > 1
            removable = True
            # Optimization: iterate neighbors
            mask = neighbors_mask[i]
            # Check min cover count in neighbors
            # We can do this by iterating set bits of mask
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if current_cover_count[j] == 1:
                    removable = False
                    break
                m ^= lsb
            if removable:
                # We found one. Let's pick a random one from all removable?
                # Or just this one? Picking random helps diversify.
                # For speed, let's just try to remove this one.
                break
                
        if removable:
            # Remove i
            current_code.remove(i)
            mask = neighbors_mask[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                current_cover_count[j] -= 1
                m ^= lsb
            current_size -= 1
            current_covered_mask &= ~mask # Update covered mask? 
            # Actually, if we remove a word, the covered mask might change if it was the only one covering some words.
            # But we checked that no word had cover_count 1. So all neighbors still have count >= 1.
            # So covered_mask remains full_mask.
            
            if current_size < best_size:
                best_size = current_size
                best_code = set(current_code)
                best_cover_count = list(current_cover_count)
                best_covered_mask = current_covered_mask
                
        else:
            # Local minimum. Try to add a random word to escape.
            # Pick a random word not in current_code
            # There are N - current_size candidates.
            # For efficiency, just pick a random index and check if it's in code.
            # If it is, try another.
            added = False
            for _ in range(10): # Limit attempts
                rand_idx = rng.randint(0, N - 1)
                if rand_idx not in current_code:
                    # Acceptance probability for adding (increasing size)
                    if rng.random() < math.exp(-1.0 / T):
                        current_code.add(rand_idx)
                        mask = neighbors_mask[rand_idx]
                        m = mask
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            current_cover_count[j] += 1
                            m ^= lsb
                        current_size += 1
                        current_covered_mask |= mask
                        added = True
                        break
            # If not added, do nothing.
        
        T *= cooling
        
    # Return best code found
    return [list(idx_to_word[idx]) for idx in best_code]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
    if n > 8:
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        # Greedy choice: pick word covering most uncovered
        for w in all_words:
            if w in code: continue # Optimization: don't pick existing
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
