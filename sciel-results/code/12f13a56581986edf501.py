import time
import random
import itertools

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the task says 
    # "the evaluator passes this task's alphabet size q, length n and covering radius".
    # The task is specifically K_3(6,1).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a generic construction if parameters differ
        # Just return a trivial code (all 0s plus variations) or similar
        # For safety, return a code that is definitely valid but large
        # e.g. all words with first n-1 coords 0 and last coord 0..q-1? No.
        # Let's just use the direct sum logic if n>=4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This only works for q=3.
            if q == 3:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # If not ternary or small n, just return a large valid code
        # Covering radius 1 means we need to cover all words.
        # A simple valid code is the whole space.
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute neighbors for all 729 words
    # Word representation: integer 0..728
    # Convert int to base-3 tuple
    def int_to_tuple(x, n):
        return [x // (3**i) % 3 for i in range(n)]
    
    def tuple_to_int(t, n):
        return sum(t[i] * (3**i) for i in range(n))
        
    def hamming_dist(t1, t2):
        return sum(1 for a, b in zip(t1, t2) if a != b)
        
    # Precompute neighbors list
    # neighbors[i] is a set of integers
    all_words = range(3**n)
    neighbors = [set() for _ in all_words]
    # For speed, precompute the list of changes
    # For a word w, neighbors are w itself and w with one digit changed to 0, 1, 2 (if different)
    for i in all_words:
        t = int_to_tuple(i, n)
        neighbors[i].add(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(3):
                if new_val != orig:
                    t[pos] = new_val
                    neighbors[i].add(tuple_to_int(t, n))
            t[pos] = orig
            
    # Initial code: Direct sum of tetracode
    tetracode_ints = []
    for a in range(3):
        for b in range(3):
            w = [a, b, (a + b) % 3, (a + 2 * b) % 3]
            tetracode_ints.append(tuple_to_int(w, 4))
            
    initial_code = []
    for h in tetracode_ints:
        for tail in itertools.product(range(3), repeat=n - 4):
            # Combine h (first 4) and tail (last 2)
            # h is an int, tail is a tuple
            # We need to construct the full int
            # h covers indices 0..3. tail covers 4..5.
            full_t = int_to_tuple(h, 4) + list(tail)
            initial_code.append(tuple_to_int(full_t, n))
            
    code = initial_code[:]
    cover_count = [0] * (3**n)
    
    def update_cover_count(c, delta):
        for w in neighbors[c]:
            cover_count[w] += delta
            
    for c in code:
        update_cover_count(c, 1)
        
    best_code = code[:]
    best_size = len(code)
    
    # Local search
    # To avoid getting stuck, we can use a simple random walk if no redundant codeword is found
    # Redundant check: for c in code, if all(cover_count[w] >= 2 for w in neighbors[c])
    
    while time.time() < deadline:
        # 1. Try to remove redundant codewords
        redundant_found = False
        # Find all redundant
        redundant = []
        for c in code:
            # Check if removing c leaves all neighbors covered
            # i.e., cover_count[w] >= 2 for all w in neighbors[c]
            is_red = True
            min_slack = float('inf')
            for w in neighbors[c]:
                if cover_count[w] < 2:
                    is_red = False
                    break
                if cover_count[w] < min_slack:
                    min_slack = cover_count[w]
            if is_red:
                redundant.append((c, min_slack))
                
        if redundant:
            # Remove the one with smallest slack (most critical to keep? No, smallest slack means it's barely redundant? 
            # Actually, if min_slack is 2, it's safe. If min_slack is 10, it's very safe.
            # We want to remove one. Let's pick the one with smallest min_slack to keep the cover tight?
            # Or largest? 
            # Usually, removing a codeword with high redundancy is safer.
            # Let's just pick one randomly from redundant to explore.
            c_to_remove, _ = random.choice(redundant)
            code.remove(c_to_remove)
            update_cover_count(c_to_remove, -1)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            redundant_found = True
            continue # Restart loop to check for new redundancies
            
        # 2. If no redundant, try a swap
        # Pick a random codeword c
        if not code:
            break
            
        c = random.choice(code)
        # Remove c
        code.remove(c)
        update_cover_count(c, -1)
        
        # Check validity
        uncovered = [w for w in range(3**n) if cover_count[w] == 0]
        
        if not uncovered:
            # We successfully removed a codeword! (This shouldn't happen if redundant check was perfect, 
            # but maybe the redundant check was slightly off or we just got lucky? 
            # Actually, if no codeword was redundant, removing any should cause uncovered words.
            # So this branch is unlikely unless redundant check missed something.)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            continue
            
        # We have uncovered words. We need to add a codeword c_new that covers all of them.
        # Candidates must be in the intersection of neighbors[w] for all w in uncovered.
        # Or simpler: pick a w from uncovered. c_new must be in neighbors[w].
        # Then check if c_new covers all other uncovered words.
        
        candidates = []
        # To speed up, pick the first uncovered word
        w0 = uncovered[0]
        for cand in neighbors[w0]:
            # Check if cand covers all uncovered
            valid = True
            for w in uncovered:
                if w not in neighbors[cand]:
                    valid = False
                    break
            if valid:
                candidates.append(cand)
                
        if not candidates:
            # This should not happen if the code was valid before removal?
            # Wait, if c was the only cover for w0, and we remove c, w0 is uncovered.
            # Any neighbor of w0 covers w0. But does it cover ALL uncovered?
            # It's possible that no single codeword covers all uncovered words.
            # In that case, we cannot simply swap one for one.
            # We must revert.
            code.append(c)
            update_cover_count(c, 1)
            continue
            
        # Pick a candidate. 
        # Heuristic: pick the one that maximizes the minimum cover count of the words it covers?
        # Or just random.
        c_new = random.choice(candidates)
        
        if c_new == c:
            # No change
            code.append(c)
            update_cover_count(c, 1)
            continue
            
        # Add c_new
        code.append(c_new)
        update_cover_count(c_new, 1)
        
        # Check if size improved (it shouldn't, it's a swap)
        # But maybe this swap enables future removals.
        
    # Convert best_code ints to lists of ints
    return [int_to_tuple(c, n) for c in best_code]
