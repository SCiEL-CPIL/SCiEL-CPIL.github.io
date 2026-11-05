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
        
    # Precompute neighbors as lists (faster iteration than sets)
    all_words = range(3**n)
    neighbors = [None] * len(all_words)
    for i in all_words:
        t = int_to_tuple(i, n)
        nb = [i]
        for pos in range(n):
            orig = t[pos]
            for new_val in range(3):
                if new_val != orig:
                    t[pos] = new_val
                    nb.append(tuple_to_int(t, n))
            t[pos] = orig
        neighbors[i] = nb
            
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
    word_in_code = [False] * (3**n)
    for c in code:
        word_in_code[c] = True

    def update_cover_count(c, delta):
        for w in neighbors[c]:
            cover_count[w] += delta

    for c in code:
        update_cover_count(c, 1)

    best_code = code[:]
    best_size = len(code)
    
    # Local search with faster inner loops
    while time.time() < deadline:
        # 1. Try to remove redundant codewords
        # Find the redundant codeword with the largest slack (safest to remove)
        best_red = None
        best_slack = -1
        for c in code:
            min_cc = min(cover_count[w] for w in neighbors[c])
            if min_cc >= 2:
                if min_cc > best_slack:
                    best_slack = min_cc
                    best_red = c
        if best_red is not None:
            code.remove(best_red)
            word_in_code[best_red] = False
            update_cover_count(best_red, -1)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            continue  # Restart to check for new redundancies

        # 2. If no redundant, try a swap
        if not code:
            break

        c = random.choice(code)
        # Remove c
        code.remove(c)
        word_in_code[c] = False
        update_cover_count(c, -1)

        # Check validity
        uncovered = [w for w in range(3**n) if cover_count[w] == 0]

        if not uncovered:
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            continue

        # We have uncovered words. Find candidates in the intersection of neighbors of all uncovered words.
        # Start with the smallest neighbor list to minimize work.
        uncovered_sorted = sorted(uncovered, key=lambda w: len(neighbors[w]))
        w0 = uncovered_sorted[0]
        candidate_set = set(neighbors[w0])
        for w in uncovered_sorted[1:]:
            if not candidate_set:
                break
            candidate_set &= set(neighbors[w])

        if not candidate_set:
            # Cannot swap one for one; revert
            code.append(c)
            word_in_code[c] = True
            update_cover_count(c, 1)
            continue

        # Pick the candidate that maximizes the minimum cover-count increment over uncovered words
        best_cand = None
        best_score = -1
        for cand in candidate_set:
            # Compute min increment over uncovered
            min_inc = min(cover_count[w] + 1 for w in uncovered)
            # Prefer candidates not already in code (to enable size reduction later)
            if not word_in_code[cand]:
                min_inc += 0.1  # slight preference for new codewords
            if min_inc > best_score:
                best_score = min_inc
                best_cand = cand

        c_new = best_cand
        if c_new == c:
            code.append(c)
            word_in_code[c] = True
            update_cover_count(c, 1)
            continue

        code.append(c_new)
        word_in_code[c_new] = True
        update_cover_count(c_new, 1)

    # Final polish: try a few more removals if time remains
    while time.time() < deadline - 2.0:
        best_red = None
        best_slack = -1
        for c in code:
            min_cc = min(cover_count[w] for w in neighbors[c])
            if min_cc >= 2:
                if min_cc > best_slack:
                    best_slack = min_cc
                    best_red = c
        if best_red is None:
            break
        code.remove(best_red)
        word_in_code[best_red] = False
        update_cover_count(best_red, -1)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Convert best_code ints to lists of ints
    return [int_to_tuple(c, n) for c in best_code]
