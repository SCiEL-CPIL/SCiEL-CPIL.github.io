import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Parameters for the specific task: q=3, n=6, radius=1
    # The task is fixed, but the function signature is general.
    # We will optimize for q=3, n=6, radius=1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        random.shuffle(all_words)
        codewords = []
        covered = set()
        for w in all_words:
            if w not in covered:
                codewords.append(list(w))
                # Add ball of w to covered
                # Simple expansion for radius 1
                for i in range(n):
                    for val in range(q):
                        if w[i] != val:
                            new_word = list(w)
                            new_word[i] = val
                            covered.add(tuple(new_word))
                covered.add(w)
        return codewords

    # Specific optimization for q=3, n=6, r=1
    total_words = 729
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = all_words
    
    # Precompute balls
    # ball_idx[idx] = list of indices in the ball of idx
    ball_idx = [None] * total_words
    for i, w in enumerate(all_words):
        b = [i]
        w_list = list(w)
        for pos in range(6):
            orig = w_list[pos]
            for val in range(3):
                if val != orig:
                    w_list[pos] = val
                    b.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        ball_idx[i] = b
        
    # Initial Cover: Direct Sum Construction (Tetracode * All_2)
    # Tetracode words: (a, b, a+b, a+2b) mod 3 for a,b in 0..2
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tails = list(itertools.product(range(3), repeat=2))
    
    initial_cw_idxs = set()
    for h in tetracode:
        for t in tails:
            word = tuple(h + list(t))
            initial_cw_idxs.add(word_to_idx[word])
            
    cw_idxs = initial_cw_idxs
    cover_count = [0] * total_words
    
    # Initialize cover counts
    for cw in cw_idxs:
        for w_idx in ball_idx[cw]:
            cover_count[w_idx] += 1
            
    def is_redundant(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            if cover_count[w_idx] == 1:
                return False
        return True
        
    def remove_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] -= 1
        cw_idxs.remove(cw_idx)
        
    def add_cw(cw_idx):
        for w_idx in ball_idx[cw_idx]:
            cover_count[w_idx] += 1
        cw_idxs.add(cw_idx)
        
    def reduce():
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid bias
            cws_to_check = list(cw_idxs)
            random.shuffle(cws_to_check)
            for cw in cws_to_check:
                if is_redundant(cw):
                    remove_cw(cw)
                    improved = True
                    break # Restart scan after removal
        
    # Main Search Loop
    best_size = len(cw_idxs)
    best_state = set(cw_idxs)
    
    # We want to find 73. If we find it, we can stop or try to find 72 (unlikely/impossible).
    # Let's run ILS until time runs out or we hit 73.
    
    attempts = 0
    max_attempts = 200000 # High limit, time is the real constraint
    
    while time.time() < deadline:
        attempts += 1
        
        # 1. Reduce
        reduce()
        
        current_size = len(cw_idxs)
        if current_size < best_size:
            best_size = current_size
            best_state = set(cw_idxs)
            
        if best_size == 73:
            # Found the known optimal (or close). Break to save time?
            # The problem says "best known is 73". If we find 73, it's a great score.
            # We can break to ensure we return quickly.
            break
            
        # 2. If not reduced, try to kick
        # Strategy: Find a codeword c with unique words, and try to swap it with w that covers those unique words.
        kick_success = False
        cws_list = list(cw_idxs)
        random.shuffle(cws_list)
        
        for c in cws_list:
            # Find words covered only by c
            unique_words = []
            for w_idx in ball_idx[c]:
                if cover_count[w_idx] == 1:
                    unique_words.append(w_idx)
            
            if not unique_words:
                # Should have been removed in reduce(), but just in case
                remove_cw(c)
                kick_success = True
                break
            
            # Find a word w (not in cw_idxs) that covers all unique_words
            # w must be in the intersection of balls of unique_words
            # Start with candidates from the first unique word's ball
            candidates = ball_idx[unique_words[0]]
            
            valid_candidate = -1
            for w_cand in candidates:
                if w_cand in cw_idxs:
                    continue
                # Check if w_cand covers all other unique words
                ok = True
                for u in unique_words[1:]:
                    # Check if u is in ball_idx[w_cand]
                    # Since ball size is small (13), linear search is fine
                    if u not in ball_idx[w_cand]:
                        ok = False
                        break
                if ok:
                    valid_candidate = w_cand
                    break
            
            if valid_candidate != -1:
                # Perform swap
                remove_cw(c)
                add_cw(valid_candidate)
                kick_success = True
                break
        
        if not kick_success:
            # No smart swap found. Perform a random perturbation.
            # Randomly pick a codeword and replace it with a random word.
            if cw_idxs:
                c_to_swap = random.choice(list(cw_idxs))
                remove_cw(c_to_swap)
                # Pick a random word from the whole space
                w_new = random.randint(0, total_words - 1)
                add_cw(w_new)
            else:
                # Empty set, add a random word
                w_new = random.randint(0, total_words - 1)
                add_cw(w_new)

    # Return the best state found
    # If best_state is better than current cw_idxs, use it.
    # Actually, the loop updates best_state whenever we reduce.
    # If we broke out early due to 73, best_state is 73.
    # If time ran out, best_state is the minimum size found.
    
    result_idxs = best_state
    result = [list(idx_to_word[idx]) for idx in result_idxs]
    
    return result
