import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Parameters
    total_words = q ** n
    # Precompute balls
    # balls[i] is a list of indices j such that dist(i, j) <= radius
    # For radius 1, it's i itself and all words differing in 1 position.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    balls = []
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                ball.append(word_to_idx[new_w])
        balls.append(ball)
        
    # Convert to numpy arrays for speed?
    # balls_np = [np.array(b) for b in balls]
    
    # Initialize with Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            init_codewords.append(word_to_idx[tuple(cw)])
            
    codewords = set(init_codewords)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Build initial cover count
    for cw in codewords:
        cover_count[balls[cw]] += 1
        
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_size = len(codewords)
    best_codewords = set(codewords)
    best_cover_count = cover_count.copy()
    
    # Local Search
    while time.time() - start_time < deadline:
        # 1. Try to delete redundant codewords
        removed = True
        while removed:
            removed = False
            to_remove = []
            for cw in list(codewords):
                # Check if redundant
                if np.all(cover_count[balls[cw]] > 1):
                    to_remove.append(cw)
            if to_remove:
                for cw in to_remove:
                    cover_count[balls[cw]] -= 1
                    codewords.remove(cw)
                removed = True
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = set(codewords)
            best_cover_count = cover_count.copy()
            
        # 2. Swap
        if len(codewords) == 0:
            break
            
        # Pick random c_out
        c_out = random.choice(list(codewords))
        
        # Find critical words
        # Words that are covered ONLY by c_out
        # cover_count[w] == 1 and c_out in balls[w]
        # Since we are removing c_out, any w in balls[c_out] with cover_count[w]==1 becomes uncovered.
        w_crit = [w for w in balls[c_out] if cover_count[w] == 1]
        
        candidates = set(range(total_words))
        if w_crit:
            # Intersect balls of all critical words
            # Start with all, intersect with balls[w] for each w in w_crit
            # Optimization: pick the one with smallest ball? All have size 17.
            # Just intersect.
            candidates = set(balls[w_crit[0]])
            for w in w_crit[1:]:
                candidates.intersection_update(balls[w])
                if not candidates:
                    break
        
        if not candidates:
            continue # Swap impossible
            
        c_in = random.choice(list(candidates))
        
        # Perform swap
        # Save state
        old_cover_count = cover_count.copy()
        old_codewords = set(codewords)
        
        # Remove c_out
        cover_count[balls[c_out]] -= 1
        codewords.remove(c_out)
        
        # Add c_in
        cover_count[balls[c_in]] += 1
        codewords.add(c_in)
        
        # Check validity (should be valid, but check anyway)
        if np.any(cover_count == 0):
            # Revert
            cover_count = old_cover_count
            codewords = old_codewords
            continue
            
        # Try to delete redundant again
        removed = True
        while removed:
            removed = False
            to_remove = []
            for cw in list(codewords):
                if np.all(cover_count[balls[cw]] > 1):
                    to_remove.append(cw)
            if to_remove:
                for cw in to_remove:
                    cover_count[balls[cw]] -= 1
                    codewords.remove(cw)
                removed = True
        
        new_size = len(codewords)
        old_size = len(old_codewords)
        
        # Acceptance criterion
        accept = False
        if new_size < old_size:
            accept = True
        elif new_size == old_size:
            # Accept with some probability to explore
            if random.random() < 0.5:
                accept = True
        else:
            accept = False # Should not happen
            
        if not accept:
            # Revert
            cover_count = old_cover_count
            codewords = old_codewords
            
    # Return best found
    # Convert indices back to words
    result = [list(all_words[i]) for i in best_codewords]
    return result
