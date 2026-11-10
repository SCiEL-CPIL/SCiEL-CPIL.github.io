import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Specific optimization for the known hard case: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_ternary_7_1()
    
    # Fallback for other parameters: Simple Greedy Set Cover
    # This is slower but works for smaller spaces
    return _construct_greedy_general(q, n, radius, deadline)

def _construct_ternary_7_1():
    q = 3
    n = 7
    total_words = 3 ** n  # 2187
    
    # Precompute mapping from index to word and vice versa
    # Index is base-3 representation
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w
    
    def word_to_idx(w):
        idx = 0
        multiplier = 1
        for i in range(n):
            idx += w[i] * multiplier
            multiplier *= 3
        return idx

    # Precompute the bitmask for each word's covering ball
    # Bit i is set if word i is covered
    balls = [0] * total_words
    
    for i in range(total_words):
        center = idx_to_word(i)
        mask = 0
        # Add center
        mask |= (1 << i)
        
        # Add neighbors at distance 1
        for pos in range(n):
            orig_val = center[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbor = center.copy()
                neighbor[pos] = new_val
                n_idx = word_to_idx(neighbor)
                mask |= (1 << n_idx)
        balls[i] = mask

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    selected_indices = []
    
    # Optimization: Keep a list of active candidates. Initially all 0..total_words-1
    # We pick the one that covers the most new bits.
    # To speed up, we can just iterate all 2187 each time. 2187 * ~200 steps = ~440k iterations.
    # Each iteration is bitwise AND + bit_count. Very fast.
    
    while uncovered:
        best_idx = -1
        max_newly = -1
        
        # Iterate through all potential codewords
        # We can skip already selected ones, but checking `in` is slow.
        # Just picking the global max is fine. If we pick a selected one, newly will be 0 (or low).
        # Actually, if we pick a selected one, its ball is already covered, so newly=0.
        # So we just pick the argmax of (balls[i] & uncovered).bit_count()
        
        for i in range(total_words):
            newly = (balls[i] & uncovered).bit_count()
            if newly > max_newly:
                max_newly = newly
                best_idx = i
                
        if max_newly == 0:
            # Should not happen if uncovered is not 0 and we have valid balls
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~balls[best_idx]
        
        # Time check
        if time.time() > deadline:
            break

    # Convert indices back to words
    codewords = [idx_to_word(i) for i in selected_indices]
    
    # Polish: Remove redundant codewords
    # Iterate through selected codewords. If removing one doesn't change coverage, remove it.
    # We need to re-verify coverage.
    # Since n is small, we can do this.
    
    # Let's do a simple reduction pass
    current_codewords = codewords[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_codewords)):
            # Check if current_codewords[i] is redundant
            # Temporarily remove it
            test_codewords = current_codewords[:i] + current_codewords[i+1:]
            
            # Check if all words are still covered
            # Efficient check: Rebuild coverage mask?
            # 2187 words * len(test_codewords) checks is too slow if done naively.
            # Better: Calculate union of balls of test_codewords and see if it covers all.
            
            union_mask = 0
            for cw in test_codewords:
                idx = word_to_idx(cw)
                union_mask |= balls[idx]
                
            if union_mask == (1 << total_words) - 1:
                # It is redundant
                current_codewords = test_codewords
                changed = True
                break # Restart loop as indices changed

    return current_codewords

def _construct_greedy_general(q, n, radius, deadline):
    # Fallback implementation for other parameters
    # Uses a simple greedy approach without bitmask optimization (slower)
    # For very large spaces, this will time out, but the task is specific to 3,7,1
    
    # Generate all words
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute distances? No, just compute on the fly for neighbors.
    # This is O(N * Q * N) which is bad.
    # For the general case, we assume small N.
    
    def get_ball(w, radius):
        covered = set()
        covered.add(w)
        # Generate all words within distance radius
        # This is expensive for large radius/n.
        # For radius 1:
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    covered.add(tuple(new_w))
        return covered

    # Greedy
    uncovered = set(all_words)
    selected = []
    
    # Precompute balls for all words to speed up greedy
    # This takes O(3^n * n * q)
    balls = {}
    for w in all_words:
        balls[w] = get_ball(w, radius)
        
    while uncovered:
        best_w = None
        max_newly = -1
        for w in all_words:
            newly = len(balls[w] & uncovered)
            if newly > max_newly:
                max_newly = newly
                best_w = w
        if max_newly == 0:
            break
        selected.append(best_w)
        uncovered -= balls[best_w]
        if time.time() > deadline:
            break
            
    return [list(w) for w in selected]
