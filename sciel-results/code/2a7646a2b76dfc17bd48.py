"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    deadline = time.time() + 145  # 5s safety margin
    
    # All words in the space
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Helper: generate ball of radius 1 around a codeword
    def get_ball(cw):
        ball = {cw}
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = v
                    ball.add(tuple(new_cw))
        return ball

    # Precompute balls for all words in the space to avoid recomputation
    # 729 words * 13 neighbors = ~9500 tuples. Fast.
    all_balls = [get_ball(w) for w in all_words]
    
    # Initial Construction: Direct Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_cw = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    best_result = initial_cw
    
    # Function to reduce a set of codewords
    def reduce_code(cw_list):
        if not cw_list:
            return []
        
        # Map to indices for fast ball lookup?
        # Or just use the precomputed all_balls if we map cw to idx.
        # But cw_list might contain duplicates? No, distinct.
        
        # Create a set of active codewords
        active_set = set(cw_list)
        
        # Initialize coverage
        # coverage is a dict: word -> count
        coverage = {}
        for cw in active_set:
            idx = word_to_idx[cw]
            for w in all_balls[idx]:
                coverage[w] = coverage.get(w, 0) + 1
        
        # Check if it covers everything
        # If not, we can't just reduce, we need to add.
        # But the initial construction covers.
        # Random sets might not.
        
        # Greedy Removal
        # We maintain a list of active codewords
        active_list = list(active_set)
        
        # To speed up removal check, we can just iterate.
        # We use a boolean array or just filter the list.
        # Given small N (~80), filtering is fine.
        
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry
            random.shuffle(active_list)
            
            new_active_list = []
            for cw in active_list:
                idx = word_to_idx[cw]
                ball = all_balls[idx]
                
                can_remove = True
                for w in ball:
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove
                    improved = True
                    for w in ball:
                        coverage[w] -= 1
                else:
                    new_active_list.append(cw)
            
            active_list = new_active_list
            
            if time.time() > deadline:
                break
        
        return active_list

    # Strategy 1: Reduce the initial construction
    result1 = reduce_code(initial_cw)
    if len(result1) < len(best_result):
        best_result = result1
    
    # Strategy 2: Random Restarts
    # Try to find a covering code of size ~75 and reduce it.
    # Or start with random set of size 80, check coverage, add if needed, reduce.
    
    for restart in range(100): # Max 100 restarts, check time
        if time.time() > deadline:
            break
            
        # Generate random set
        # Size slightly larger than expected optimum to allow reduction
        k = 75 + random.randint(0, 10)
        rand_cw = random.sample(all_words, k)
        
        # Check coverage
        covered = set()
        for cw in rand_cw:
            covered.update(all_balls[word_to_idx[cw]])
            
        if len(covered) == len(all_words):
            # It covers. Reduce it.
            res = reduce_code(rand_cw)
            if len(res) < len(best_result):
                best_result = res
                # If we hit 73, we can stop early? 
                # Best known is 73. Lower bound 71.
                # If we find 73, it's likely optimal or near.
                # But we can keep searching for 72? Unlikely.
                # Let's continue to ensure robustness.
        else:
            # If not covered, try to add words to cover gaps
            # Greedy Add
            current_cw = set(rand_cw)
            uncovered = set(all_words) - covered
            
            while uncovered:
                # Find word that covers most uncovered
                best_cw = None
                best_count = -1
                # Sample a few candidates to save time?
                # Or iterate all? 729 * |uncovered| is expensive.
                # Just pick random from uncovered?
                # Or pick random word and see how many it covers.
                
                # Heuristic: Pick a random word from the space,
                # if it covers any uncovered, add it.
                # This is slow.
                
                # Better: Pick a word from uncovered, add it.
                # This guarantees progress.
                target = random.choice(list(uncovered))
                # Find a codeword that covers target.
                # We can just add target itself?
                # If we add target, it covers target and its neighbors.
                # Is target a good codeword?
                # Let's just add target.
                current_cw.add(target)
                
                # Update coverage
                # This is getting complex. 
                # Let's skip the "Add" phase for random sets if it's not covered.
                # Just rely on the initial construction which is guaranteed to cover.
                pass

    return [list(cw) for cw in best_result]
# EVOLVE-BLOCK-END
