import numpy as np
import time

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with given radius.
    Uses a vectorized Greedy Set Cover approach followed by a local optimization (removal) phase.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin before 150s kill

    total_words = q ** n

    def idx_to_list(idx, base, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = idx % base
            idx //= base
        return res

    def list_to_idx(lst, base):
        idx = 0
        for d in lst:
            idx = idx * base + d
        return idx

    # Precompute balls as lists of indices for efficient numpy operations
    # Ball size for q=3, n=7, r=1 is 1 + 7*2 = 15
    balls = []
    for i in range(total_words):
        word = idx_to_list(i, q, n)
        ball_set = set()
        ball_set.add(i)
        
        # Add neighbors at distance 1
        for pos in range(n):
            original_val = word[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                neighbor = word.copy()
                neighbor[pos] = new_val
                idx = list_to_idx(neighbor, q)
                ball_set.add(idx)
        balls.append(sorted(ball_set))
        
    # Convert balls to numpy array for vectorized operations
    # Shape: (total_words, ball_size)
    ball_size = len(balls[0])
    balls_np = np.array(balls, dtype=np.int32)

    # Initialize uncovered mask
    uncovered_mask = np.ones(total_words, dtype=bool)
    code = []

    # Greedy Phase
    # We want to pick the word that covers the most uncovered points.
    # Coverage of word w = sum(uncovered_mask[balls_np[w]])
    
    while np.any(uncovered_mask):
        if time.time() > deadline:
            break
            
        # Calculate coverage for all words
        # This creates a (total_words, ball_size) boolean array, which is large (2187 * 15 ~ 32k bools)
        # But summing along axis 1 is fast.
        # To save memory/time, we can just index the mask.
        # covered_counts = uncovered_mask[balls_np].sum(axis=1)
        
        # Optimization: Only check words that are currently uncovered?
        # Actually, a word that is already covered might still cover many *other* uncovered words.
        # However, in standard greedy set cover, we usually pick from the universe.
        # Here the "sets" are the balls. The "elements" are the words.
        # We can pick any word from the full space as a codeword.
        
        covered_counts = uncovered_mask[balls_np].sum(axis=1)
        best_word = np.argmax(covered_counts)
        
        if covered_counts[best_word] == 0:
            # No word covers any uncovered element? Should not happen if uncovered_mask has True.
            # Unless radius is 0 and we are in a weird state.
            break
            
        code.append(int(best_word))
        
        # Update uncovered mask
        uncovered_mask[balls_np[best_word]] = False
        
    # Local Optimization Phase: Remove redundant codewords
    # A codeword c is redundant if every point in its ball is covered by at least one other codeword.
    # We can maintain a count of how many codewords cover each point.
    # If count[p] > 1 for all p in ball(c), then c is removable.
    
    if time.time() < deadline - 5.0: # Need time for optimization
        # Initialize coverage counts
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_counts[balls_np[c]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove from the end to avoid index shifting issues, or iterate carefully
            # Iterating from end is safer for popping.
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any point in ball(c) has cover_counts == 1
                # If so, removing c would uncover that point.
                # If all points in ball(c) have cover_counts > 1, c is redundant.
                ball_points = balls_np[c]
                counts_in_ball = cover_counts[ball_points]
                
                if np.all(counts_in_ball > 1):
                    # Safe to remove
                    cover_counts[ball_points] -= 1
                    code.pop(i)
                    changed = True
        
    # Convert indices back to list of lists of integers
    result = [idx_to_list(c, q, n) for c in code]
    
    # Ensure distinctness (greedy ensures this, but just in case)
    seen = set()
    unique_result = []
    for r in result:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique_result.append(r)
            
    return unique_result

# EVOLVE-BLOCK-END
