import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Specific to the task: q=3, n=7, radius=1
    # But the function signature is generic. We handle the specific case efficiently.
    # If q, n, radius are different, we fall back to a simple greedy or return seed-like.
    # The task states: "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And "Score: the number of codewords...".
    # We assume the evaluator calls with q=3, n=7, radius=1.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: simple greedy
        # For now, just return a valid cover using a simple method if params differ
        # But the task is specific. Let's assume it's always 3,7,1.
        # To be safe, implement a generic slow path or just handle 3,7,1.
        # Given the prompt "Football pool problem for 7 matches", it's fixed.
        pass

    M = q ** n  # 2187
    # Precompute balls for radius 1
    # Each ball has 1 + n*(q-1) words
    ball_size = 1 + n * (q - 1)  # 1 + 7*2 = 15? No, radius 1 in Hamming distance.
    # Wait, radius 1 means distance <= 1.
    # Center: 1 word.
    # Distance 1: n positions * (q-1) values.
    # Total 1 + 7*2 = 15 words per ball.
    
    # Generate all words
    words = np.array(list(itertools.product(range(q), repeat=n)))
    
    # Map word tuple to index
    # Since M is small, we can just use the index in the `words` array if we generate them systematically.
    # `itertools.product` generates in lexicographic order, so index is base-q value.
    # We can map a word to its index by converting the base-q tuple to an integer.
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute balls as list of lists of indices
    balls = []
    for i in range(M):
        w = words[i]
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                ball.append(word_to_idx(tuple(new_w)))
        balls.append(ball)
        
    balls_arr = np.array(balls) # Shape (M, ball_size)
    
    best_codewords = None
    best_size = M # Upper bound is M (all words)
    
    # Run multiple restarts of Greedy + Redundancy Removal
    restarts = 50
    rng = np.random.default_rng(42)
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        covered = np.zeros(M, dtype=bool)
        selected = []
        
        # Greedy Phase
        for step in range(M):
            if time.time() > deadline:
                break
            
            # Calculate gain for each candidate
            # gain[i] = number of words in balls[i] that are NOT covered
            # ~covered is boolean array
            # balls_arr is (M, ball_size)
            # (~covered)[balls_arr] is (M, ball_size) boolean
            # .sum(axis=1) is (M,)
            
            # Optimization: If all covered, break
            if covered.all():
                break
                
            gains = (~covered)[balls_arr].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one from candidates to encourage diversity
            best_idx = rng.choice(candidates)
            
            selected.append(best_idx)
            covered[balls_arr[best_idx]] = True
            
        # If not fully covered, this restart failed (should not happen with M candidates)
        if not covered.all():
            continue
            
        # Redundancy Removal Phase
        # Convert selected to set for O(1) lookup if needed, but we work with list
        # We need to check if removing a codeword breaks coverage.
        # Maintain counts: how many selected codewords cover each word.
        
        counts = np.zeros(M, dtype=int)
        for c in selected:
            counts[balls_arr[c]] += 1
            
        improved = True
        while improved:
            improved = False
            new_selected = []
            for c in selected:
                if time.time() > deadline:
                    break
                
                # Tentatively remove c
                counts[balls_arr[c]] -= 1
                
                # Check if any word has count 0
                if np.min(counts) > 0:
                    # c was redundant, do not add back
                    improved = True
                else:
                    # c is essential, add back
                    counts[balls_arr[c]] += 1
                    new_selected.append(c)
            
            if time.time() > deadline:
                break
                
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_codewords = [words[c].tolist() for c in selected]
            
    if best_codewords is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_codewords
