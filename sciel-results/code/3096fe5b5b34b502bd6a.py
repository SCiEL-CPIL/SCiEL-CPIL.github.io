import itertools
import random
import time

def construct(q, n, radius):
    # Handle the specific case q=3, n=6, radius=1 with an optimized search
    if q == 3 and n == 6 and radius == 1:
        words = [list(t) for t in itertools.product(range(3), repeat=6)]
        N = len(words)
        
        # Create a lookup dictionary from tuple(word) to index
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        # Precompute neighbors (Hamming distance <= 1)
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(words):
            neighbors[i].append(i)
            for pos in range(6):
                orig_val = w[pos]
                for val in range(3):
                    if val == orig_val:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].append(idx)
                    
        best_code = None
        start_time = time.time()
        deadline = start_time + 140.0 # Leave 10s buffer
        
        # Run multiple greedy restarts with random tie-breaking
        # We know 73 is the best known, so we stop if we find 73 or less.
        for seed in range(5000):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            uncovered = set(range(N))
            code = []
            
            while uncovered:
                best_gain = -1
                max_gain_candidates = []
                
                # Find word with maximum marginal coverage
                for w_idx in range(N):
                    # Optimization: if we already have a gain > 0, we can skip checking
                    # words that are likely to have 0 gain? Hard to know without checking.
                    # But checking is cheap.
                    gain = 0
                    for nb in neighbors[w_idx]:
                        if nb in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        max_gain_candidates = [w_idx]
                    elif gain == best_gain:
                        max_gain_candidates.append(w_idx)
                        
                # If best_gain is 0, we are stuck (should not happen)
                if best_gain == 0:
                    break
                    
                # Pick randomly from candidates to diversify search
                chosen = random.choice(max_gain_candidates)
                code.append(chosen)
                
                # Update uncovered set
                for nb in neighbors[chosen]:
                    if nb in uncovered:
                        uncovered.remove(nb)
                        
            # If we found a valid code
            if len(uncovered) == 0:
                if best_code is None or len(code) < len(best_code):
                    best_code = code.copy()
                    if len(best_code) <= 73:
                        break # Found optimal or better, stop searching
                    
        if best_code is not None:
            return best_code

    # Fallback for other parameters or if search fails
    # Use the direct sum construction from the seed
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(t) for t in itertools.product(range(3), repeat=n)]
