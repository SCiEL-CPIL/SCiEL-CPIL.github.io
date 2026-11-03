import time
import numpy as np
import random

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    
    if radius == 0:
        return [list(np.array_divmod(i, q)) for i in range(total)] # Placeholder, needs proper conversion

    # Precompute neighbors for radius 1
    # Each index is an integer 0..total-1
    # Convert to base-q digits
    # Neighbor is index +/- q^i for each position i, if digit allows it.
    
    # We will use a greedy set cover approach.
    # 1. Create list of all possible codewords (indices).
    # 2. Precompute the set of points covered by each codeword.
    
    # For n=6, q=3, total=729.
    # Radius 1 means the codeword covers itself and words at distance 1.
    # Number of neighbors = 1 + n*(q-1) = 1 + 6*2 = 13.
    
    neighbors = []
    for i in range(total):
        word = [0] * n
        tmp = i
        for k in range(n):
            word[k] = tmp % q
            tmp //= q
        
        covered = set()
        covered.add(i)
        
        # Generate neighbors
        for pos in range(n):
            for new_digit in range(q):
                if new_digit == word[pos]:
                    continue
                # Calculate new index
                # index = sum(d_j * q^j)
                # diff = (new_digit - word[pos]) * q^pos
                new_idx = i + (new_digit - word[pos]) * (q ** pos)
                covered.add(new_idx)
        
        neighbors.append(covered)

    best_code = None
    best_len = total + 1
    
    # Run multiple randomized greedy instances
    # Use different random shuffles of the candidate order
    # Or just randomize the tie-breaking.
    
    # To speed up, we can use a priority queue or just scan.
    # Scanning 729 items is fast.
    
    for run in range(50): # 50 restarts should be fast enough
        if time.monotonic() > deadline:
            break
            
        rng = random.Random(run)
        
        # Initialize coverage counts
        # We track which points are covered.
        # covered_mask: boolean array of size total
        covered_mask = np.zeros(total, dtype=bool)
        
        code = []
        
        # To make it deterministic but varied, we can shuffle the order in which we consider candidates
        # or just pick the best one. If multiple have same gain, pick random among them.
        
        remaining_candidates = list(range(total))
        rng.shuffle(remaining_candidates)
        
        # We need to efficiently find the candidate that covers the most uncovered points.
        # gain[c] = sum(1 for u in neighbors[c] if not covered_mask[u])
        
        # Initial gains
        gains = np.zeros(total, dtype=int)
        for c in range(total):
            gains[c] = len(neighbors[c]) # Initially all uncovered
        
        while True:
            # Check if all covered
            if np.all(covered_mask):
                break
                
            # Find candidate with max gain
            # If we want randomness, we can perturb gains slightly or pick randomly from top-k.
            # Let's just pick max, and if tie, pick first in shuffled order?
            # Better: pick max gain. If tie, pick random from tied.
            
            max_gain = -1
            best_candidates = []
            
            # Iterate through remaining candidates? 
            # Actually, we can just iterate through all, and if gain is 0, skip.
            # But we should remove candidates that are already "used"? No, set cover allows reuse? 
            # No, codewords must be distinct. So once picked, remove from candidates.
            
            # Optimization: Maintain a list of active candidates.
            # But checking 729 is cheap.
            
            for c in remaining_candidates:
                if gains[c] == 0:
                    continue
                if gains[c] > max_gain:
                    max_gain = gains[c]
                    best_candidates = [c]
                elif gains[c] == max_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                # Should not happen if not all covered
                break
                
            # Pick one
            c = rng.choice(best_candidates)
            code.append(c)
            
            # Update coverage
            for u in neighbors[c]:
                if not covered_mask[u]:
                    covered_mask[u] = True
                    # Update gains for all candidates that cover u
                    # This is expensive: for each u, check all candidates covering u.
                    # Precompute: for each u, list of candidates covering u.
                    # This is the dual of neighbors.
                    # Since we are in a loop, let's just recalculate gains for affected candidates?
                    # Or just recalculate gains for ALL candidates? 
                    # 729 candidates * 13 neighbors = ~9k ops per step.
                    # 73 steps * 9k = 650k ops. Very fast.
                    pass
            
            # Recalculate gains for simplicity and correctness
            # gains[c] = count of uncovered in neighbors[c]
            for c in remaining_candidates:
                g = 0
                for u in neighbors[c]:
                    if not covered_mask[u]:
                        g += 1
                gains[c] = g
            
            # Remove c from remaining_candidates? 
            # We can just set gains[c] = 0 or leave it and skip if 0.
            # But we must not pick it again.
            # We can mark it as used.
            gains[c] = -1 # Mark as used/invalid
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
            # Optional: Local Search / Pruning
            # Try to remove redundant codewords
            # A codeword c is redundant if all points in neighbors[c] are covered by other codewords in code.
            # We can check this by counting coverage.
            
            if best_len <= 80: # Only try if we are close
                # Count coverage from all codewords
                cover_count = np.zeros(total, dtype=int)
                for c in best_code:
                    for u in neighbors[c]:
                        cover_count[u] += 1
                
                # Try removing each
                new_code = best_code[:]
                for i in range(len(new_code)):
                    c = new_code[i]
                    # Check if any u in neighbors[c] has count 1 (only covered by c)
                    redundant = True
                    for u in neighbors[c]:
                        if cover_count[u] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove c
                        new_code.pop(i)
                        # Update cover_count
                        for u in neighbors[c]:
                            cover_count[u] -= 1
                        # Reset i to check previous again? No, just continue.
                        # Actually, removing one might make another redundant.
                        # Simple pass is enough for now.
                        i = -1 # Restart loop to catch cascading redundancies? 
                        # This might be slow if many. But code is small.
                    
                if len(new_code) < best_len:
                    best_len = len(new_code)
                    best_code = new_code

    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        result.append(word)
        
    return result
