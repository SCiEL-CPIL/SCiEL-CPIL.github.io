import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    N = 3**7
    words = []
    for i in range(N):
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % 3)
            tmp //= 3
        words.append(digits)

    # Precompute ball masks
    # Using a list of lists for faster indexing in Python loops
    # ball_matrix[i] is a set of indices covered by word i
    ball_sets = []
    for i in range(N):
        w = words[i]
        s = {i}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                new_idx = 0
                power = 1
                for d in new_w:
                    new_idx += d * power
                    power *= 3
                s.add(new_idx)
        ball_sets.append(s)
    
    # Convert to numpy arrays for faster operations if needed, or just use sets
    # Let's stick to sets for the "covered" logic to avoid numpy overhead in tight loops if possible?
    # Actually, numpy is good for the "gain" calculation.
    # Let's create a numpy matrix for gain calculation.
    ball_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in ball_sets[i]:
            ball_matrix[i, j] = 1

    # Greedy Set Cover
    def greedy_randomized():
        covered = np.zeros(N, dtype=bool)
        code_indices = []
        
        while not covered.all():
            # Calculate gains
            # Sum of ball_matrix rows over uncovered indices
            # This is equivalent to: for each row i, count how many j in ball_sets[i] are not in covered
            # Using numpy: ball_matrix[:, ~covered].sum(axis=1)
            gains = ball_matrix[:, ~covered].sum(axis=1)
            
            # Random tie-breaking
            noise = np.random.random(N) * 0.1
            gains_with_noise = gains + noise
            
            best_idx = np.argmax(gains_with_noise)
            
            if gains[best_idx] == 0:
                break
                
            code_indices.append(best_idx)
            covered |= ball_matrix[best_idx].astype(bool)
            
        return code_indices

    # Local Search / Polish
    def polish(code_indices):
        # Try to remove redundant codewords
        # A codeword c is redundant if every word in ball(c) is covered by other codewords in the set.
        # We iterate through the code and check for redundancy.
        # To speed up, we can check if the "unique" contribution of a codeword is empty.
        # Unique contribution = ball(c) \ union(ball(o) for o in code, o != c)
        
        # Optimization: Only check codewords that might be redundant.
        # In greedy, later added codewords are more likely to be redundant.
        # But let's check all.
        
        # We need to be careful with time.
        # Checking redundancy of one codeword takes O(|code| * 16) operations if we use sets.
        # With numpy: 
        # Union of others = OR of ball_matrix[o] for o in code, o != c
        # This is expensive if done naively for each c.
        
        # Alternative:
        # Maintain a count of how many codewords cover each word.
        # If a codeword c is removed, we subtract its ball from the counts.
        # If all counts in ball(c) are still > 0, then c is redundant.
        
        counts = np.zeros(N, dtype=np.int16)
        for c in code_indices:
            counts += ball_matrix[c]
            
        # Check redundancy
        # We can iterate through the code in reverse order (greedy added less useful ones last)
        # or random order.
        
        # To be efficient, let's just check each one.
        # If we remove one, we update counts.
        
        new_code = []
        # Sort by "gain" at time of addition? We don't have that.
        # Let's just iterate.
        
        # To make it faster, we can check if the minimum count in ball(c) is > 1.
        # min(counts[ball(c)]) > 1 implies redundant.
        
        # However, if we remove c, we must update counts.
        # Let's try a single pass of removal.
        
        removed_any = True
        while removed_any and time.time() < deadline: # Need deadline passed in or global?
            removed_any = False
            # Shuffle to avoid bias? Or just reverse?
            # Let's check in reverse order of addition (most likely redundant first)
            for i in range(len(code_indices) - 1, -1, -1):
                c = code_indices[i]
                # Check if c is redundant
                # Get the indices covered by c
                # mask = ball_matrix[c] == 1
                # If min(counts[mask]) > 1, then redundant.
                
                # Optimization: 
                # counts[ball_matrix[c] == 1].min()
                # This creates a boolean mask and indexes.
                # Might be fast enough.
                
                # Let's check if the minimum count in the ball is > 1
                # Note: ball_matrix[c] is 0/1.
                # We need to check counts at positions where ball_matrix[c] is 1.
                
                # This is a bit slow in Python loop over code_indices.
                # Let's try it.
                
                # Get the ball of c
                # ball_c = ball_matrix[c]
                # min_count = counts[ball_c == 1].min()
                
                # If min_count > 1, remove c.
                
                # To speed up, we can precompute the list of indices in the ball?
                # ball_indices[c] = list of indices
                # Then min(counts[j] for j in ball_indices[c])
                # This is 16 iterations in Python. 186 * 16 = ~3000 ops. Very fast.
                
                # Let's precompute ball_indices
                # Actually, we have ball_sets.
                
                min_cnt = min(counts[j] for j in ball_sets[c])
                
                if min_cnt > 1:
                    # Remove c
                    for j in ball_sets[c]:
                        counts[j] -= 1
                    code_indices.pop(i)
                    removed_any = True
                    # Break and restart loop to re-index? Or continue?
                    # If we continue, indices shift.
                    # Better to break and restart the while loop.
                    break
                    
        return code_indices

    # Baseline
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute ball_indices for polish
    # ball_sets is already a set of indices.
    
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized()
        
        # Polish
        # We need to pass deadline to polish or check inside
        # Let's modify polish to check time
        polished_indices = polish(code_indices)
        
        if len(polished_indices) < best_len:
            best_len = len(polished_indices)
            best_code = [words[i] for i in polished_indices]
            
        run_count += 1
        
    return best_code
