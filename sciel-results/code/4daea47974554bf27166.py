import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precompute the space
    N = q ** n
    # Map integer index to list of q digits
    # And list of digits to integer index
    
    # We work with integers 0..N-1 representing base-q numbers
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        return word

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute balls of radius 'radius'
    # For q=3, n=6, radius=1, ball size is 1 + 6*(3-1) = 13
    # We store the list of indices covered by each index
    balls = []
    for i in range(N):
        w = idx_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                ball.append(word_to_idx(w))
            w[pos] = orig
        balls.append(ball)
        
    # Greedy Covering Code Construction
    def greedy_construct(seed):
        rng = random.Random(seed)
        covered = [False] * N
        code = []
        
        # To speed up, we can maintain a list of uncovered counts per candidate?
        # Or just iterate all candidates each time. N=729 is small.
        
        # Optimization: Keep track of how many times each word is "needed" or just 
        # calculate gain on the fly.
        
        # We want to pick the word that covers the most UNCOVERED words.
        
        # Initial shuffle to break symmetry
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        while True:
            # Check if all covered
            if all(covered):
                break
                
            best_candidate = -1
            best_gain = -1
            best_ball = None
            
            # Iterate through all candidates
            # We can skip candidates that are already in the code? 
            # No, adding a word already in the code is useless.
            # We should only consider words NOT in the code.
            
            # However, checking "not in code" for all N is O(N^2) total.
            # With N=729 and ~73 steps, 73*729 is small.
            
            for c in candidates:
                if c in code:
                    continue
                
                gain = 0
                for w in balls[c]:
                    if not covered[w]:
                        gain += 1
                        # Early exit if gain can't beat best?
                        # Max possible gain is 13.
                        if gain > best_gain:
                            # We can't break inner loop easily if we want exact gain 
                            # but we can break if gain > best_gain and we know max is 13?
                            # No, we need to know if it's strictly better.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    best_ball = balls[c]
                    
                    # If we found a perfect gain (all its neighbors were uncovered),
                    # we can't get better than ball_size.
                    if best_gain == len(best_ball):
                        break
                        
            if best_candidate == -1:
                # Should not happen if space is finite and we haven't covered all
                break
                
            # Add best candidate
            code.append(best_candidate)
            for w in best_ball:
                covered[w] = True
                
        return code

    def optimize(code):
        # Try to remove redundant codewords
        # A codeword c is redundant if its ball is a subset of the union of balls of other codewords.
        
        # Convert to set for faster lookup?
        # code_set = set(code)
        
        # Calculate coverage counts
        # count[w] = number of codewords covering w
        counts = [0] * N
        for c in code:
            for w in balls[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(code):
                # Check if any word in balls[c] has count == 1
                # If so, c is NOT redundant
                # If all words in balls[c] have count > 1, c IS redundant
                is_redundant = True
                for w in balls[c]:
                    if counts[w] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    for w in balls[c]:
                        counts[w] -= 1
                    code.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        return code

    # Run multiple random greedy constructions
    best_code = None
    best_len = float('inf')
    
    num_restarts = 50 # Adjust based on speed
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        code = greedy_construct(i)
        code = optimize(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
        # If we hit the theoretical lower bound or known best, we can stop early?
        # Known best is 73. Lower bound 71.
        # If we find 73, we can try harder for 71/72, but 73 is a great score.
        if best_len <= 73:
             # We found a very good code. Let's do some local search swaps to see if we can go lower.
             # But for now, 73 is a solid win over 81.
             pass

    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to words
    return [idx_to_word(c) for c in best_code]
