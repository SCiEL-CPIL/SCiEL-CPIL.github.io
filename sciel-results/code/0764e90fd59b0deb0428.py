import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls of radius 1
    # ball[i] is a list of indices of words within distance 1 of all_words[i]
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    covered = [False] * N
    code = []
    
    # Greedy construction
    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        if all(covered):
            break
            
        # Find best candidate
        best_count = -1
        best_idx = -1
        
        for i in range(N):
            # Optional: skip if already in code? No, codewords are distinct, but we can pick any word.
            # Actually, picking a word that is already covered is fine, it might cover more uncovered words.
            count = 0
            for j in balls[i]:
                if not covered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.append(all_words[best_idx])
        for j in balls[best_idx]:
            covered[j] = True
            
    # Post-processing: try to remove redundant codewords
    # We might be able to remove some codewords if their coverage is redundant
    # We iterate until no more can be removed
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            # Tentatively remove code[i]
            c = code[i]
            idx_c = word_to_idx[c]
            
            # Check if removing c leaves any word uncovered
            # A word w is uncovered if its only covering codeword was c
            # This is expensive to check directly.
            # Alternative: maintain coverage counts?
            # Let's just check coverage from scratch for the remaining code?
            # That is O(|C| * N * radius_cost). |C| ~ 70, N=729. 70*729*13 ~ 660k.
            # Doing this for each of 70 words is 46M ops. Might be tight in 120s in Python.
            # Let's optimize: 
            # We can check if there is any word in ball(c) that is NOT covered by any other codeword in C\{c}.
            
            # For efficiency, we can precompute coverage counts?
            # Let's try a simpler check: 
            # For each word w in ball(c):
            #   is there another c' in C\{c} such that dist(w, c') <= radius?
            # If for all w in ball(c) there is such a c', then c is redundant.
            
            redundant = True
            for w_idx in balls[idx_c]:
                w = all_words[w_idx]
                # Check if w is covered by any other codeword
                covered_by_other = False
                for j in range(len(code)):
                    if j == i:
                        continue
                    c_other = code[j]
                    # Calculate distance
                    dist = 0
                    for k in range(n):
                        if w[k] != c_other[k]:
                            dist += 1
                            if dist > radius:
                                break
                    if dist <= radius:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                code.pop(i)
                changed = True
                break # Restart the loop as indices changed
                
    return code
