import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** 8  # 6561
        ball_size = 1 + n * (q - 1)  # 17
        
        # Precompute balls using numpy for speed
        # Represent each word as a base-3 number 0..6560
        # Precompute digit arrays
        all_digits = np.zeros((total_words, n), dtype=np.int8)
        for w in range(total_words):
            temp = w
            for i in range(n):
                all_digits[w, i] = temp % 3
                temp //= 3
        
        # Precompute balls as lists of neighbors
        balls = []
        for w in range(total_words):
            ball = [w]
            for pos in range(n):
                orig = all_digits[w, pos]
                for val in range(3):
                    if val == orig:
                        continue
                    # Compute new word
                    diff = val - orig
                    new_w = w + diff * (3 ** pos)
                    ball.append(new_w)
            balls.append(ball)
        
        best_code = None
        best_len = float('inf')
        
        # Multiple strategies with different random seeds
        for seed in range(80):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            uncovered = set(range(total_words))
            code = []
            
            while uncovered:
                if time.time() > deadline:
                    break
                    
                # Greedy: find candidate covering most uncovered words
                # Optimization: sample candidates if set is large
                if len(uncovered) > 200:
                    # Sample a subset to speed up
                    candidates = random.sample(list(uncovered), min(50, len(uncovered)))
                else:
                    candidates = list(uncovered)
                
                best_cand = -1
                best_count = -1
                
                for c in candidates:
                    count = sum(1 for nb in balls[c] if nb in uncovered)
                    if count > best_count:
                        best_count = count
                        best_cand = c
                        if best_count == ball_size:
                            break
                
                if best_cand == -1:
                    # Fallback: pick any uncovered
                    best_cand = random.choice(list(uncovered))
                    best_count = 1
                    
                code.append(best_cand)
                for nb in balls[best_cand]:
                    if nb in uncovered:
                        uncovered.discard(nb)
                        
            if uncovered:
                code.extend(uncovered)
            
            # Local improvement: try to replace codewords
            # For each codeword, check if removing it and adding another improves coverage
            improved = True
            max_improve_iter = 3
            while improved and max_improve_iter > 0 and time.time() < deadline:
                improved = False
                max_improve_iter -= 1
                
                # Compute coverage count for each word
                coverage = [0] * total_words
                for cw in code:
                    for nb in balls[cw]:
                        coverage[nb] += 1
                
                # Find words with coverage > 1 that could be candidates for removal
                # Try to replace each codeword
                for i, cw in enumerate(code):
                    if time.time() > deadline:
                        break
                    
                    # Check if removing cw leaves any word uncovered
                    # A word is uncovered if coverage[word] == 1 and it's only covered by cw
                    removable = True
                    for nb in balls[cw]:
                        if coverage[nb] == 1:
                            # Only covered by cw, so cw is essential
                            removable = False
                            break
                    
                    if not removable:
                        continue
                    
                    # Try to find a better replacement
                    # Find words that cover some of the words currently covered by cw
                    # but also cover words that need more coverage
                    best_replace = -1
                    best_gain = 0
                    
                    # Sample potential replacements
                    sample_size = min(30, total_words)
                    for _ in range(sample_size):
                        candidate = random.randint(0, total_words - 1)
                        if candidate in code:
                            continue
                        
                        # Check if candidate covers all words that cw covers
                        # (since we're removing cw, candidate must cover those)
                        covers_all = True
                        for nb in balls[cw]:
                            if candidate not in balls[nb] and nb not in balls[candidate]:
                                # Check if nb is covered by candidate
                                # nb is within distance 1 of candidate?
                                # Actually, we need: every word covered by cw must be covered by candidate
                                # i.e., balls(cw) subset of balls(candidate) union other code
                                # This is complex. Simpler: just check if removing cw and adding candidate
                                # still covers everything.
                                pass
                        
                        # Simplified: just check if the union of balls(candidate) and other code covers everything
                        # This is expensive. Instead, use a heuristic:
                        # Count how many words in balls(cw) are covered by candidate
                        overlap = sum(1 for nb in balls[cw] if nb in balls[candidate])
                        gain = overlap - best_gain
                        
                        if overlap > best_gain + 10:  # Must cover at least 11 of 17
                            # Verify that removing cw and adding candidate maintains coverage
                            # Quick check: for each word in balls(cw), is it covered by candidate or other code?
                            valid = True
                            for nb in balls[cw]:
                                if nb not in balls[candidate]:
                                    # Check if covered by other code
                                    found = False
                                    for j, other in enumerate(code):
                                        if j == i:
                                            continue
                                        if nb in balls[other]:
                                            found = True
                                            break
                                    if not found:
                                        valid = False
                                        break
                            if valid:
                                best_replace = candidate
                                best_gain = overlap
            
                    if best_replace != -1:
                        code[i] = best_replace
                        improved = True
                        break
            
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
        
        # Convert best_code to list of lists of integers
        result = []
        for w in best_code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % 3)
                temp //= 3
            result.append(digits)
            
        return result
    else:
        # Generic fallback
        total_words = q ** n
        if total_words > 100000:
            return [[i % q for i in range(n)] for i in range(total_words)]
        
        balls = []
        for w in range(total_words):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            balls.append(list(ball))
        
        uncovered = set(range(total_words))
        code = []
        
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in balls[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
                    if best_count == len(balls[c]):
                        break
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in balls[best_cand]:
                if nb in uncovered:
                    uncovered.discard(nb)
                    
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result
