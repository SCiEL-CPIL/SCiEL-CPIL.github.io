import time
import random
import itertools

def construct(q, n, radius):
    # This task is specifically for q=3, n=8, radius=1
    # But the interface is general. We should handle the general case if possible, 
    # or at least not crash, though the evaluator passes the specific task params.
    # The prompt says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # So q=3, n=8, radius=1.
    
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a simple construction for other parameters to avoid errors
        # Direct sum of a small code if possible, or random
        # For this specific task, we assume q=3, n=8, radius=1
        # If not, we return a trivial upper bound (all words) or a simple greedy
        # Let's implement a generic simple greedy for robustness, though it might be slow for large n.
        # Given the context, the evaluator likely only tests the specific case.
        pass

    total_words = q ** n
    if total_words > 10000:
        # Fallback for larger spaces to prevent timeout
        # Use the seed's direct sum logic if applicable or random
        # For now, just return a random sample or all words if small
        # But the task is fixed. We will focus on q=3, n=8.
        pass

    # Specific implementation for q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        # Precompute balls
        # Word to list of neighbors (including self)
        # Represent words as base-3 integers
        balls = []
        for w in range(total_words):
            ball = [w]
            # Convert to list of digits
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % 3
                temp //= 3
            
            for pos in range(n):
                original = digits[pos]
                for val in range(3):
                    if val == original:
                        continue
                    digits[pos] = val
                    # Convert back to int
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= 3
                    ball.append(new_w)
                digits[pos] = original
            balls.append(ball)
        
        best_code = None
        best_len = float('inf')
        
        # Run multiple randomizations
        for seed in range(50):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            uncovered = set(range(total_words))
            code = []
            
            while uncovered:
                if time.time() > deadline:
                    break
                    
                # Candidates are uncovered words
                # To speed up, we can sample if the set is large, but 6561 is small.
                # We want to find the candidate that covers the most uncovered words.
                best_cand = -1
                best_count = -1
                
                # Iterate through uncovered words
                # Optimization: list(uncovered) creates a copy. 
                # We can iterate directly.
                for c in uncovered:
                    # Count how many neighbors of c are in uncovered
                    # Since c is in uncovered, count >= 1
                    count = 0
                    for nb in balls[c]:
                        if nb in uncovered:
                            count += 1
                            
                    if count > best_count:
                        best_count = count
                        best_cand = c
                        
                    # Early stopping optimization? 
                    # Max possible is len(balls[c]) which is 17.
                    # If best_count == 17, we can't do better.
                    if best_count == 17:
                        break
                        
                if best_cand == -1:
                    # Should not happen if uncovered is not empty
                    break
                    
                code.append(best_cand)
                # Remove covered words
                for nb in balls[best_cand]:
                    if nb in uncovered:
                        uncovered.remove(nb)
                        
            if uncovered:
                # If we hit deadline, the code is incomplete. 
                # We must ensure validity. 
                # If we break early, we need to cover the rest.
                # Greedy on the remaining uncovered words (ignoring the "best" part, just pick any)
                # Or just add all uncovered words to the code to ensure validity.
                # Adding all uncovered words is safe but increases size.
                # Since we break on deadline, we should finish quickly.
                # Let's add all remaining uncovered words.
                code.extend(uncovered)
                
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
        # Generic fallback for other parameters
        # Simple greedy
        # ... (implementation similar to above but generic)
        # For the purpose of this specific task, the above block is sufficient.
        # However, to be safe, let's implement a generic version.
        
        total_words = q ** n
        if total_words > 100000:
            # Too large for simple greedy in 120s
            # Return random sample? No, must be valid.
            # Return all words? Size is q^n.
            # This case is unlikely in the evaluator.
            return [[i % q for i in range(n)] for i in range(total_words)]
            
        # Precompute balls
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
                count = 0
                for nb in balls[c]:
                    if nb in uncovered:
                        count += 1
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
                    uncovered.remove(nb)
                    
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
