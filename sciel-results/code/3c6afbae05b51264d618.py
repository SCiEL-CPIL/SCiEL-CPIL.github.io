import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute balls
    balls = []
    for w_int in range(total_words):
        w_digits = [0] * n
        temp = w_int
        for i in range(n):
            w_digits[i] = temp % q
            temp //= q
        
        ball = set()
        ball.add(w_int)
        for i in range(n):
            for val in range(q):
                if val == w_digits[i]:
                    continue
                new_w = w_digits.copy()
                new_w[i] = val
                new_int = 0
                for j in range(n):
                    new_int = new_int * q + new_w[j]
                ball.add(new_int)
        balls.append(ball)
        
    covered_count = np.zeros(total_words, dtype=np.int32)
    code = []
    code_set = set()
    
    def is_covered():
        return np.all(covered_count > 0)
        
    # Greedy Phase
    while not is_covered():
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_word = -1
        
        for w in range(total_words):
            if w in code_set:
                continue
            
            gain = 0
            for v in balls[w]:
                if covered_count[v] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word == -1:
            break
            
        code.append(best_word)
        code_set.add(best_word)
        
        for v in balls[best_word]:
            covered_count[v] += 1

    # Removal Phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if removable
            removable = True
            for v in balls[c]:
                if covered_count[v] == 1:
                    removable = False
                    break
            
            if removable:
                for v in balls[c]:
                    covered_count[v] -= 1
                code.pop(i)
                code_set.remove(c)
                improved = True
                
            if time.time() > deadline:
                break

    # Swap Phase
    if time.time() < deadline:
        swap_improved = True
        # Limit swap passes to avoid running out of time
        for _ in range(5): 
            if time.time() > deadline:
                break
            swap_improved = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    break
                c = code[i]
                # Temporarily remove c
                for v in balls[c]:
                    covered_count[v] -= 1
                
                # Find holes
                holes = [v for v in range(total_words) if covered_count[v] == 0]
                
                if not holes:
                    code.pop(i)
                    code_set.remove(c)
                    swap_improved = True
                    continue
                
                hole_set = set(holes)
                best_w = -1
                best_cover = -1
                target_cover = len(holes)
                
                for w in range(total_words):
                    if w in code_set:
                        continue
                    cover = 0
                    for v in balls[w]:
                        if v in hole_set:
                            cover += 1
                    if cover > best_cover:
                        best_cover = cover
                        best_w = w
                        if cover == target_cover:
                            break
                
                if best_w != -1 and best_cover == target_cover:
                    code_set.remove(c)
                    code_set.add(best_w)
                    code[i] = best_w
                    for v in balls[best_w]:
                        covered_count[v] += 1
                    swap_improved = True
                else:
                    for v in balls[c]:
                        covered_count[v] += 1
        
    # Convert to required format
    result = []
    for w in code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        result.append(digits)
        
    return result
