import itertools

def construct(q, n, radius):
    # This solution is specialized for q=3, n=8, radius=1
    # but should handle general inputs if possible, or at least not crash.
    # Given the task contract, q, n, radius are passed.
    # The specific problem is K_3(8,1).
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a simple construction if parameters differ
        # e.g., direct sum or repetition
        # But the task says "evaluator passes this task's alphabet size q, length n and covering radius"
        # implying it will be 3, 8, 1.
        # We can implement a generic greedy for small spaces, or just return the specialized one.
        # Let's assume the evaluator sticks to the problem description.
        pass

    # Specialized for 3, 8, 1
    total_words = 3 ** n
    # Precompute neighbors
    # Map int to base-3 tuple is useful for generation, but we can work with ints.
    # To generate neighbors of an int w:
    # We need to change one digit.
    # It is faster to precompute the "step" values.
    # For position i, the value is 3**i.
    # If digit d is 0, neighbors are d+1, d+2.
    # If digit d is 1, neighbors are d+1 (2), d-1 (0).
    # If digit d is 2, neighbors are d-1 (1), d-2 (0).
    
    # Let's precompute neighbors for all 6561 words.
    neighbors = [[] for _ in range(total_words)]
    powers = [3**i for i in range(n)]
    
    for w in range(total_words):
        # Get digits
        digits = [0]*n
        tmp = w
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        
        neighbors[w].append(w)
        for i in range(n):
            d = digits[i]
            p = powers[i]
            # Change d to (d+1)%3
            new_d = (d + 1) % 3
            new_w = w + (new_d - d) * p
            neighbors[w].append(new_w)
            # Change d to (d-1)%3
            new_d = (d - 1) % 3
            new_w = w + (new_d - d) * p
            neighbors[w].append(new_w)
            
    covered = [False] * total_words
    code = []
    uncovered_count = total_words
    
    while uncovered_count > 0:
        best_word = -1
        best_count = 0
        for w in range(total_words):
            # Optimization: if the current best count is already 17 (max possible), break?
            # Max count is 17.
            if best_count == 17:
                break
            cnt = 0
            for nb in neighbors[w]:
                if not covered[nb]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        # Add best_word to code
        # Convert best_word to list of ints
        digits = [0]*n
        tmp = best_word
        for i in range(n):
            digits[i] = tmp % 3
            tmp //= 3
        code.append(digits)
        
        # Mark covered
        for nb in neighbors[best_word]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
                
    return code
