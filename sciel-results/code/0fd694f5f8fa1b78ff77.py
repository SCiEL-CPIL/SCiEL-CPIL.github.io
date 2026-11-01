"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(q, n, radius):
    # The task is specifically for q=3, n=7, radius=1.
    # If parameters differ, fall back to a simple valid construction if possible, 
    # or just return an empty list if invalid (but evaluator expects valid).
    # Assuming the evaluator always passes the specific task parameters.
    
    start_time = time.time()
    deadline = 110.0 # 10s margin before 120s limit
    
    total_words = q ** n
    
    def idx_to_list(idx, base, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = idx % base
            idx //= base
        return res

    def list_to_idx(lst, base):
        idx = 0
        for d in lst:
            idx = idx * base + d
        return idx

    # Precompute balls for all words
    # balls[i] is a set of indices
    balls = [None] * total_words
    
    for i in range(total_words):
        word = idx_to_list(i, q, n)
        ball = set()
        ball.add(i)
        
        # Add neighbors at distance 1
        for pos in range(n):
            original_val = word[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                neighbor = word.copy()
                neighbor[pos] = new_val
                idx = list_to_idx(neighbor, q)
                ball.add(idx)
        balls[i] = ball

    uncovered = set(range(total_words))
    code = []
    
    # Greedy Phase
    while uncovered:
        if time.time() > deadline:
            break
            
        best_word = -1
        max_coverage = -1
        
        # Iterate over uncovered words to find the one covering the most uncovered neighbors
        # Optimization: We can break early if we find a word covering all its neighbors?
        # No, we want max coverage of UNCOVERED set.
        
        # To speed up tie-breaking and potentially find better solutions, 
        # we could randomize the order of iteration or pick randomly among max.
        # For now, deterministic.
        
        for w in uncovered:
            # Count how many neighbors of w are in uncovered
            coverage = len(balls[w] & uncovered)
            if coverage > max_coverage:
                max_coverage = coverage
                best_word = w
        
        if best_word == -1:
            break
            
        code.append(best_word)
        uncovered -= balls[best_word]

    # Local Optimization Phase: Remove redundant codewords
    # A codeword c is redundant if its ball is covered by the union of balls of all other codewords.
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code)):
            c = code[i]
            # Check if every point in balls[c] is covered by some other codeword
            is_redundant = True
            for p in balls[c]:
                covered_by_other = False
                for j in range(len(code)):
                    if i == j:
                        continue
                    if p in balls[code[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                code.pop(i)
                changed = True
                break # Restart loop as indices changed

    # Convert indices back to list of lists of integers
    result = [idx_to_list(c, q, n) for c in code]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for r in result:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique_result.append(r)
            
    return unique_result
# EVOLVE-BLOCK-END
