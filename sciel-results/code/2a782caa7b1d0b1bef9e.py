import itertools
import time
import random

def construct(q, n, radius):
    # For this specific task q=3, n=6, radius=1
    # But we should handle general case if possible, or at least not crash.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we must assume q=3, n=6, radius=1 for the optimization, but write code that works generally if possible.
    # However, the space size q**n grows exponentially. For q=3, n=6, it's 729.
    # If q or n are larger, this approach will fail.
    # The task description says "Football pool problem for 6 matches... K_3(6,1)".
    # So we can assume q=3, n=6, radius=1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the original direct sum construction for other parameters
        # This is just a safety net, the main logic is for 3,6,1
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback for smaller n, just brute force or simple
            return [list(x) for x in itertools.product(range(3), repeat=n)]

    total_words = 3 ** 6
    start_time = time.time()
    deadline = start_time + 110  # 10 sec margin
    
    # Map word tuple to index
    word_to_idx = {}
    idx_to_word = {}
    idx = 0
    for w in itertools.product(range(3), repeat=6):
        word_to_idx[w] = idx
        idx_to_word[idx] = w
        idx += 1
        
    # Precompute balls
    # ball[i] is a list of indices of words covered by codeword i
    balls = [None] * total_words
    for i in range(total_words):
        w = idx_to_word[i]
        covered = [i]
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                covered.append(word_to_idx[new_w])
        balls[i] = covered
        
    best_code = None
    best_size = total_words # Max possible
    
    # We will run several randomized greedy + local search iterations
    num_restarts = 2000 # We can do many fast iterations
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Construction
        uncovered = set(range(total_words))
        code = []
        
        # To randomize, we can shuffle the order in which we evaluate candidates
        # or pick randomly among the top few.
        
        while uncovered:
            # Find best candidate
            max_covered = 0
            best_candidates = []
            
            # Optimization: We only need to check candidates that cover at least one uncovered word.
            # But checking all 729 is fast enough.
            for c in range(total_words):
                # Count how many uncovered words c covers
                # This is sum(1 for w in balls[c] if w in uncovered)
                # Since balls[c] has 13 elements, this is fast.
                cnt = 0
                for w in balls[c]:
                    if w in uncovered:
                        cnt += 1
                if cnt > max_covered:
                    max_covered = cnt
                    best_candidates = [c]
                elif cnt == max_covered and cnt > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break # Should not happen if uncovered is not empty
                
            # Pick one randomly from best candidates
            c = random.choice(best_candidates)
            code.append(c)
            
            # Update uncovered
            for w in balls[c]:
                if w in uncovered:
                    uncovered.remove(w)
                    
        # Local Search: Prune redundant codewords
        # cover_count[w] = number of codewords in code covering w
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        improved = True
        while improved:
            improved = False
            # Try to remove any codeword
            for i in range(len(code)):
                c = code[i]
                can_remove = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    improved = True
                    break # Restart the loop to check others
                    
        # Check if better
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            if best_size == 71: # Theoretical lower bound
                break

    # Convert indices back to words
    if best_code is None:
        # Fallback
        best_code = list(range(729))
        
    return [list(idx_to_word[c]) for c in best_code]
