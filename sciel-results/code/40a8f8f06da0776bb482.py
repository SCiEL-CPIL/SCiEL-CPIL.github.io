import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Problem parameters
    N = q ** n  # 2187 for q=3, n=7
    
    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute all words
    words = [idx_to_word(i) for i in range(N)]
    
    # Build word to index mapping for fast neighbor lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Compute neighbors (ball of radius 1) as lists of indices
    # For q=3, n=7, radius=1, ball size is 1 + 7*2 = 15
    neighbors = []
    for i in range(N):
        w_i = words[i]
        ball = [i]
        for pos in range(n):
            orig_val = w_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w_i[:]
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                ball.append(j)
        neighbors.append(ball)
        
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are often fine for small N. We'll stick to lists for flexibility 
    # but use numpy for coverage counts.
    
    # Precompute cover_map: for each word w, which codewords cover it?
    # This is the inverse of neighbors.
    # cover_map[w] = list of c such that w in neighbors[c]
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in neighbors[c]:
            cover_map[w].append(c)
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        # cnt[c] = number of uncovered words in ball(c)
        cnt = [len(neighbors[c]) for c in range(N)]
        covered = [False] * N
        code = []
        uncovered_count = N
        
        # To speed up max finding, we could use a heap, but N is small.
        # We'll scan for max.
        while uncovered_count > 0:
            # Find max cnt
            max_val = -1
            candidates = []
            for i in range(N):
                if cnt[i] > max_val:
                    max_val = cnt[i]
                    candidates = [i]
                elif cnt[i] == max_val:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered_count > 0
                
            c = rng.choice(candidates)
            code.append(c)
            
            for w in neighbors[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    # Update counts for all codewords that cover w
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1
                        
        return code

    def refine(code):
        if not code:
            return code
            
        cover_count = [0] * N
        for c in code:
            for w in neighbors[c]:
                cover_count[w] += 1
                
        code_list = list(code)
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            for i in range(len(code_list)):
                c = code_list[i]
                safe = True
                for w in neighbors[c]:
                    if cover_count[w] == 1:
                        safe = False
                        break
                if safe:
                    # Remove c
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    changed = True
                    break # Restart loop to handle index shifts
        return code_list

    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
            
        cover_count = [0] * N
        for c in code_list:
            for w in neighbors[c]:
                cover_count[w] += 1
                
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                
                # Temporarily remove c_remove
                for w in neighbors[c_remove]:
                    cover_count[w] -= 1
                    
                # Find uncovered words
                uncovered = []
                for w in range(N):
                    if cover_count[w] == 0:
                        uncovered.append(w)
                        
                if not uncovered:
                    # We can just remove it
                    code_list.pop(i)
                    improved = True
                    break
                    
                # Try to find a replacement c_add that covers all uncovered words
                # and is not already in code_list (or is c_remove)
                found = -1
                # Optimization: Check candidates that cover at least one uncovered word
                candidates = set()
                for w in uncovered:
                    candidates.update(cover_map[w])
                
                # Filter out those already in code (except c_remove)
                code_set = set(code_list)
                code_set.remove(c_remove)
                
                for c_add in candidates:
                    if c_add in code_set:
                        continue
                    # Check if c_add covers all uncovered
                    ball_add = set(neighbors[c_add])
                    if all(w in ball_add for w in uncovered):
                        found = c_add
                        break
                        
                if found >= 0:
                    # Swap
                    code_list[i] = found
                    for w in neighbors[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Restore
                    for w in neighbors[c_remove]:
                        cover_count[w] += 1
                        
        return code_list

    best_code = None
    best_len = float('inf')
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        code = greedy_cover(i)
        refined = refine(code)
        
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined
            
        # Run local search on the best code found so far (or current if better)
        # To save time, only run LS if we have time
        if time.time() < deadline - 20:
            ls_code = local_search(refined, i, deadline - 10)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code

    if best_code is None:
        # Fallback: return empty or single word if something went wrong
        return [words[0]]

    # Final refinement
    final_code = refine(best_code)
    
    return [words[c] for c in final_code]
