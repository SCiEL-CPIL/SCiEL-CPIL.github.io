import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    total_words = q ** n
    
    # For very large spaces, this approach might fail, but for 3^6=729 it is perfect.
    if total_words > 100000:
         # Fallback: Return a simple valid code (e.g. all zeros? No, must cover).
         # For this specific task, we assume small n.
         pass

    def tuple_to_idx(t):
        idx = 0
        for d in t:
            idx = idx * q + d
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute balls
    balls = []
    for i in range(total_words):
        center = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == center[pos]:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                ball.append(tuple_to_idx(tuple(neighbor)))
        balls.append(ball)
        
    def greedy_construction(randomize=False):
        covered = np.zeros(total_words, dtype=bool)
        uncovered = total_words
        code = []
        used = set()
        
        while uncovered > 0:
            if time.time() - start_time > deadline - safety_margin:
                break
                
            best_cand = -1
            best_gain = -1
            top_cands = []
            
            for cand_idx in range(total_words):
                if cand_idx in used:
                    continue
                gain = 0
                for w in balls[cand_idx]:
                    if not covered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    top_cands = [cand_idx]
                elif gain == best_gain and randomize:
                    top_cands.append(cand_idx)
            
            if best_cand == -1 and not top_cands:
                break
                
            if randomize and top_cands:
                best_cand = random.choice(top_cands)
            else:
                best_cand = top_cands[0] if top_cands else -1
                
            if best_cand == -1:
                break

            code.append(best_cand)
            used.add(best_cand)
            for w in balls[best_cand]:
                if not covered[w]:
                    covered[w] = True
                    uncovered -= 1
                    
        return code

    def prune(code):
        # code is a list of indices
        counts = np.zeros(total_words, dtype=int)
        for c in code:
            for w in balls[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(code):
                if time.time() - start_time > deadline - safety_margin:
                    return code
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if counts[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        counts[w] -= 1
                    code.pop(i)
                    changed = True
                else:
                    i += 1
        return code

    # Initial deterministic greedy
    code = greedy_construction(randomize=False)
    code = prune(code)
    best_code = code
    best_len = len(code)
    
    # Randomized restarts
    # We can run this loop until time is up
    while time.time() - start_time < deadline - safety_margin:
        code = greedy_construction(randomize=True)
        code = prune(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
        # If we found the theoretical lower bound or close to it, we can stop?
        # Lower bound is 71. Best known 73.
        if best_len <= 73:
            # We can try to find 72 or 71, but 73 is a very good score.
            # Continue searching for a bit longer?
            pass

    result = []
    for idx in best_code:
        result.append(list(idx_to_tuple(idx)))
        
    return result
