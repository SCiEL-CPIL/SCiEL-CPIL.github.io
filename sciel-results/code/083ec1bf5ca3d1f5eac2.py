import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    # Precompute balls as bitmasks
    balls = [0] * total_words
    
    # Helper to get word from index
    def int_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Helper to get index from word
    def word_to_int(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Precompute all balls
    for i in range(total_words):
        w = int_to_word(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = w[:]
                    new_w[pos] = new_val
                    new_idx = word_to_int(new_w)
                    mask |= (1 << new_idx)
        balls[i] = mask

    # 1. Greedy Set Cover
    def greedy_cover():
        covered = 0
        selected = []
        remaining = list(range(total_words))
        
        while covered != all_mask:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_gain = 0
            for i in remaining:
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            covered |= balls[best_idx]
            remaining.remove(best_idx)
            
        return selected

    # 2. Local Search Polish
    def polish(solution):
        if not solution:
            return solution
            
        improved = True
        while improved:
            if time.time() > deadline - 5:
                return solution
                
            improved = False
            
            # Phase 1: Remove redundant codewords
            new_solution = []
            for c in solution:
                if time.time() > deadline - 5:
                    return solution
                    
                mask_others = 0
                for c2 in solution:
                    if c2 != c:
                        mask_others |= balls[c2]
                
                if mask_others == all_mask:
                    improved = True
                else:
                    new_solution.append(c)
            
            solution = new_solution
            if not solution:
                return solution
                
            # Phase 2: Exchange codewords
            swapped = False
            sol_set = set(solution)
            
            for i, c in enumerate(solution):
                if time.time() > deadline - 5:
                    return solution
                    
                mask_others = 0
                for j, c2 in enumerate(solution):
                    if j != i:
                        mask_others |= balls[c2]
                
                gap = all_mask & ~mask_others
                
                # Find a replacement c'
                # Randomize the order of c' to escape local minima?
                # We can shuffle a list of candidates.
                candidates = [x for x in range(total_words) if x not in sol_set]
                random.shuffle(candidates)
                
                for c_prime in candidates:
                    if (gap & ~balls[c_prime]) == 0:
                        solution[i] = c_prime
                        swapped = True
                        improved = True
                        break
                        
                if swapped:
                    break
            
        return solution

    best_solution = None
    best_size = float('inf')
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        random.seed(attempt)
        
        sol = greedy_cover()
        if not sol:
             continue
             
        sol = polish(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
    if best_solution is None:
        # Fallback
        sol = greedy_cover()
        if not sol:
             return []
        best_solution = sol

    result = []
    for idx in best_solution:
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = idx % q
            idx //= q
        result.append(w)
        
    return result
