import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    # 1. Precompute balls as bitmasks
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = [0] * total_words
    for i in range(total_words):
        w = words[i]
        mask = 1 << i
        if radius == 1:
            for p in range(n):
                for val in range(q):
                    if val != w[p]:
                        new_w = w.copy()
                        new_w[p] = val
                        new_i = word_to_idx[tuple(new_w)]
                        mask |= (1 << new_i)
        else:
            # General radius fallback
            for dist in range(radius + 1):
                if dist == 0: continue
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if any(v == w[p] for v, p in zip(vals, positions)):
                            continue
                        new_w = w.copy()
                        for p, v in zip(positions, vals):
                            new_w[p] = v
                        new_i = word_to_idx[tuple(new_w)]
                        mask |= (1 << new_i)
        balls[i] = mask

    def greedy_init(seed=0):
        random.seed(seed)
        uncovered = all_mask
        code = []
        while uncovered:
            best_idx = -1
            max_cov = 0
            # Randomize tie-breaking to explore different basins
            candidates = []
            for i in range(total_words):
                cov = (balls[i] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    candidates = [i]
                elif cov == max_cov:
                    candidates.append(i)
            if best_idx == -1: break
            # Pick random from best candidates
            best_idx = random.choice(candidates) if candidates else -1
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        return code

    def minimize_code(code):
        code_set = set(code)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Iterate in random order to avoid bias
            for c in random.sample(code, len(code)):
                other_union = 0
                for other in code:
                    if other != c:
                        other_union |= balls[other]
                if (balls[c] & ~other_union) == 0:
                    code.remove(c)
                    changed = True
                    break
        return code

    def find_valid_swap(code, c_in):
        code_set = set(code)
        others_union = 0
        for other in code:
            if other != c_in:
                others_union |= balls[other]
        
        deficit = balls[c_in] & ~others_union
        if deficit == 0:
            return -1
        
        # Find c_out covering deficit
        # Optimization: Precompute list of words? No, just iterate.
        for c_out in range(total_words):
            if c_out in code_set:
                continue
            if (balls[c_out] & deficit) == deficit:
                return c_out
        return -1

    def perturb(code, k):
        """Remove k random codewords and re-add greedily"""
        if len(code) <= k:
            return code[:]
        
        # Remove k random
        to_remove = random.sample(code, k)
        new_code = [c for c in code if c not in to_remove]
        
        # Calculate uncovered
        covered = 0
        for c in new_code:
            covered |= balls[c]
        uncovered = all_mask & ~covered
        
        # Greedily add
        while uncovered:
            best_idx = -1
            max_cov = 0
            for i in range(total_words):
                if i in new_code:
                    continue
                cov = (balls[i] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1: break
            new_code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return new_code

    # Main Search
    best_code = []
    best_size = float('inf')
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() > deadline - 10: break
        
        current_code = greedy_init(seed)
        current_code = minimize_code(current_code)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
        
        # Local Search on this instance
        no_improve_count = 0
        while time.time() < deadline - 10 and no_improve_count < 100:
            improved = False
            
            # Try single swaps
            for _ in range(20):
                if time.time() > deadline - 10: break
                if not current_code: break
                
                c_in_idx = random.randint(0, len(current_code)-1)
                c_in = current_code[c_in_idx]
                
                c_out = find_valid_swap(current_code, c_in)
                
                if c_out != -1:
                    # Perform swap
                    current_code[c_in_idx] = c_out
                    new_code = minimize_code(current_code[:])
                    
                    if len(new_code) < best_size:
                        best_size = len(new_code)
                        best_code = new_code
                        current_code = new_code
                        improved = True
                        no_improve_count = 0
                    else:
                        # Keep current structure even if size didn't drop immediately
                        current_code = new_code
                        if len(new_code) < len(current_code): # This check is redundant if we assign
                            pass
                        # If size didn't change, we might still want to keep the new config
                        # but we need to be careful not to loop forever on same size
                        # Let's accept equal size swaps to move around
                        if len(new_code) == len(current_code):
                             current_code = new_code
                             improved = True # Count as progress? No, let's not reset no_improve
                             # Actually, accepting equal size swaps helps escape minima.
                             # But we should limit them.
            
            if not improved:
                no_improve_count += 1
                # Perturb
                k = random.choice([3, 4, 5])
                current_code = perturb(current_code, k)
                current_code = minimize_code(current_code)
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    no_improve_count = 0

    return [words[c] for c in best_code]
