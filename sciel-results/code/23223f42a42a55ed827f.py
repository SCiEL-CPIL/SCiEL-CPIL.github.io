import math

def construct():
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        # Bitmask for sums and diffs
        sum_mask = 0
        diff_mask = 0
        
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    # Strategy: Difference-sparse construction via greedy growth
    # We start with a seed set and greedily add elements to maximize the score.
    # The "difference-sparse" idea is implicitly explored by the greedy step
    # which will favor elements that add many new differences relative to sums.
    
    # We try several different starting points and maximum limits to find a good local optimum.
    
    # Candidate pools:
    # 1. Powers of 2 (exponential growth, very sparse)
    # 2. Small integers (dense)
    # 3. Mixed
    
    # We'll run a greedy search from a few different initial states.
    
    def greedy_search(initial_u, max_val, max_iter=50):
        u = sorted(initial_u)
        current_score = calc_score(u)
        
        for _ in range(max_iter):
            best_candidate = None
            best_gain = 0
            best_new_score = current_score
            
            # Try adding any element from 1 to max_val not already in u
            # To speed up, we can limit the range or use a heuristic.
            # Since max_val is small (e.g., 100), checking all is O(max_val * n^2) which is fine.
            for x in range(1, max_val + 1):
                if x in u:
                    continue
                
                new_u = u + [x]
                new_u.sort()
                s = calc_score(new_u)
                
                if s > best_new_score:
                    best_new_score = s
                    best_candidate = x
                    best_gain = s - current_score
            
            if best_candidate is None or best_gain <= 0:
                break
            
            u.append(best_candidate)
            u.sort()
            current_score = best_new_score
            
        return u, current_score

    # Try different initial seeds and max values
    seeds = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
    ]
    
    max_vals = [20, 30, 50, 100]
    
    for seed in seeds:
        for mv in max_vals:
            u, score = greedy_search(seed, mv)
            if score > best_score:
                best_score = score
                best_set = u

    # Also try a "difference-sparse" specific construction:
    # U = {0, d, 2d, ..., (k-1)d} is an arithmetic progression.
    # |U+U| = 2k-1, |U-U| = 2k-1. Ratio 1. Score 1.
    # If we perturb it?
    
    # Let's also try a construction based on base-b representation with sparse digits.
    # e.g., U = { sum a_i * b^i : a_i in {0,1} } for small b.
    # This is just powers of b.
    
    # What if we use two scales?
    # U = { i * D + j : i in {0..I}, j in {0..J} }
    # This is a grid.
    # Let's test a few grid-like structures explicitly.
    
    for D in range(2, 20):
        for I in range(1, 10):
            for J in range(1, 10):
                u = []
                for i in range(I + 1):
                    for j in range(J + 1):
                        u.append(i * D + j)
                u.sort()
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u

    return best_set
