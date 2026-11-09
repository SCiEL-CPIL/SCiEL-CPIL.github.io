import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    # 1. Precompute balls as bitmasks
    # Map word tuple to index
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
            # General radius fallback (slower, but n=7 is small)
            for dist in range(radius + 1):
                if dist == 0:
                    continue
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

    # 2. Greedy Initialization
    def greedy_init():
        uncovered = all_mask
        code = []
        while uncovered:
            best_idx = -1
            max_cov = 0
            for i in range(total_words):
                cov = (balls[i] & uncovered).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_idx = i
            if best_idx == -1: break
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        return code

    # 3. Minimize Code (Remove Redundant)
    def minimize_code(code):
        code_set = set(code)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Sort by potential redundancy? Random order is fine for now.
            for c in list(code):
                # Compute union of all other balls
                other_union = 0
                for other in code:
                    if other != c:
                        other_union |= balls[other]
                # Check if ball[c] is covered by others
                if (balls[c] & ~other_union) == 0:
                    code.remove(c)
                    changed = True
                    break # Restart
        return code

    # 4. Find Valid Swap
    def find_valid_swap(code, c_in):
        """
        Find c_out not in code such that swapping c_in with c_out keeps cover valid.
        Returns c_out or -1.
        """
        code_set = set(code)
        # Compute union of Code \ {c_in}
        others_union = 0
        for other in code:
            if other != c_in:
                others_union |= balls[other]
        
        # Deficit: points covered by c_in but NOT by others
        deficit = balls[c_in] & ~others_union
        
        if deficit == 0:
            return -1 # c_in is already redundant, should have been removed in minimize_code
        
        # Find c_out such that balls[c_out] covers deficit
        # Also c_out must not be in code
        # Optimization: Only check c_out that intersects deficit?
        for c_out in range(total_words):
            if c_out in code_set:
                continue
            # Check if balls[c_out] covers deficit
            if (balls[c_out] & deficit) == deficit:
                return c_out
        return -1

    # 5. Main Loop
    current_code = greedy_init()
    current_code = minimize_code(current_code)
    best_code = current_code[:]
    best_size = len(current_code)
    
    while time.time() < deadline - 10:
        # Try to improve by swapping and minimizing
        improved = False
        
        # Try a few random swaps
        if len(current_code) > 0:
            for _ in range(10): # Try 10 swap attempts per iteration
                if time.time() > deadline - 10: break
                # Pick random c_in
                c_in_idx = random.randint(0, len(current_code)-1)
                c_in = current_code[c_in_idx]
                
                # Find valid c_out
                c_out = find_valid_swap(current_code, c_in)
                
                if c_out != -1:
                    # Perform Swap
                    current_code[c_in_idx] = c_out
                    
                    # Minimize the new code
                    new_code = minimize_code(current_code[:])
                    
                    # If new code is smaller, accept
                    if len(new_code) < best_size:
                        best_size = len(new_code)
                        best_code = new_code
                        current_code = new_code
                        improved = True
                    elif len(new_code) == len(current_code):
                        # Even if size didn't reduce, keep the new structure?
                        # Yes, to escape local minima.
                        # But we must ensure it's still valid. minimize_code returns valid code.
                        current_code = new_code
                    # If len(new_code) > best_size, we still keep current_code as new_code
                    # because we might be in a different local basin.
                    # Actually, if we didn't improve, we might want to revert to a previous state?
                    # No, ILS usually keeps the current state and perturbs again.
                    
        # If we improved, we restart the search from this better point.
        # If not, we continue.
        
        # If we haven't improved in a long time, we might need stronger perturbation.
        # For now, random swaps should suffice.
        
    return [words[c] for c in best_code]
