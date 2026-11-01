import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Specific to the task: q=3, n=6, radius=1
    # General fallback for other parameters if needed, but task is specific.
    if q != 3 or n != 6 or radius != 1:
        # Generic greedy fallback
        return _generic_greedy(q, n, radius)

    deadline = time.time() + 115.0
    N = q ** n  # 729
    
    # Precompute neighbors for each word (ball of radius 1)
    # word -> list of words in ball(word)
    # We represent words as integers 0..728
    balls = [None] * N
    for w in range(N):
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        neighbors = [w] # Include self
        for i in range(n):
            orig = digits[i]
            for val in range(q):
                if val == orig:
                    continue
                new_digits = digits[:]
                new_digits[i] = val
                new_int = 0
                mult = 1
                for j in range(n):
                    new_int += new_digits[j] * mult
                    mult *= q
                neighbors.append(new_int)
        balls[w] = neighbors

    # Helper to convert int to list of digits
    def to_digits(w):
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        return digits

    # Initial Construction: Greedy Set Cover
    # We want to cover all 729 points.
    # Start with empty set, add word that covers most uncovered points.
    
    def greedy_cover():
        code = set()
        uncovered = np.ones(N, dtype=bool)
        
        # Precompute which words are covered by which codeword candidate
        # Actually, just iterate.
        while np.any(uncovered):
            best_word = -1
            max_covered = 0
            # To speed up, we can sample or just iterate all 729.
            # 729 is small.
            for w in range(N):
                if w in code:
                    continue
                # Count how many uncovered points are in balls[w]
                # balls[w] is a list of ints
                # We need to check how many of these are in uncovered
                # This is slow if done naively for every w in every iteration.
                # Optimization: Only consider words that cover at least one uncovered point.
                
                # Simple check:
                covered_count = 0
                for nb in balls[w]:
                    if uncovered[nb]:
                        covered_count += 1
                        if covered_count > max_covered:
                            max_covered = covered_count
                            best_word = w
                            if max_covered == len(balls[w]): # Can't do better
                                break
                if covered_count == 0:
                    continue
            if best_word == -1:
                break # Should not happen
            code.add(best_word)
            for nb in balls[best_word]:
                uncovered[nb] = False
        
        return code

    # Better Initial: Use the 81-word tetracode-based construction as a start?
    # The seed did that. Let's try a randomized greedy or a specific structured start.
    # Actually, a good start is often a random valid code found by greedy, 
    # or the 81-word code.
    
    # Let's use the 81-word code as a deterministic strong start.
    # Tetracode is a [4,2,3] code.
    # Direct sum with 2 random positions? No, direct sum with [2,0,1]?
    # The seed used: h in tetracode (16 words), tail in {0,1,2}^2 (9 words). 16*9 = 144?
    # Wait, seed code:
    # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)] -> 9 words?
    # No, a in 0..2, b in 0..2. 3*3=9 words.
    # Tail length n-4 = 2. 3^2 = 9 tails.
    # Total 9 * 9 = 81 words.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            w_int = 0
            mult = 1
            for d in word:
                w_int += d * mult
                mult *= q
            initial_code.add(w_int)
            
    # Verify initial code is valid (it should be)
    # We will run local search from this state.
    
    current_code = initial_code
    cover_count = np.zeros(N, dtype=np.int8)
    
    def update_cover(c, sign):
        for nb in balls[c]:
            cover_count[nb] += sign

    for c in current_code:
        update_cover(c, 1)
        
    # Check validity
    def is_valid():
        return np.min(cover_count) >= 1

    best_code = set(current_code)
    best_size = len(best_code)
    
    # Local Search Loop
    # We use a timer to switch strategies or restart
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove codewords
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            update_cover(c, -1)
            if is_valid():
                current_code.remove(c)
                improved = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                # Don't break, try to remove more in the same pass?
                # Or break and restart? Restarting is safer to avoid order bias.
                # But removing multiple in one pass is faster.
                # Let's continue to try removing others from the original list?
                # No, the list is stale. Let's break and restart the outer loop.
                break
            else:
                update_cover(c, 1)
                
        if improved:
            continue
        
        # 2. If no removal, try swaps
        # We want to find a swap that enables a future removal.
        # Heuristic: Replace a codeword with one that covers points currently
        # covered by only 1 codeword (critical points), or just random.
        
        swaps_made = False
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c_old in code_list:
            # Generate candidates
            # 1. Random
            # 2. Neighbor of c_old
            # 3. Word covering many "critical" points?
            
            found_swap = False
            # Try a few random candidates
            candidates = set()
            # Add some randoms
            for _ in range(5):
                candidates.add(random.randint(0, N - 1))
            # Add neighbors
            for nb in balls[c_old]:
                candidates.add(nb)
            
            for c_new in candidates:
                if c_new in current_code:
                    continue
                
                # Swap
                update_cover(c_old, -1)
                update_cover(c_new, 1)
                
                if is_valid():
                    current_code.remove(c_old)
                    current_code.add(c_new)
                    swaps_made = True
                    found_swap = True
                    break
                else:
                    update_cover(c_new, -1)
                    update_cover(c_old, 1)
                    
            if found_swap:
                break # Restart removal phase
        
        if swaps_made:
            continue
        
        # 3. Stuck. Perturb.
        # Replace 2 random codewords with 2 random new ones.
        # This might break validity, so we must repair or reject.
        # If we reject, we are stuck.
        # Better: "2-opt" style. Remove 2, Add 2. Check validity.
        
        c_list = list(current_code)
        if len(c_list) >= 2:
            c1, c2 = random.sample(c_list, 2)
            success = False
            for _ in range(50): # Try 50 random pairs
                n1 = random.randint(0, N - 1)
                n2 = random.randint(0, N - 1)
                if n1 in current_code or n2 in current_code or n1 == n2:
                    continue
                
                update_cover(c1, -1)
                update_cover(c2, -1)
                update_cover(n1, 1)
                update_cover(n2, 1)
                
                if is_valid():
                    current_code.remove(c1)
                    current_code.remove(c2)
                    current_code.add(n1)
                    current_code.add(n2)
                    success = True
                    break
                else:
                    update_cover(n1, -1)
                    update_cover(n2, -1)
                    update_cover(c1, 1)
                    update_cover(c2, 1)
            
            if success:
                continue
        
        # If even perturbation fails, we might be in a very deep local minimum.
        # Restart from a slightly perturbed version of the best code found so far?
        # Or just keep trying random swaps with more iterations?
        
        # Let's try a more aggressive swap: try all neighbors of c_old as candidates.
        # If that fails, try random.
        
        # If we are stuck for a long time, we might want to re-initialize
        # with a different random seed or a different starting code.
        # For now, let's just keep trying the above.
        
        # To avoid infinite loops of failing perturbations, we can add a counter
        # and if it gets too high, reset to best_code and shuffle.
        
        # Simple reset logic:
        # If we haven't improved in a while (hard to track without counters),
        # just let the loop run. The random nature will eventually break out.
        
        # Optimization: If we are stuck, try swapping a codeword with a word
        # that is far away (random) vs close (neighbor).
        # The current logic mixes them.

    # Convert best_code to list of lists
    result = []
    for w_int in best_code:
        result.append(to_digits(w_int))
        
    return result

def _generic_greedy(q, n, radius):
    # Fallback for other parameters
    # Simple greedy set cover
    N = q ** n
    if N > 10000:
        # Too large for simple greedy in Python without optimization, 
        # but for the task, q=3, n=6 is the only case.
        pass
        
    # Precompute balls
    balls = [None] * N
    for w in range(N):
        digits = [0] * n
        temp = w
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        neighbors = [w]
        for i in range(n):
            orig = digits[i]
            for val in range(q):
                if val == orig:
                    continue
                new_digits = digits[:]
                new_digits[i] = val
                new_int = 0
                mult = 1
                for j in range(n):
                    new_int += new_digits[j] * mult
                    mult *= q
                neighbors.append(new_int)
        balls[w] = neighbors

    code = set()
    uncovered = np.ones(N, dtype=bool)
    
    while np.any(uncovered):
        best_word = -1
        max_covered = 0
        for w in range(N):
            if w in code:
                continue
            covered_count = 0
            for nb in balls[w]:
                if uncovered[nb]:
                    covered_count += 1
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        if best_word == -1:
            break
        code.add(best_word)
        for nb in balls[best_word]:
            uncovered[nb] = False
            
    result = []
    for w_int in code:
        digits = [0] * n
        temp = w_int
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        result.append(digits)
    return result
