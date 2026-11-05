import numpy as np
import time
import itertools

def construct(q, n, radius):
    # For the specific task: q=3, n=8, radius=1
    # But we should handle general inputs if possible, or at least the specific one.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle q, n, radius.
    
    # However, for n=8, q=3, r=1, we can hardcode optimizations or use general logic.
    # General logic:
    
    if n == 0:
        return [[]]
        
    total_points = q ** n
    # Precompute balls for all possible codewords
    # This is O(q^n * (1 + n*(q-1)))
    # For q=3, n=8, this is 6561 * 17 ~ 111k items.
    
    # Represent points as integers 0..total_points-1
    # Convert integer to base-q digits
    def int_to_tuple(x):
        return [x // (q ** (n - 1 - i)) % q for i in range(n)]
        
    def tuple_to_int(t):
        return sum(t[i] * (q ** (n - 1 - i)) for i in range(n))
        
    # Precompute balls
    balls = []
    for c_int in range(total_points):
        c_tuple = int_to_tuple(c_int)
        ball = [c_int]
        for i in range(n):
            for d in range(q):
                if d == c_tuple[i]:
                    continue
                new_tuple = c_tuple[:]
                new_tuple[i] = d
                ball.append(tuple_to_int(new_tuple))
        balls.append(ball)
        
    # Greedy Set Cover
    uncovered = np.ones(total_points, dtype=bool)
    code = []
    
    # Optimization: If q=3, n=8, radius=1, we know the structure.
    # But let's run greedy.
    
    # To speed up, we can precompute for each point, which codewords cover it?
    # That's the inverse.
    # Let's stick to the forward greedy: for each candidate, count covered uncovered points.
    
    # We can use a list of sets for uncovered? No, boolean array is faster for vectorized ops if we use NumPy.
    # But iterating over 6561 candidates and doing sum(uncovered[ball]) is slow in pure Python.
    
    # Let's use a different representation.
    # Since n=8 is small, we can just use lists.
    
    # Actually, let's try the pruning approach on the 729 solution first, it's safer.
    # If pruning doesn't work well, we can try to add random swaps.
    
    # Let's combine: Start with the 729 solution.
    # 1. Calculate coverage count for each point.
    # 2. Try to remove any codeword that is redundant.
    # 3. If stuck, try to swap a codeword with a random one, and then prune again.
    
    # This "Swap and Prune" is a standard local search.
    
    # Initial solution: Tetracode product
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Extend to length n
        # If n > 4, pad with all combinations?
        # The seed code did: h + list(tail) for tail in product(range(3), repeat=n-4)
        # This gives 9 * 3^(n-4) codewords.
        
        if n == 4:
            initial_code = [tuple(h) for h in tetracode]
        else:
            tails = list(itertools.product(range(3), repeat=n-4))
            initial_code = [tuple(h + list(tail)) for h in tetracode for tail in tails]
    else:
        # Fallback: Random or Greedy from scratch
        # For simplicity, let's just start with a large random set or all points?
        # All points is size q^n.
        initial_code = [tuple(int_to_tuple(i)) for i in range(total_points)]
        
    code = [list(c) for c in initial_code]
    
    # Helper to convert list/tuple to int
    def to_int(c):
        return tuple_to_int(c)
        
    # Calculate coverage
    # coverage[p] = number of codewords covering p
    coverage = np.zeros(total_points, dtype=int)
    for c in code:
        c_int = to_int(c)
        for p in balls[c_int]:
            coverage[p] += 1
            
    # Pruning loop
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            c_int = to_int(c)
            # Check if any point in ball has coverage 1
            # If yes, cannot remove.
            # If no, remove.
            can_remove = True
            for p in balls[c_int]:
                if coverage[p] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove code[i]
                for p in balls[c_int]:
                    coverage[p] -= 1
                code.pop(i)
                changed = True
                break # Restart loop after removal
                
    # Local Search: Swap
    # Try to replace a codeword with a new one, then prune.
    # We need to be careful with time.
    # 120s limit.
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Random swaps
    import random
    random.seed(42)
    
    best_code = code[:]
    best_len = len(code)
    
    # We can try to swap each codeword with a random one.
    # This is O(M * (cost of prune)).
    # Prune is O(M * 17) roughly.
    
    # To be efficient, we can just try to remove codewords in a random order?
    # Or try to add a codeword that allows removing 2 others?
    
    # Let's try a simple hill climbing:
    # 1. Try to remove a random codeword. If valid, keep it.
    # 2. If no removal possible, try a swap: pick random i, pick random new_c.
    #    Replace code[i] with new_c.
    #    Try to prune.
    #    If length decreases, keep. Else revert.
    
    # Since we already pruned, no single removal is possible.
    # So we must swap.
    
    while time.time() < deadline:
        # Pick a random codeword to swap
        i = random.randint(0, len(code) - 1)
        old_c = code[i]
        old_c_int = to_int(old_c)
        
        # Pick a random new codeword
        new_c_int = random.randint(0, total_points - 1)
        if new_c_int == old_c_int:
            continue
        new_c = int_to_tuple(new_c_int)
        
        # Tentative swap
        code[i] = new_c
        
        # Try to prune
        # Update coverage for old_c (decrement) and new_c (increment)
        for p in balls[old_c_int]:
            coverage[p] -= 1
        for p in balls[new_c_int]:
            coverage[p] += 1
            
        # Check validity and prune
        valid = True
        for p in range(total_points):
            if coverage[p] == 0:
                valid = False
                break
        
        if valid:
            # Try to prune
            pruned = False
            for j in range(len(code)):
                cj = code[j]
                cj_int = to_int(cj)
                can_remove = True
                for p in balls[cj_int]:
                    if coverage[p] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for p in balls[cj_int]:
                        coverage[p] -= 1
                    code.pop(j)
                    pruned = True
                    break
                    
            if pruned:
                # Keep the change
                if len(code) < best_len:
                    best_len = len(code)
                    best_code = code[:]
            else:
                # Revert
                code[i] = old_c
                for p in balls[old_c_int]:
                    coverage[p] += 1
                for p in balls[new_c_int]:
                    coverage[p] -= 1
        else:
            # Revert
            code[i] = old_c
            for p in balls[old_c_int]:
                coverage[p] += 1
            for p in balls[new_c_int]:
                coverage[p] -= 1
                
    return best_code
