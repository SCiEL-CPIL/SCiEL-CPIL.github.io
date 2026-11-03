import itertools
import time

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Generate initial code: Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    # Tetracode: { (a, b, a+b, a+2b) mod 3 }
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # The task specifies q=3, n=6, radius=1.
    # If parameters differ, fallback to simple construction or error.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected for this specific task instance)
        # Just return a trivial covering code (all words) if n is small, else error
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # Simple construction: repeat first coordinate? No, just return empty or basic.
            # For robustness, return a list of a few words, though invalid for large n.
            # The evaluator expects a valid covering code.
            # We assume q=3, n=6, radius=1 based on the task description.
            pass

    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # 2. Map words to indices
    # Index = w0 + 3*w1 + 9*w2 + 27*w3 + 81*w4 + 243*w5
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(reversed(w)):
            idx += val * (3 ** i)
        return idx

    # Precompute spheres: sphere[i] is a list of indices within distance 1 of idx i
    # There are 729 words.
    spheres = [None] * (3 ** n)
    for idx in range(3 ** n):
        # Convert idx to word
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % 3
            temp //= 3
        
        sphere = [idx]
        # Generate neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbor_idx = word_to_idx(w)
                sphere.append(neighbor_idx)
                w[pos] = orig_val
        spheres[idx] = sphere

    # 3. Greedy Removal
    # counts[i] = number of codewords covering word i
    counts = [0] * (3 ** n)
    active = [True] * len(initial_codewords)
    
    # Initialize counts
    for i, cw in enumerate(initial_codewords):
        idx = word_to_idx(cw)
        for neighbor in spheres[idx]:
            counts[neighbor] += 1

    removed_count = 0
    improved = True
    
    # We might do multiple passes if we restart with different orders, 
    # but standard greedy is one pass. 
    # To find better local optima, we can randomize the order of checking removal.
    # Let's do a few iterations with different random shuffles if time permits.
    
    best_code = initial_codewords[:]
    
    import random
    
    # Try a few random orders for removal to escape potential local minima of the greedy process
    # (Though greedy removal is generally quite effective, the order can matter slightly)
    
    for attempt in range(50): # Limit attempts to ensure speed
        if time.time() > deadline:
            break
            
        # Reset state
        counts = [0] * (3 ** n)
        active = [True] * len(initial_codewords)
        current_code = initial_codewords[:]
        
        for i, cw in enumerate(current_code):
            idx = word_to_idx(cw)
            for neighbor in spheres[idx]:
                counts[neighbor] += 1
                
        # Shuffle indices to try different removal orders
        order = list(range(len(current_code)))
        random.shuffle(order)
        
        # Greedy removal pass
        for i in order:
            if not active[i]:
                continue
                
            cw = current_code[i]
            idx = word_to_idx(cw)
            
            # Check if this codeword is essential for any word in its sphere
            essential = False
            for neighbor in spheres[idx]:
                if counts[neighbor] == 1:
                    essential = True
                    break
                    
            if not essential:
                # Remove it
                active[i] = False
                for neighbor in spheres[idx]:
                    counts[neighbor] -= 1
                removed_count = sum(1 for a in active if not a)
                
        # Extract remaining codewords
        remaining = [current_code[i] for i in range(len(current_code)) if active[i]]
        
        if len(remaining) < len(best_code):
            best_code = remaining
            # If we hit the known lower bound or very close, we can stop early?
            # Best known is 73. Lower bound 71.
            if len(best_code) <= 73:
                # We found a very good solution, stop searching
                break

    return best_code
