import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for safety
    
    # Parameters for the specific task: q=3, n=6, radius=1
    # The function signature is generic, but the task is specific.
    # We assume q=3, n=6, radius=1 as per the task description.
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Direct Product or Simple Greedy
        # For now, handle the specific case optimally, fallback to basic
        # A simple greedy covering code for general case
        space_size = q ** n
        all_words = [list(itertools.product(range(q), repeat=n)) for _ in range(1)] # Placeholder
        # Actually, let's just return a trivial valid code if params don't match
        # to avoid crashing, though the task is specific.
        # For q=3, n=6, r=1, we use the specialized solver.
        # If other params, use a basic greedy.
        pass

    # Specialized solver for q=3, n=6, r=1
    N = 729
    # Precompute neighbors for each word in {0,1,2}^6
    # A word is represented as an integer 0..728
    # Neighbor: same word, or change one digit by +/- 1 mod 3
    
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        # Decode i to digits
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits = tuple(digits)
        
        neigh_set = {i}
        for pos in range(n):
            for delta in [1, -1]:
                new_digit = (digits[pos] + delta) % q
                new_digits = list(digits)
                new_digits[pos] = new_digit
                new_val = 0
                for j in range(n):
                    new_val = new_val * q + new_digits[j]
                neigh_set.add(new_val)
        neighbors[i] = list(neigh_set)
    
    # Initialize with Direct Product of Tetracode (81 words)
    # Tetracode: 9 words of length 4
    tetracode = []
    for a in range(3):
        for b in range(3):
            c = (a + b) % 3
            d = (a + 2 * b) % 3
            # Convert to integer index for length 4 part
            idx4 = a * (3**3) + b * (3**2) + c * 3 + d
            tetracode.append(idx4)
            
    initial_code = []
    # The product is Tetracode (4 coords) x All (2 coords)
    # Wait, the tetracode is length 4. The remaining n-4 = 2 coords are free.
    # Total size 9 * 3^2 = 81.
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            # Combine h (first 4) and tail (last 2) into a single index
            # Index calculation: h is index of first 4 digits, tail is index of last 2?
            # No, h is an integer representing 4 digits.
            # We need to map (h, tail) to a single 6-digit integer.
            # h corresponds to digits d0, d1, d2, d3.
            # tail corresponds to d4, d5.
            # Value = d0*3^5 + d1*3^4 + d2*3^3 + d3*3^2 + d4*3^1 + d5*3^0
            # h = d0*3^3 + d1*3^2 + d2*3^1 + d3*3^0
            # So h * 3^2 + tail_val
            tail_val = tail[0] * 3 + tail[1]
            idx = h * 9 + tail_val
            initial_code.append(idx)
            
    best_code = initial_code
    best_size = len(best_code)
    
    def get_cover_count(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def is_valid(code):
        counts = get_cover_count(code)
        return np.all(counts > 0)

    def greedy_reduce(code):
        # code is a list of ints
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # Check if c is redundant
                # If all neighbors have count > 1, we can remove c
                # Note: counts includes c's contribution
                min_count = np.min(counts[neighbors[c]])
                if min_count > 1:
                    # Remove c
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
                    if time.time() > deadline:
                        return list(code_set)
        return list(code_set)

    # Initial reduction
    current_code = greedy_reduce(initial_code)
    if len(current_code) < best_size:
        best_code = current_code
        best_size = len(best_code)
        
    # Iterated Local Search
    # We will try to perturb and reduce.
    # Perturbation: Swap a random codeword with a random other codeword.
    # Or: Remove a random codeword, add a random codeword.
    # Since we want to minimize size, removing and adding keeps size same, 
    # but might allow further reduction.
    
    all_words = list(range(N))
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Perturb
        # Pick a random codeword to remove
        if not current_code:
            break
        idx_remove = random.randint(0, len(current_code) - 1)
        c_remove = current_code[idx_remove]
        
        # Pick a random new codeword to add
        # To increase chance of improvement, maybe pick a word that is "under-covered"
        # or just random. Random is fine for ILS.
        c_add = random.choice(all_words)
        
        # Create new set
        new_set = set(current_code)
        new_set.remove(c_remove)
        new_set.add(c_add)
        
        # If we just swapped, size is same.
        # Greedy reduce will try to shrink it.
        reduced_code = greedy_reduce(list(new_set))
        
        if len(reduced_code) < best_size:
            best_code = reduced_code
            best_size = len(best_code)
            current_code = reduced_code
        else:
            # Accept with some probability (Simulated Annealing style)
            # Or just always accept to explore (Random Walk)
            # Given we want to find global min, let's do a simple random walk
            # that accepts if size is same or smaller, or with prob p if larger.
            # Actually, if size is larger, it's bad. 
            # Let's just accept if size <= best_size to keep exploring good states?
            # No, if size is same, it's a different local minimum.
            # Let's accept if len(reduced_code) <= len(current_code) 
            # This ensures we don't get stuck in a local min of same size.
            if len(reduced_code) <= len(current_code):
                current_code = reduced_code
            else:
                # Revert
                pass
                
        # Occasionally, do a stronger perturbation: 
        # Remove 2, Add 2? Or just restart from best?
        if iterations % 50 == 0:
             # Reset to best to avoid drifting too far
             if random.random() < 0.1:
                 current_code = list(best_code)
                 
    # Final check
    if is_valid(best_code):
        # Convert back to list of lists of ints
        result = []
        for c in best_code:
            digits = []
            x = c
            for _ in range(n):
                digits.append(x % q)
                x //= q
            result.append(digits)
        return result
    else:
        # Fallback to initial valid code
        result = []
        for c in initial_code:
            digits = []
            x = c
            for _ in range(n):
                digits.append(x % q)
                x //= q
            result.append(digits)
        return result
