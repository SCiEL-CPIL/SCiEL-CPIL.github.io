import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want to be safe)
    
    # --- 1. Setup and Fallback ---
    # The task specifies q=3, n=6, radius=1.
    # If parameters differ, fallback to a simple valid construction.
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # For robustness, return a list of a few words, though likely invalid for large n.
            # We assume q=3, n=6, radius=1 based on the task description.
            # A trivial valid code is all words.
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    
    # --- 2. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres (Hamming balls of radius 1) as sets of indices
    # spheres[i] contains all indices within distance 1 of index i
    spheres = [None] * total_words
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres[idx] = sphere_set

    # --- 3. Initial Code Construction ---
    # Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    # Tetracode: { (a, b, a+b, a+2b) mod 3 }
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices for fast processing
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 4. Optimization Loop ---
    
    best_code = initial_codewords[:]
    best_len = len(best_code)
    
    # We will try multiple random orders for the greedy removal process.
    # The greedy removal is:
    # 1. Start with a set of codewords.
    # 2. Compute coverage counts for all words.
    # 3. Iterate through codewords in a random order.
    # 4. If a codeword is "redundant" (no word in its sphere is covered only by it), remove it.
    # 5. Update counts.
    # 6. Repeat until no more removals or time limit.
    
    # To speed up, we can use a more efficient data structure or just optimize the Python loop.
    # Given n=6, total_words=729, and code size ~80-100, the inner loop is small enough.
    
    rng = random.Random(42)
    
    attempts = 0
    # We allow many attempts as long as we are within the time limit.
    # Each attempt involves O(|Code| * |Sphere|) operations for the first pass, which is ~100 * 19 = 1900 ops.
    # Very fast. We can do thousands of attempts.
    
    while time.time() < deadline:
        # Reset state for this attempt
        counts = [0] * total_words
        active = [True] * len(initial_indices)
        
        # Initialize counts
        for idx in initial_indices:
            for neighbor in spheres[idx]:
                counts[neighbor] += 1
        
        # Create a random order for removal
        order = list(range(len(initial_indices)))
        rng.shuffle(order)
        
        # Greedy removal pass
        # We iterate through the order. If a codeword is removable, we remove it.
        # Note: Removing a codeword changes the "removability" of others.
        # A single pass in a fixed order is a heuristic. 
        # To be more thorough, we could loop until no changes, but that might be slow if we do it many times.
        # However, for this small problem, a single pass with a random order is a good balance.
        # Actually, let's do a "sweep" approach:
        # We go through the list. If we remove one, we continue.
        # Since the order is random, this approximates a random greedy set cover minimization.
        
        for i in order:
            if not active[i]:
                continue
                
            idx = initial_indices[i]
            
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
        
        # Extract remaining codewords
        remaining_indices = [initial_indices[i] for i in range(len(initial_indices)) if active[i]]
        
        if len(remaining_indices) < best_len:
            best_code = [list(idx_to_word(idx)) for idx in remaining_indices]
            best_len = len(remaining_indices)
            
            # If we hit the known lower bound or very close, we can stop early?
            # Best known is 73. Lower bound 71.
            if best_len <= 73:
                # We found a very good solution, stop searching
                break
        
        attempts += 1
        
        # Optional: If we are not improving, we might want to restart with a slightly different initial set?
        # But the initial set is fixed (direct sum). The randomness is in the removal order.
        # This is a standard local search.
        
        # To diversify further, we could also try adding random words and then pruning,
        # but the direct sum is already a strong start.
        # Let's stick to the removal heuristic on the direct sum.
        
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break

    return best_code
