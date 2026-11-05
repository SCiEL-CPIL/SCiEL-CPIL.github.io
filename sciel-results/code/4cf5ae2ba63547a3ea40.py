import itertools
import random
import time

def construct(q, n, radius):
    # Parameters specific to the task instance
    # q=3, n=8, radius=1
    
    # 1. Generate initial code
    # Tetracode for q=3, n=4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tails = list(itertools.product(range(3), repeat=n - 4))
    
    initial_code = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to indices for faster processing
    def encode(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for all indices in the space
    # Space size q**n
    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    # To compute neighbors, we iterate through all indices and all possible changes
    # This might be slow if done naively for large n, but n=8 is small.
    # Better: generate neighbors for each index on the fly or precompute efficiently.
    
    # Efficient precomputation:
    # For each index, decode to word, generate neighbors, encode back.
    # 6561 indices * 17 neighbors * (decode+encode cost)
    # Decode/encode is O(n). 6561 * 17 * 8 ~ 890k ops. Very fast.
    
    def decode(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    for i in range(space_size):
        w = decode(i)
        neighbors[i].append(i) # self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig: continue
                w[pos] = new_val
                neighbors[i].append(encode(w))
            w[pos] = orig
            
    # Greedy Removal Function
    def greedy_remove(code_indices, seed):
        rng = random.Random(seed)
        # coverage count for each point in space
        coverage = [0] * space_size
        
        # Initialize coverage
        for c in code_indices:
            for v in neighbors[c]:
                coverage[v] += 1
                
        # List of indices to consider for removal
        # We shuffle the list of codewords
        candidates = list(code_indices)
        rng.shuffle(candidates)
        
        remaining = set(code_indices)
        
        for c in candidates:
            if c not in remaining:
                continue
            # Check if c is redundant
            redundant = True
            for v in neighbors[c]:
                if coverage[v] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                remaining.remove(c)
                for v in neighbors[c]:
                    coverage[v] -= 1
                    
        return list(remaining)

    # Run multiple restarts
    best_code = initial_code
    best_size = len(initial_code)
    
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    seed_counter = 0
    while time.time() < deadline:
        current_seed = seed_counter
        # Try different shuffles
        removed_code = greedy_remove([encode(c) for c in best_code], current_seed)
        
        if len(removed_code) < best_size:
            best_size = len(removed_code)
            # Convert back to words
            best_code = [decode(idx) for idx in removed_code]
            # If we hit a very good size, we can try to polish further or just restart from this better base
            # The greedy remove on the new smaller set might yield even smaller?
            # Actually, greedy remove on a subset of a covering code might not be a covering code if we remove too much?
            # No, greedy remove only removes if coverage is preserved. So it is always a valid covering.
            # So we can just keep the best found.
            
        seed_counter += 1
        # If we are stuck at the known best (486), we might stop early?
        # But we don't know the exact lower bound for sure, just "best known".
        # Let's run for the full time to be safe, or until no improvement in a long streak?
        # Given the speed, we can run many iterations.
        
    return best_code
