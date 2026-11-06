import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N
    
    # 1. Generate the 256-word linear seed code
    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Generator vector g as an integer
    g = 0
    for q in Q:
        g |= (1 << q)
        
    # Basis: cyclic shifts of g
    basis = [g]
    for i in range(1, 9):
        # Rotate left by 1
        g = ((g << 1) | (g >> (N - 1))) & (NUM_WORDS - 1)
        basis.append(g)
        
    # Generate all 2^9 linear combinations
    # We need the ones with bit 0 == 0 (shortening)
    seed_words = set()
    for mask in range(512):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        # Shortening: keep only if bit 0 is 0
        if (w & 1) == 0:
            seed_words.add(w)
            
    seed_list = list(seed_words)
    
    # 2. Precompute offsets for distance < 6 (i.e., distance 0 to 5)
    # offsets[i] = 1 if popcount(i) <= 5 else 0
    # We only need the list of offsets where popcount <= 5
    offsets = []
    for i in range(NUM_WORDS):
        if i.bit_count() <= 5:
            offsets.append(i)
    offsets = np.array(offsets, dtype=np.uint32)
    
    # 3. Greedy Fill Function with Local Search (Swap)
    def greedy_fill(start_words, random_seed):
        valid = np.ones(NUM_WORDS, dtype=bool)
        selected = []
        
        # Add start words
        for w in start_words:
            if valid[w]:
                selected.append(w)
                # Invalidate w and all neighbors with dist < 6
                valid[w ^ offsets] = False
        
        # Get remaining valid candidates
        candidates = np.where(valid)[0]
        
        # Shuffle candidates with specific seed
        rng = np.random.default_rng(random_seed)
        rng.shuffle(candidates)
        
        # Greedy selection
        for c in candidates:
            if valid[c]:
                selected.append(c)
                valid[c ^ offsets] = False
        
        # Local Search: Try to improve by swapping
        # We keep the list of selected words
        # For each word in the code, try to remove it and add a better one?
        # Or simpler: try to add a word that conflicts with exactly one existing word,
        # and swap if the new word allows more additions?
        # Given time constraints, a simple random restart is often better than complex local search
        # unless the local search is very fast.
        
        # Let's try a simple "swap" improvement pass if time permits
        # But for now, let's rely on many fast restarts.
        
        return selected

    # 4. Main Loop with Restarts
    best_solution = seed_list[:]
    best_len = len(best_solution)
    
    # Estimate cost: 
    # Invalidation is ~9400 ops.
    # ~300 picks per run -> ~3M ops.
    # Python loop overhead ~300 iterations.
    # Should be < 0.1s per run.
    # 80s / 0.1s = 800 runs.
    
    run_id = 0
    while time.time() < deadline:
        # Run greedy fill starting from seed
        res = greedy_fill(seed_list, run_id)
        
        if len(res) > best_len:
            best_len = len(res)
            best_solution = res[:]
            
        run_id += 1
        
        # Check time more frequently
        if run_id % 5 == 0:
            if time.time() > deadline:
                break

    return best_solution
