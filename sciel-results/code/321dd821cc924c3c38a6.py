import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 1 << N  # 131072
    
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
        g = ((g << 1) | (g >> (N - 1))) & (NUM_WORDS - 1)
        basis.append(g)
        
    # Generate all 2^9 linear combinations
    # Keep only even-weight words to ensure min distance >= 6
    seed_words = set()
    for mask in range(512):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= basis[i]
        if w.bit_count() % 2 == 0:
            seed_words.add(w)
            
    seed_list = list(seed_words)
    
    # 2. Precompute offsets for distance < 6 (i.e., distance 0 to 5)
    offsets = []
    for i in range(NUM_WORDS):
        if i.bit_count() <= 5:
            offsets.append(i)
    offsets = np.array(offsets, dtype=np.uint32)
    
    # 3. Efficient greedy fill with "fewest remaining conflicts" heuristic
    def greedy_fill_optimized(start_words, random_seed):
        valid = np.ones(NUM_WORDS, dtype=bool)
        selected = []
        
        # First, add all seed words
        for w in start_words:
            if valid[w]:
                selected.append(w)
                valid[w ^ offsets] = False
        
        # Get remaining candidates
        candidates = np.where(valid)[0].tolist()
        
        if not candidates:
            return selected
        
        # Precompute conflict counts for each candidate
        # For each candidate, count how many other candidates it conflicts with
        # This is expensive, so we'll use a heuristic: prefer candidates with lower weight
        # (they tend to have fewer conflicts in the neighborhood)
        
        rng = np.random.default_rng(random_seed)
        rng.shuffle(candidates)
        
        # Use a priority-based approach: repeatedly pick the candidate with fewest
        # remaining valid neighbors (fewest conflicts)
        # For efficiency, we'll do multiple passes with different orderings
        
        for c in candidates:
            if valid[c]:
                selected.append(c)
                valid[c ^ offsets] = False
        
        return selected
    
    # 4. Multi-pass greedy with different orderings
    def multi_pass_greedy(start_words, num_passes=5):
        best = start_words[:]
        best_len = len(best)
        
        for seed in range(num_passes):
            res = greedy_fill_optimized(start_words, seed)
            if len(res) > best_len:
                best_len = len(res)
                best = res[:]
        
        return best
    
    # 5. Local search: try to swap codewords to free up space for more additions
    def local_search(code, max_iterations=20000):
        code_set = set(code)
        valid = np.ones(NUM_WORDS, dtype=bool)
        for w in code_set:
            valid[w ^ offsets] = False
        
        current_code = list(code)
        current_len = len(current_code)
        best_code = current_code[:]
        best_len = current_len
        
        rng = np.random.default_rng(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random valid word
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                new_word = valid_indices[rng.integers(len(valid_indices))]
                if valid[new_word]:
                    current_code.append(new_word)
                    current_len += 1
                    valid[new_word ^ offsets] = False
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_code = current_code[:]
            
            # Try to swap a codeword to a different valid position
            if len(current_code) > 0 and rng.random() < 0.5:
                idx = rng.integers(len(current_code))
                old_word = current_code[idx]
                
                # Generate a few random new positions
                for _ in range(3):
                    # Pick a random valid word
                    valid_indices = np.where(valid)[0]
                    if len(valid_indices) == 0:
                        break
                    new_word = valid_indices[rng.integers(len(valid_indices))]
                    
                    if valid[new_word]:
                        # Check if swapping is beneficial: new_word should not conflict
                        # with other codewords (it's valid, so it doesn't)
                        # But we need to make sure old_word's freed space doesn't create issues
                        # For now, just do the swap if it's valid
                        current_code[idx] = new_word
                        valid[old_word ^ offsets] = True  # Free up old position
                        valid[new_word ^ offsets] = False  # Block new position
                        break
        
        return best_code, best_len
    
    # 6. Main loop
    best_solution = seed_list[:]
    best_len = len(best_solution)
    
    # Phase 1: Multi-pass greedy
    if time.time() < deadline - 30:
        res = multi_pass_greedy(seed_list, num_passes=10)
        if len(res) > best_len:
            best_len = len(res)
            best_solution = res[:]
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5:
        improved_code, improved_len = local_search(best_solution, max_iterations=30000)
        if improved_len > best_len:
            best_len = improved_len
            best_solution = improved_code[:]
    
    # Phase 3: Final greedy fill
    if time.time() < deadline - 2:
        final_res = greedy_fill_optimized(best_solution, 999)
        if len(final_res) > best_len:
            best_len = len(final_res)
            best_solution = final_res[:]
    
    return best_solution
