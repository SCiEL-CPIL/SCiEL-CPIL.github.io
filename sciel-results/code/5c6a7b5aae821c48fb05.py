import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 145  # Safety margin from 150s
    
    # Total words in the space
    total_words = q ** n
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return tuple(t)
    
    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    # Helper to generate sphere of radius 1 around a codeword
    def get_sphere(codeword):
        # codeword is a tuple
        sphere = [tuple(codeword)]
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_cw = list(codeword)
                    new_cw[i] = val
                    sphere.append(tuple(new_cw))
        return sphere

    # Precompute spheres for all possible codewords?
    # 729 codewords * 19 words/sphere = ~14k tuples. This is small.
    # Let's precompute spheres as lists of indices for faster set operations.
    all_spheres = {}
    for idx in range(total_words):
        cw = idx_to_tuple(idx)
        sphere_indices = set()
        # Center
        sphere_indices.add(idx)
        # Neighbors
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = val
                    sphere_indices.add(tuple_to_idx(tuple(new_cw)))
        all_spheres[idx] = sphere_indices

    # Initial Code: Tetracode product
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = tuple(h + list(tail))
            initial_code_indices.append(tuple_to_idx(cw))
            
    # We work with a list of indices for codewords
    current_code = set(initial_code_indices)
    
    # Function to check if code covers all words and return coverage counts
    def get_coverage_counts(code_set):
        counts = [0] * total_words
        for cw_idx in code_set:
            for w_idx in all_spheres[cw_idx]:
                counts[w_idx] += 1
        return counts

    # Function to prune code: remove any codeword that is redundant
    def prune_code(code_set):
        changed = True
        while changed:
            changed = False
            for cw_idx in list(code_set):
                # Check if removing cw_idx leaves any word uncovered
                # We need to know coverage counts. 
                # Efficient way: 
                # 1. Calculate coverage counts for current set.
                # 2. For each cw, check if there is any word in its sphere 
                #    that has count 1 (only covered by this cw).
                # If no such word exists, remove cw.
                
                # To avoid recalculating full counts for every check, 
                # we can maintain counts. But since we modify the set, 
                # we recalculate after each removal or batch removal.
                # Given small size, recalculating is fine.
                
                # Optimization: 
                # Let's just try to remove and check.
                # Actually, let's compute counts once, then check all.
                pass
        
        # Better Prune Implementation:
        # 1. Compute counts.
        # 2. Find all removable codewords.
        # 3. Remove them.
        # 4. Repeat.
        
        while True:
            counts = get_coverage_counts(code_set)
            removable = []
            for cw_idx in code_set:
                is_redundant = True
                for w_idx in all_spheres[cw_idx]:
                    if counts[w_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    removable.append(cw_idx)
            
            if not removable:
                break
            
            for cw_idx in removable:
                code_set.remove(cw_idx)
                
    # Initial Prune
    prune_code(current_code)
    best_code = set(current_code)
    best_size = len(best_code)
    
    # Local Search
    # We try to swap a codeword to enable more pruning or directly reduce size.
    # Heuristic: 
    # 1. Pick a codeword c to remove.
    # 2. Identify words U that become uncovered (count 1 in current set).
    # 3. If U is empty, remove c (size decreases).
    # 4. If U is not empty, try to find a new codeword c' that covers all words in U.
    #    If found, replace c with c'. Then prune.
    #    If not found, try to find c' that covers max words in U. 
    #    This might not help immediately but could lead to a better configuration.
    #    However, swapping only helps if it enables pruning.
    #    So, we should only accept a swap if it allows us to remove another codeword later?
    #    Or simply: accept swap if it leads to a valid code of same size, 
    #    then try to prune. If pruning succeeds, size decreases.
    
    # To make it faster, we can limit the number of swaps per iteration.
    
    while time.time() - start_time < time_limit:
        # Try to find a reduction
        improved = False
        
        # 1. Try to remove any redundant codeword (should be none if pruned, but check)
        prune_code(current_code)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)
            improved = True
        
        # 2. Try swaps
        # Pick a random codeword to remove
        if current_code:
            c_to_remove = random.choice(list(current_code))
            
            # Calculate coverage counts before removal
            counts_before = get_coverage_counts(current_code)
            
            # Words that will become uncovered if we remove c_to_remove
            # These are words in sphere(c_to_remove) that have count 1
            U = []
            for w_idx in all_spheres[c_to_remove]:
                if counts_before[w_idx] == 1:
                    U.append(w_idx)
            
            if not U:
                # Can remove directly
                current_code.remove(c_to_remove)
                improved = True
                # Loop back to prune
                continue
            
            # Need to find a replacement c' that covers all words in U
            # Iterate through all possible codewords
            # Optimization: Only consider codewords that cover at least one word in U?
            # Or just iterate all 729.
            
            best_c_prime = None
            best_uncovered_count = len(U) + 1 # Initialize to worse than current
            
            # We want to find c' such that U is subset of sphere(c')
            # If multiple, pick one that might be "better" (e.g., covers more words with count 1?)
            # For now, any valid replacement is good to enable pruning.
            
            # To speed up, we can check candidates.
            # Note: c' cannot be in current_code (unless we are just moving, but we removed c)
            # Actually, c' can be any codeword. If c' is in current_code \ {c}, 
            # adding it does nothing (it's already there). 
            # So c' must be NOT in current_code.
            
            # Pre-filter: c' must cover all words in U.
            # We can check this quickly.
            
            candidates = []
            for c_prime in range(total_words):
                if c_prime in current_code:
                    continue
                # Check if c' covers all words in U
                # U is small (usually 1-5)
                covers_all = True
                for w_idx in U:
                    if w_idx not in all_spheres[c_prime]:
                        covers_all = False
                        break
                if covers_all:
                    candidates.append(c_prime)
            
            if candidates:
                # Pick a random candidate
                c_prime = random.choice(candidates)
                # Perform swap
                current_code.remove(c_to_remove)
                current_code.add(c_prime)
                # Prune to see if size decreases
                prune_code(current_code)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
                    improved = True
                # If size didn't decrease, we might still keep the swap 
                # to explore a different part of the solution space?
                # Or revert? 
                # If we keep it, we might get stuck in a local minimum of size 73 
                # but a different configuration might allow dropping to 72.
                # So we keep the swap even if size doesn't drop immediately, 
                # as long as it's a valid code.
                # But we must ensure we don't loop forever on same size.
                # Let's add a counter or just rely on randomization.
            else:
                # No single replacement covers all U.
                # Maybe we can remove c and add TWO codewords that cover U?
                # That would increase size by 1, which is bad.
                # So we revert (do nothing).
                pass
        
        # If no improvement in size, and we've been stuck, 
        # maybe restart with a slightly different initial code?
        # Or just continue random swaps.
        
        # To prevent infinite loop on same state, we can track visited states?
        # With 729 possible codewords and ~70 selected, state space is huge.
        # Randomization should be enough.
        
        # If we find a code of size 73, we can try to break early?
        # No, we want to find 72 or 71 if possible.
        # But 73 is the best known. 
        # If we find 73, we can stop after some time to save CPU?
        # The score is 1/size. 1/73 > 1/81.
        # Let's aim for the smallest possible.
        
        # Safety: If we find 73, we can relax the search a bit?
        # Or just keep going.
        
        # If time is running out, return best found.
        
    # Convert best_code indices back to list of lists
    best_codewords = []
    for idx in best_code:
        best_codewords.append(list(idx_to_tuple(idx)))
        
    return best_codewords
