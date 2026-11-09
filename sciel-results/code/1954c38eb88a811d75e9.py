def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 85
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    best_code = []
    best_size = 0
    
    indices = list(range(1, SIZE))
    indices.sort(key=lambda x: -weights[x])
    
    def try_greedy_linear(order, time_limit):
        current_code = [0]
        basis = []
        target_basis_size = 8
        
        for g in order:
            if time.time() > time_limit:
                break
            if len(basis) >= target_basis_size:
                break
                
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(g)
                new_words = [c ^ g for c in current_code]
                current_code.extend(new_words)
                
        return current_code

    # Try deterministic first
    code = try_greedy_linear(indices, deadline - 60) # Leave 60s for extension/other attempts
    if len(code) > best_size:
        best_code = code
        best_size = len(code)
        
    # Try a few random attempts if time permits
    # Each attempt takes some time.
    # We have ~60s left.
    # Let's try 5 random attempts.
    
    if best_size < 256:
        # If we didn't find 256, we might not have a good base.
        # But let's try harder.
        pass
        
    # If we have a 256 code, extend it.
    # We will do the extension in a helper function to keep it clean.
    
    def extend_code(C, time_limit):
        C_set = set(C)
        C_np = np.array(C, dtype=np.int32)
        
        # We need a list of candidates.
        # Using `indices` is fine.
        
        rounds = 0
        while time.time() < time_limit:
            rounds += 1
            current_candidates = [w for w in indices if w not in C_set]
            if not current_candidates:
                break
                
            random.shuffle(current_candidates)
            
            added_in_round = 0
            for w in current_candidates:
                if time.time() > time_limit:
                    break
                
                try:
                    dists = weights[w ^ C_np]
                    if dists.min() >= MIN_D:
                        C.append(w)
                        C_set.add(w)
                        C_np = np.append(C_np, w)
                        added_in_round += 1
                except:
                    pass
                    
            if added_in_round == 0:
                break
                
        return C

    if best_size >= 256:
        # We have a good base. Let's extend it.
        # We should do this on a copy so we can try other bases if we want?
        # But the extension is expensive.
        # Let's just extend the best one found so far.
        
        # To allow for multiple bases, we should structure it as:
        # 1. Find Base 1. Extend. Record Size.
        # 2. Find Base 2. Extend. Record Size.
        # ...
        
        # But finding a base takes time.
        # Let's try to find a few bases first, then extend the best one?
        # No, extending a "bad" base might yield a better final size than extending a "good" base?
        # Unlikely, but possible.
        # However, time is tight.
        
        # Strategy:
        # 1. Extend the current best_code (256 words).
        # 2. If time remains, try to find a different 256-word code and extend it.
        
        extended = extend_code(best_code, deadline - 5)
        if len(extended) > best_size:
            best_code = extended
            best_size = len(best_code)
            
        # If we still have time, try another linear code construction
        if time.time() < deadline - 30:
             # Try 2 more random linear constructions
             for _ in range(2):
                 if time.time() > deadline - 30: break
                 random.shuffle(indices) # This modifies indices in place!
                 # We need a copy or re-generate.
                 # indices is a list.
                 idx_copy = indices[:]
                 random.shuffle(idx_copy)
                 
                 code2 = try_greedy_linear(idx_copy, deadline - 30)
                 if len(code2) >= 256:
                     ext2 = extend_code(code2, deadline - 5)
                     if len(ext2) > best_size:
                         best_code = ext2
                         best_size = len(best_code)

    return best_code if best_code else [0]
