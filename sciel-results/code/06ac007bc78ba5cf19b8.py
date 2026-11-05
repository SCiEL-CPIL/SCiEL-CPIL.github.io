import numpy as np
import time
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # 1. Precomputation
    def decode(val):
        res = [0] * n
        for i in range(n):
            res[i] = val % q
            val //= q
        return res
        
    def encode(tup):
        val = 0
        for i in range(n):
            val = val * q + tup[i]
        return val

    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        t = decode(w)
        neighbors[w].append(w)
        for i in range(n):
            orig = t[i]
            for new_val in range(q):
                if new_val == orig: continue
                t[i] = new_val
                neighbors[w].append(encode(t))
            t[i] = orig
            
    neighbors_np = np.array(neighbors, dtype=np.int32)
    num_neigh = neighbors_np.shape[1]
    
    best_code = []
    best_size = float('inf')
    
    def reduce_code(code):
        # code is a list of ints
        # Returns reduced list
        changed = True
        while changed:
            changed = False
            if not code:
                break
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in code:
                cover_count[neighbors_np[c]] += 1
                
            for i in range(len(code)):
                c = code[i]
                # If min count in neighbors is >= 2, c is redundant
                if cover_count[neighbors_np[c]].min() >= 2:
                    code.pop(i)
                    changed = True
                    break
        return code

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        uncovered = np.ones(total_words, dtype=np.int8)
        code = []
        
        while uncovered.sum() > 0:
            scores = uncovered[neighbors_np].sum(axis=1)
            max_score = scores.max()
            candidates = np.where(scores == max_score)[0]
            
            if len(candidates) > 1:
                best_c = candidates[np.random.randint(len(candidates))]
            else:
                best_c = candidates[0]
                
            code.append(best_c)
            uncovered[neighbors_np[best_c]] = 0
        return code

    def kick_and_reduce(code, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        if not code:
            return code
            
        # Make a copy
        current_code = list(code)
        
        # Pick a random codeword to remove
        idx = random.randint(0, len(current_code) - 1)
        c_old = current_code[idx]
        current_code.pop(idx)
        
        # Calculate uncovered holes
        # We need to know what is covered by current_code
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            cover_count[neighbors_np[c]] += 1
            
        # Holes are words with cover_count == 0
        holes = np.where(cover_count == 0)[0]
        
        if len(holes) == 0:
            # It was redundant, just return reduced version
            return reduce_code(current_code)
            
        # Greedily cover holes
        # We only care about covering the holes, but adding a codeword covers its neighbors.
        # We should prioritize codewords that cover many holes.
        
        # Create a boolean array for holes for fast lookup?
        # Or just intersect.
        
        while len(holes) > 0:
            # For each candidate c, count how many holes it covers
            # holes is an array of indices.
            # We can use neighbors_np[c] and check intersection with holes.
            
            # To speed up, we can create a set of holes
            holes_set = set(holes.tolist())
            
            best_c = -1
            max_covered = 0
            
            # We can restrict candidates to those that are "good"
            # Or just check all? 2187 * 15 checks is fine.
            for c in range(total_words):
                # Count how many neighbors of c are in holes_set
                # neighbors_np[c] is an array
                covered = 0
                for nb in neighbors_np[c]:
                    if nb in holes_set:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_c = c
                    
            if best_c == -1:
                break # Should not happen
                
            current_code.append(best_c)
            
            # Update holes
            # Remove neighbors of best_c from holes
            # holes = holes[~np.isin(holes, neighbors_np[best_c])]
            # Using set is faster for small sets
            for nb in neighbors_np[best_c]:
                holes_set.discard(nb)
            holes = np.array(list(holes_set), dtype=np.int32)
            
        return reduce_code(current_code)

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Initial run
    code = greedy_construct(42)
    code = reduce_code(code)
    best_code = code
    best_size = len(code)
    
    # Optimization loop
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        # Strategy: Alternate between random restarts and kicks on best
        if iter_count % 2 == 0 and len(best_code) > 0:
            # Kick
            seed = random.randint(0, 2**32)
            new_code = kick_and_reduce(best_code, seed)
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
        else:
            # Random Restart
            seed = random.randint(0, 2**32)
            code = greedy_construct(seed)
            code = reduce_code(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
    return [decode(c) for c in best_code]
