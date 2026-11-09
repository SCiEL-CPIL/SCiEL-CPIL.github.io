import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = q ** n
    
    # Precompute neighbors
    # For q=3, n=8, radius=1, each node has 17 neighbors (itself + 16)
    # We assume radius=1 and q=3 for optimization, but keep it general if possible?
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # But the specific problem is K_3(8,1).
    # I will assume q=3, n=8, radius=1 for the hard-coded optimizations if needed, 
    # but the code should work for the given parameters.
    
    # General neighbor generation
    def to_digits(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Calculate max neighbors for radius 1
    max_neigh = 1 + n * (q - 1)
    
    neighbors = np.zeros((N, max_neigh), dtype=np.int32)
    
    for i in range(N):
        digits = to_digits(i)
        neigh_list = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                digits[pos] = new_val
                new_val_int = 0
                mult = 1
                for d in digits:
                    new_val_int += d * mult
                    mult *= q
                neigh_list.append(new_val_int)
                digits[pos] = original
        neighbors[i, :len(neigh_list)] = neigh_list
    
    best_codewords = None
    best_size = N
    
    while time.time() < deadline:
        # Greedy Set Cover with random tie-breaking
        uncovered = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        codewords = []
        
        while np.any(uncovered):
            scores = uncovered[neighbors].sum(axis=1)
            scores[selected] = -1
            
            max_score = np.max(scores)
            if max_score < 0:
                break # Should not happen
            
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected[best_idx] = True
            codewords.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
        # Redundancy Removal
        cover_count = np.zeros(N, dtype=np.int32)
        for c in codewords:
            cover_count[neighbors[c]] += 1
        
        changed = True
        while changed:
            changed = False
            # Shuffle codewords to check in random order? 
            # Or just iterate.
            # Iterating in reverse might be slightly better if recent additions are more likely to be redundant?
            # Doesn't matter much.
            for idx in range(len(codewords) - 1, -1, -1):
                c = codewords[idx]
                if np.min(cover_count[neighbors[c]]) > 1:
                    cover_count[neighbors[c]] -= 1
                    codewords.pop(idx)
                    changed = True
                    break # Restart scan
                    
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert back to list of lists
    result = []
    for c in best_codewords:
        digits = [0] * n
        x = c
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
        
    return result
