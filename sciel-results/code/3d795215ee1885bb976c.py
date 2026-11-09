import numpy as np
import itertools

def construct(q, n, radius):
    # Fallback for general parameters or if q!=3, n!=7, radius!=1
    if q != 3 or n != 7 or radius != 1:
        # Simple Greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        # Map words to indices
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        idx_to_word = {i: w for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = []
        for w in all_words:
            nbs = [w]
            for i in range(n):
                for d in range(q):
                    if d != w[i]:
                        new_w = list(w)
                        new_w[i] = d
                        nbs.append(tuple(new_w))
            neighbors.append([word_to_idx[nb] for nb in nbs])
        
        N = len(all_words)
        cover_count = np.zeros(N, dtype=np.int8)
        uncovered_mask = np.ones(N, dtype=bool)
        code_indices = []
        
        # Precompute neighbor lists as numpy arrays for faster indexing?
        # Or just use lists.
        
        while np.any(uncovered_mask):
            best_gain = -1
            best_idx = -1
            for i in range(N):
                if i in code_indices: # Optimization: skip already chosen? 
                    # Actually, we can choose the same word? No, codewords must be distinct.
                    # But if we pick a word, it's in code. We shouldn't pick it again.
                    # We can maintain a boolean array 'in_code'.
                    pass
                
                # Calculate gain
                # gain = number of neighbors of i that are currently uncovered
                nbs = neighbors[i]
                gain = np.sum(uncovered_mask[nbs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            
            # Update coverage
            nbs = neighbors[best_idx]
            cover_count[nbs] += 1
            uncovered_mask[nbs] = (cover_count[nbs] == 0)
            
        return [list(idx_to_word[i]) for i in code_indices]

    # Specific optimization for q=3, n=7, R=1 using Greedy Set Cover
    # 1. Generate all words
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # 2. Precompute neighbors for all words
    # Each word has 1 + 7*2 = 15 neighbors
    neighbors = []
    for w in all_words:
        nbs = [w]
        for i in range(7):
            for d in range(3):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    nbs.append(tuple(new_w))
        neighbors.append([word_to_idx[nb] for nb in nbs])
        
    # Convert neighbors to numpy arrays for faster vectorized operations if needed, 
    # but lists are fine for summing small arrays.
    # Let's stick to lists for simplicity and low overhead in the inner loop.
    
    N = 2187
    cover_count = np.zeros(N, dtype=np.int8)
    uncovered_mask = np.ones(N, dtype=bool)
    in_code = np.zeros(N, dtype=bool)
    code_indices = []
    
    # Greedy Loop
    while np.any(uncovered_mask):
        best_gain = -1
        best_idx = -1
        
        # To speed up, we can iterate only over candidates not in code?
        # But checking `if not in_code[i]` is cheap.
        for i in range(N):
            if in_code[i]:
                continue
                
            nbs = neighbors[i]
            # Count how many neighbors are currently uncovered
            gain = np.sum(uncovered_mask[nbs])
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        # Add best_idx to code
        in_code[best_idx] = True
        code_indices.append(best_idx)
        
        # Update coverage
        nbs = neighbors[best_idx]
        cover_count[nbs] += 1
        uncovered_mask[nbs] = (cover_count[nbs] == 0)
        
    # Optional: Local Search to improve further
    # We can try to remove codewords that are redundant.
    # This is similar to the seed's local search but applied to the greedy result.
    # Since the greedy result is likely ~190-200, and the seed (Direct Sum + LS) might be ~200-240,
    # this should be an improvement.
    
    # Let's add a simple removal pass.
    # Iterate through code, try to remove each one.
    # If removing it doesn't uncover any word, keep it removed.
    # This is O(K * N * R).
    
    # Re-verify cover counts for the greedy code
    # Actually, the loop above maintains cover_count correctly.
    
    # Removal Pass
    for i in range(len(code_indices) - 1, -1, -1):
        c_idx = code_indices[i]
        # Check if all neighbors of c_idx are covered by other codewords
        # i.e., cover_count[neighbor] > 1 for all neighbor in neighbors[c_idx]
        nbs = neighbors[c_idx]
        if np.all(cover_count[nbs] > 1):
            # Remove c_idx
            cover_count[nbs] -= 1
            code_indices.pop(i)
            
    return [list(idx_to_word[i]) for i in code_indices]
