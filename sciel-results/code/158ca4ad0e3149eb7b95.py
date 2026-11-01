import numpy as np
import scipy.sparse as sp
import time
import itertools
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate all words
    S = q ** n
    
    # If S is very large, this might timeout or OOM.
    # For n=8, q=3, S=6561.
    # If S > 10000, we proceed, but be aware of memory.
    # Sparse matrix of size S x S with ~17 non-zeros per row is small.
    
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors using BFS up to 'radius'
    neighbors = []
    for i, w in enumerate(words):
        visited = {i}
        queue = deque([(i, 0)])
        nbs_list = []
        while queue:
            curr_idx, dist = queue.popleft()
            nbs_list.append(curr_idx)
            if dist == radius:
                continue
            curr_w = words[curr_idx]
            for pos in range(n):
                orig_val = curr_w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Create new word tuple
                    # Optimization: avoid list copy if possible?
                    # n is small, list copy is fine.
                    new_w_list = curr_w.copy()
                    new_w_list[pos] = new_val
                    new_w_tuple = tuple(new_w_list)
                    if new_w_tuple in word_to_idx:
                        next_idx = word_to_idx[new_w_tuple]
                        if next_idx not in visited:
                            visited.add(next_idx)
                            queue.append((next_idx, dist + 1))
        neighbors.append(nbs_list)
        
    # Build sparse matrix A
    # A[i, j] = 1 if j is a neighbor of i
    row_indices = []
    col_indices = []
    data = []
    for i in range(S):
        for j in neighbors[i]:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = sp.coo_matrix((data, (row_indices, col_indices)), shape=(S, S)).tocsr()
    
    best_code = None
    best_len = float('inf')
    
    # Greedy Set Cover with Randomized Restarts
    # We use np.int8 for uncovered to allow direct dot product
    while time.time() < deadline:
        uncovered = np.ones(S, dtype=np.int8)
        code = []
        
        # Random seed for this run? 
        # np.random uses global state. 
        # To ensure diversity, we can just let it run.
        
        while np.any(uncovered):
            # Calculate gains
            gains = A.dot(uncovered)
            
            # Add small random noise to break ties
            # Max gain is at most len(neighbors[i]) <= 17 (for R=1)
            # Noise 0..0.1 is sufficient to break ties without changing order of significantly different gains
            gains += np.random.uniform(0, 0.1, size=S)
            
            idx = np.argmax(gains)
            
            # If no gain, break (should not happen)
            if gains[idx] == 0:
                break
                
            code.append(idx)
            
            # Update uncovered
            covered_indices = neighbors[idx]
            uncovered[covered_indices] = 0
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
    if best_code is None:
        # Fallback
        return [words[0]]
        
    return [words[i] for i in best_code]
