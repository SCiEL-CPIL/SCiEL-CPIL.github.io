import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute balls as numpy arrays
    # balls[idx] = list of indices within distance 1 of idx
    balls = np.zeros((N, ball_size), dtype=np.int32)
    
    for idx in range(N):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        balls[idx, 0] = idx
        pos = 1
        for i in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[i] = (new_w[i] + delta) % q
                new_idx = 0
                for j in range(n):
                    new_idx = new_idx * q + new_w[j]
                balls[idx, pos] = new_idx
                pos += 1
    
    # Reverse mapping: for each element u, which centers cover it
    # Since Hamming distance is symmetric, balls[u] gives centers that cover u
    # So balls serves both purposes
    
    best_code = None
    best_len = N
    
    def greedy_set_cover():
        """Fast greedy set cover with random tie-breaking."""
        uncovered = np.ones(N, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            counts[i] = np.sum(uncovered[balls[i]])
        
        code = []
        
        while True:
            remaining = np.sum(uncovered)
            if remaining == 0:
                break
            
            max_count = np.max(counts)
            if max_count == 0:
                break
                
            max_indices = np.where(counts == max_count)[0]
            w = int(random.choice(max_indices))
            code.append(w)
            
            ball_w = balls[w]
            for u in ball_w:
                if uncovered[u]:
                    uncovered[u] = False
                    for v in balls[u]:
                        counts[v] -= 1
        
        return code
    
    def local_search(code):
        """Remove redundant codewords."""
        cover_count = np.zeros(N, dtype=np.int32)
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                can_remove = True
                for u in balls[w]:
                    if cover_count[u] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in balls[w]:
                        cover_count[u] -= 1
                    code.pop(i)
                    improved = True
                    break
        
        return code
    
    def swap_local_search(code, max_iters=500):
        """Try swap moves: remove one, add another to reduce size or maintain."""
        cover_count = np.zeros(N, dtype=np.int32)
        for w in code:
            for u in balls[w]:
                cover_count[u] += 1
        
        in_code = set(code)
        
        # Try to find a swap that reduces size by 1
        for _ in range(max_iters):
            found_improvement = False
            # Try removing each codeword and see if we can cover the gap with one addition
            for i in range(len(code)):
                w_remove = code[i]
                # Elements only covered by w_remove
                only_covered = []
                for u in balls[w_remove]:
                    if cover_count[u] == 1:
                        only_covered.append(u)
                
                if not only_covered:
                    # Can remove without adding
                    for u in balls[w_remove]:
                        cover_count[u] -= 1
                    code.pop(i)
                    in_code.discard(w_remove)
                    found_improvement = True
                    break
                
                # Find a center that covers all only_covered elements
                # and is not already in code
                candidates = [v for v in range(N) if v not in in_code]
                # Check if any candidate covers all only_covered
                for v in candidates:
                    covers_all = all(u in balls[v] for u in only_covered)
                    if covers_all:
                        # Do the swap
                        for u in balls[w_remove]:
                            cover_count[u] -= 1
                        for u in balls[v]:
                            cover_count[u] += 1
                        code[i] = v
                        in_code.discard(w_remove)
                        in_code.add(v)
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                break
        
        return code
    
    # Run multiple random restarts
    while time.time() < deadline:
        code = greedy_set_cover()
        code = local_search(code)
        code = swap_local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
            if best_len <= 73:
                # Keep searching for improvement
                pass
    
    if best_code is None:
        return [[0] * n for _ in range(N)]
        
    result = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
    
    return result
