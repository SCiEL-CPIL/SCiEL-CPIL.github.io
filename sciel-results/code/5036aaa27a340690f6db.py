import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    # Handle non-target parameters with a generic greedy fallback
    if q != 3 or n != 8 or radius != 1:
        N = q ** n
        if N > 1000000:
            return [[0]*n]
        
        words = []
        for i in range(N):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            w.reverse()
            words.append(tuple(w))
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = []
        for i, w in enumerate(words):
            neigh = set([i])
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    new_w = tuple(new_w)
                    neigh.add(word_to_idx[new_w])
            neighbors.append(list(neigh))
        
        uncovered = np.ones(N, dtype=bool)
        C = []
        start = time.time()
        limit = 100.0
        while np.any(uncovered):
            best_idx = -1
            best_cnt = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_cnt <= 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start > limit: break
            
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Generate all words
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(8):
            w.append(x % 3)
            x //= 3
        w.reverse()
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors efficiently
    neighbors = np.zeros((N, 17), dtype=int)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        cnt = 1
        for pos in range(8):
            for delta in [1, 2]:
                new_val = (w[pos] + delta) % 3
                new_w = list(w)
                new_w[pos] = new_val
                new_w = tuple(new_w)
                neighbors[i, cnt] = word_to_idx[new_w]
                cnt += 1

    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        while True:
            removed = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed:
                break
        return current_code, current_counts

    def calc_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def greedy_init():
        """Randomized greedy initialization with multiple strategies."""
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered):
            # Compute coverage for all candidates
            cov = np.sum(uncovered[neighbors], axis=1)
            # Weighted random selection: favor higher coverage
            weights = cov ** 2 + 1  # Quadratic weighting to prefer high coverage
            weights[weights <= 0] = 0
            if np.sum(weights) == 0:
                break
            # Normalize
            total = np.sum(weights)
            if total == 0:
                break
            probs = weights / total
            # Sample
            best_idx = np.random.choice(N, p=probs)
            if cov[best_idx] <= 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
        return C

    def local_search(C, counts, max_iter=5000):
        """Improved local search with swap and kick operations."""
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Reduction pass
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            
            if len(C) < 410:  # Good enough, keep polishing
                continue
            
            # Try swaps
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit - 5:
                    break
                
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 4:
                    continue
                
                # Find candidates covering all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                else:
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= set(neighbors[u].tolist())
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                if not candidates_set:
                    continue
                
                # Try multiple candidates
                cands = list(candidates_set)
                random.shuffle(cands)
                
                for c_prime in cands[:5]:
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                    
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved = True
                        break
                    else:
                        # Accept with some probability to escape local minima
                        if random.random() < 0.1:
                            C = new_C
                            counts = new_counts
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                # Random kick: remove a random codeword and add a random one
                for _ in range(3):
                    if time.time() - start_time > time_limit - 5:
                        break
                    if not C:
                        break
                    
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    c_prime = random.randint(0, N-1)
                    if c_prime in C:
                        continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                    
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(new_C) < best_size:
                            best_size = len(new_C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts
        
        return best_C, best_counts

    # Multiple restarts with different random seeds
    best_overall_C = None
    best_overall_size = float('inf')
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        random.seed(restart * 12345 + 777)
        np.random.seed(restart * 9999 + 111)
        
        # Greedy initialization
        C = greedy_init()
        counts = calc_counts(C)
        
        # Local search
        C, counts = local_search(C, counts)
        
        if len(C) < best_overall_size:
            best_overall_size = len(C)
            best_overall_C = C.copy()
        
        # Early termination if we reach known best
        if best_overall_size <= 403:
            break
    
    # Final reduction
    if best_overall_C is not None:
        best_overall_C, _ = reduce_code(best_overall_C, calc_counts(best_overall_C))
    
    return [list(words[i]) for i in best_overall_C]
