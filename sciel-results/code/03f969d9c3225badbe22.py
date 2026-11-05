import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    neigh_set.add(word_to_idx[tuple(new_w)])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_list = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_list.append(word_to_idx[new_w])
        neighbors.append(neigh_list)
        
    # Precompute neighbor masks as integers for fast gain calculation
    # N=6561 requires 103 uint64s. Python int is arbitrary precision, so it works.
    neighbor_masks = [0] * N
    for i in range(N):
        mask = 0
        for p in neighbors[i]:
            mask |= (1 << p)
        neighbor_masks[i] = mask

    def greedy_init():
        uncovered_mask = (1 << N) - 1
        C = []
        start_time = time.time()
        
        while uncovered_mask:
            if time.time() - start_time > 110.0:
                break
            best_idx = -1
            best_gain = -1
            # Iterate all candidates
            for i in range(N):
                # Calculate gain: points in uncovered_mask that are also in neighbor_masks[i]
                gain = (uncovered_mask & neighbor_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
                
            C.append(best_idx)
            uncovered_mask &= ~neighbor_masks[best_idx]
            
        return C

    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            for p in neighbors[c]:
                counts[p] += 1
        return counts

    def reduce_code(code, counts):
        """
        Removes redundant codewords.
        A codeword c is redundant if all its neighbors are covered by at least 2 codewords.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        c_set = set(current_code)
        
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # Check if c is redundant
                redundant = True
                for p in neighbors[c]:
                    if current_counts[p] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for p in neighbors[c]:
                        current_counts[p] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
            c_set = set(current_code)
            
        return current_code, current_counts

    # Main Execution
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Greedy Initialization
    C = greedy_init()
    counts = get_counts(C)
    C, counts = reduce_code(C, counts)
    
    best_C = C.copy()
    best_counts = counts.copy()
    best_size = len(C)
    
    # 2. Local Search
    # We will perform random Add-Reduce steps.
    # To speed up "is in C", we maintain a set.
    C_set = set(C)
    
    # Precompute list of all indices for random choice
    all_indices = list(range(N))
    
    while time.time() - start_time < time_limit:
        # 1. Try 1-Add-Reduce
        # Pick a random candidate
        c_prime = random.choice(all_indices)
        
        # Skip if already in C (rare, but possible)
        if c_prime in C_set:
            continue
            
        # Backup
        old_C = C.copy()
        old_counts = counts.copy()
        old_size = len(C)
        
        # Add
        C.append(c_prime)
        C_set.add(c_prime)
        for p in neighbors[c_prime]:
            counts[p] += 1
            
        # Reduce
        C, counts = reduce_code(C, counts)
        C_set = set(C)
        
        # Evaluate
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            # Accept new state
        elif len(C) == best_size:
            # Accept with probability to allow random walk on plateau
            if random.random() < 0.5:
                # Accept
                pass
            else:
                # Revert
                C = old_C
                counts = old_counts
                C_set = set(C)
        else:
            # Revert
            C = old_C
            counts = old_counts
            C_set = set(C)
            
        # 2. Occasionally try 2-Add-Reduce
        if random.random() < 0.2:
            c1 = random.choice(all_indices)
            c2 = random.choice(all_indices)
            if c1 == c2 or c1 in C_set or c2 in C_set:
                continue
                
            old_C = C.copy()
            old_counts = counts.copy()
            old_size = len(C)
            
            C.append(c1)
            C.append(c2)
            C_set.add(c1)
            C_set.add(c2)
            for p in neighbors[c1]:
                counts[p] += 1
            for p in neighbors[c2]:
                counts[p] += 1
                
            C, counts = reduce_code(C, counts)
            C_set = set(C)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            elif len(C) == best_size:
                if random.random() < 0.3:
                    pass
                else:
                    C = old_C
                    counts = old_counts
                    C_set = set(C)
            else:
                C = old_C
                counts = old_counts
                C_set = set(C)

    # Final reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
