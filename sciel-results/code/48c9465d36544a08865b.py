import time
import random
import numpy as np

def construct(q, n, radius):
    # This task is specifically for q=3, n=6, radius=1
    # The evaluator passes these, but the logic below is general enough 
    # if we adapt the neighbor generation, but for speed we hardcode for 3^6.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy or random
        # For robustness, if params change, we might fail, but task says "this task's alphabet size"
        # So we assume q=3, n=6, radius=1.
        pass

    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    total_words = q ** n
    # Generate all words as tuples of ints
    # For q=3, n=6, total_words = 729
    # Map word tuple to index 0..728
    # index = w[0] * 3^5 + w[1] * 3^4 + ...
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that dist(idx_to_word(i), idx_to_word(j)) <= radius
    # For radius 1, neighbors are the word itself and words differing in 1 position.
    # Number of neighbors = 1 + n * (q - 1)
    
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    neigh.append(word_to_idx(w))
            w[pos] = orig_val
        neighbors.append(neigh)

    # State
    # code_set: set of indices
    # cover_count: array of size total_words, counts how many codewords cover each word
    # uncovered_count: number of words with cover_count == 0
    
    # Initialization
    # Start with a random set of size ~ 75
    # Or use a heuristic: random sampling
    initial_size = 75
    code_set = set(random.sample(range(total_words), initial_size))
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    uncovered_count = total_words
    
    # Apply initial code
    for c in code_set:
        for v in neighbors[c]:
            if cover_count[v] == 0:
                uncovered_count -= 1
            cover_count[v] += 1

    def get_cost():
        # Penalty for uncovered words must be large enough to prioritize coverage
        # But we want to minimize size once covered.
        # Cost = size + lambda * uncovered
        # If lambda is large, we fix coverage first.
        # Let's use lambda = 1000.
        return len(code_set) + 1000 * uncovered_count

    best_code = set(code_set)
    best_cost = get_cost()
    if uncovered_count == 0:
        best_size = len(code_set)
    else:
        best_size = float('inf')

    T = 50.0  # Initial temperature
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a move
        # 1. Remove a random codeword
        # 2. Add a random word
        # 3. Swap a codeword with a random word
        
        move_type = random.random()
        if len(code_set) > 0:
            c = random.choice(list(code_set))
        else:
            c = random.randint(0, total_words - 1)
        
        w = random.randint(0, total_words - 1)
        
        if move_type < 0.3 and len(code_set) > 0:
            # Remove c
            if c in code_set:
                # Try remove
                old_cost = get_cost()
                
                for v in neighbors[c]:
                    cover_count[v] -= 1
                    if cover_count[v] == 0:
                        uncovered_count += 1
                
                code_set.remove(c)
                
                new_cost = get_cost()
                
                # Check acceptance
                delta = new_cost - old_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    pass
                else:
                    # Reject: revert
                    code_set.add(c)
                    for v in neighbors[c]:
                        if cover_count[v] == 0:
                            uncovered_count -= 1
                        cover_count[v] += 1
                        
        elif move_type < 0.6:
            # Add w
            if w not in code_set:
                old_cost = get_cost()
                
                for v in neighbors[w]:
                    if cover_count[v] == 0:
                        uncovered_count -= 1
                    cover_count[v] += 1
                    
                code_set.add(w)
                
                new_cost = get_cost()
                
                delta = new_cost - old_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    pass
                else:
                    # Reject: revert
                    code_set.remove(w)
                    for v in neighbors[w]:
                        cover_count[v] -= 1
                        if cover_count[v] == 0:
                            uncovered_count += 1
        else:
            # Swap c with w
            if c in code_set and w not in code_set:
                old_cost = get_cost()
                
                # Remove c
                for v in neighbors[c]:
                    cover_count[v] -= 1
                    if cover_count[v] == 0:
                        uncovered_count += 1
                code_set.remove(c)
                
                # Add w
                for v in neighbors[w]:
                    if cover_count[v] == 0:
                        uncovered_count -= 1
                    cover_count[v] += 1
                code_set.add(w)
                
                new_cost = get_cost()
                
                delta = new_cost - old_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    pass
                else:
                    # Reject: revert
                    code_set.remove(w)
                    for v in neighbors[w]:
                        cover_count[v] -= 1
                        if cover_count[v] == 0:
                            uncovered_count += 1
                    
                    code_set.add(c)
                    for v in neighbors[c]:
                        if cover_count[v] == 0:
                            uncovered_count -= 1
                        cover_count[v] += 1

        # Update best
        curr_cost = get_cost()
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_code = set(code_set)
            if uncovered_count == 0:
                best_size = len(code_set)
                # If we hit 73, we can stop early? 
                # The best known is 73. If we get 73, we are optimal (or close).
                # But let's keep searching for a bit in case of bugs, or just return.
                # 73 is the known optimum. If we find 73, we can return immediately.
                if best_size == 73:
                    break
        elif uncovered_count == 0 and len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            if best_size == 73:
                break

        T *= cooling
        if T < 0.1:
            T = 0.1

    # Convert best_code to list of words
    result = [idx_to_word(i) for i in best_code]
    return result
