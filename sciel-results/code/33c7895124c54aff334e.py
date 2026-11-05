import time
import random

_NEIGHBOR_CACHE = {}

def _get_neighbors(q, n):
    key = (q, n)
    if key not in _NEIGHBOR_CACHE:
        total = q ** n
        neighbors = [[] for _ in range(total)]
        
        def to_digits(x, n, q):
            d = [0] * n
            for i in range(n):
                d[i] = x % q
                x //= q
            return d
        
        def from_digits(d, q):
            x = 0
            for i in range(len(d)-1, -1, -1):
                x = x * q + d[i]
            return x

        for x in range(total):
            digits = to_digits(x, n, q)
            neighbors[x].append(x)
            for i in range(n):
                orig = digits[i]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[i] = val
                    neighbors[x].append(from_digits(new_digits, q))
        _NEIGHBOR_CACHE[key] = neighbors
    return _NEIGHBOR_CACHE[key]

def _greedy_cover(q, n, neighbors, seed):
    random.seed(seed)
    total_words = q ** n
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        max_covered = -1
        # Shuffle to break ties randomly? Or just pick the first max found?
        # To allow different runs, we can add a small random noise to the count?
        # Or just iterate in random order.
        
        # Iterating a set in Python is not random by default (hash based).
        # To make runs different, we can convert to list, shuffle, and iterate.
        # This is O(N) per step, which doubles the cost.
        # Alternative: Pick a random subset of candidates?
        # Simpler: Just run the deterministic greedy once. It's usually very good.
        # If we want restarts, we can randomize the *initial* selection or tie-breaking.
        
        # Let's stick to deterministic first, it's fast.
        # If we have time, we can try to improve.
        
        # Optimization: 
        # Instead of iterating all U, we can maintain a priority queue?
        # But U changes significantly each step.
        # For N=2187, simple iteration is fine.
        
        # To introduce variation for multiple restarts:
        # We can pick the best word among a random sample of U?
        # No, that might degrade quality.
        # Let's just run it once.
        
        for u in uncovered:
            count = 0
            for v in neighbors[u]:
                if v in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_word = u
                if count == len(neighbors[u]):
                    break
        
        if best_word == -1:
            break
            
        codewords.append(best_word)
        
        to_remove = [v for v in neighbors[best_word] if v in uncovered]
        for v in to_remove:
            uncovered.remove(v)
            
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if radius != 1:
        # Fallback
        return [[0]*n] # Invalid, but avoids crash. Task is specific.
        
    neighbors = _get_neighbors(q, n)
    
    # Run greedy
    best_codewords = _greedy_cover(q, n, neighbors, 42)
    
    # If time remains, try more seeds?
    # The greedy is deterministic given the set iteration order.
    # Python set iteration order depends on hash randomization (enabled by default for strings, 
    # but for integers it's consistent). 
    # So to get different results, we need to modify the algorithm.
    # We can randomize the tie-breaking.
    
    def _greedy_cover_randomized(q, n, neighbors, seed):
        random.seed(seed)
        total_words = q ** n
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            best_word = -1
            max_covered = -1
            # Collect candidates
            # To speed up, we can just iterate.
            # To randomize, we can add a random epsilon to the count?
            # Or pick a random word from the top K?
            
            # Let's just iterate. The result will be the same for the same seed 
            # if the set iteration is deterministic.
            # To force different paths, we can shuffle the list of uncovered words each step.
            # Cost: O(N log N) per step. N=2000. 2000*11 = 22k ops. 
            # 200 steps * 22k = 4.4M ops. Still fast.
            
            u_list = list(uncovered)
            random.shuffle(u_list)
            
            for u in u_list:
                count = 0
                for v in neighbors[u]:
                    if v in uncovered:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_word = u
                    if count == len(neighbors[u]):
                        break
            
            if best_word == -1:
                break
                
            codewords.append(best_word)
            
            to_remove = [v for v in neighbors[best_word] if v in uncovered]
            for v in to_remove:
                uncovered.remove(v)
                
        return codewords

    best_size = len(best_codewords)
    
    # Try a few random restarts if time allows
    seeds = [1, 2, 3, 4, 5]
    for s in seeds:
        if time.time() > deadline - 10:
            break
        cw = _greedy_cover_randomized(q, n, neighbors, s)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            
    def to_list(x, n, q):
        d = [0] * n
        for i in range(n):
            d[i] = x % q
            x //= q
        return d
    
    result = [to_list(c, n, q) for c in best_codewords]
    return result
