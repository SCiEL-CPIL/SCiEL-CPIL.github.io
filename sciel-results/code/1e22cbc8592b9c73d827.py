import itertools
import random
import math
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all words' digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Precompute spheres as lists of indices
    # sphere[i] contains indices of all words within Hamming distance radius of i
    spheres = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        base_digits = all_digits[i]
        sphere = set()
        for r in range(radius + 1):
            for pos_subset in itertools.combinations(range(n), r):
                if r == 0:
                    sphere.add(i)
                    continue
                values = range(q)
                for vals in itertools.product(values, repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    new_idx = digits_to_index(new_digits)
                    sphere.add(new_idx)
        spheres[i] = list(sphere)

    def fast_greedy(seed):
        rng = random.Random(seed)
        covered = [False] * total_words
        remaining = total_words
        codewords = []
        
        while remaining > 0:
            best_idx = -1
            best_count = -1
            candidates = []
            
            for i in range(total_words):
                new_count = 0
                for w in spheres[i]:
                    if not covered[w]:
                        new_count += 1
                
                if new_count > best_count:
                    best_count = new_count
                    candidates = [i]
                elif new_count == best_count and new_count > 0:
                    candidates.append(i)
            
            if best_count == 0:
                break
            
            best_idx = rng.choice(candidates)
            codewords.append(best_idx)
            
            for w in spheres[best_idx]:
                if not covered[w]:
                    covered[w] = True
                    remaining -= 1
        
        # Prune
        cover_count = [0] * total_words
        for c in codewords:
            for w in spheres[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords)):
                c = codewords[i]
                redundant = True
                for w in spheres[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in spheres[c]:
                        cover_count[w] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        return codewords

    def local_search(initial_code, time_limit=10.0):
        code_set = set(initial_code)
        cover_count = [0] * total_words
        for c in code_set:
            for w in spheres[c]:
                cover_count[w] += 1
        
        T = 1.0
        start_time = time.time()
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            T *= 0.999
            if T < 0.01:
                break
            
            # Pick a random index
            i = rng.randint(0, total_words - 1)
            
            if i in code_set:
                # Try to remove
                # Check if all w in sphere[i] have count > 1
                can_remove = True
                for w in spheres[i]:
                    if cover_count[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for w in spheres[i]:
                        cover_count[w] -= 1
                    code_set.remove(i)
                else:
                    # Reject
                    pass
            else:
                # Try to add
                # Energy increases by 1
                if rng.random() < math.exp(-1.0 / T):
                    for w in spheres[i]:
                        cover_count[w] += 1
                    code_set.add(i)
        
        # Final Prune
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for w in spheres[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in spheres[c]:
                        cover_count[w] -= 1
                    code_set.remove(c)
                    changed = True
        return list(code_set)

    # Get initial code
    best_initial = None
    best_size = float('inf')
    for seed in range(5):
        code = fast_greedy(seed)
        if len(code) < best_size:
            best_size = len(code)
            best_initial = code
    
    # Run Local Search
    best_code = local_search(best_initial)
    
    # If we have time, try another run with different start?
    # For now, just return the result.
    
    result = []
    for idx in best_code:
        result.append(index_to_digits(idx))
        
    return result
