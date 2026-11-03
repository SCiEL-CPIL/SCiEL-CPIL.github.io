import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # For the specific task: q=3, n=6, radius=1
    # General implementation for small q, n
    if q * (q ** n) > 10**6:
        # Fallback to simple construction if space is too large for bitset greedy
        # This shouldn't happen for the given task (729)
        return [[0] * n]
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks
    # coverage[i] is an integer bitmask of points covered by word i
    coverage = [0] * N
    
    # For each word, find all words within Hamming distance <= radius
    for i, w in enumerate(all_words):
        mask = 0
        # Generate neighbors within radius
        # For radius 1, neighbors are w itself and w with one position changed
        if radius == 1:
            # Self
            mask |= (1 << i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_w = tuple(new_w)
                    j = word_to_idx[new_w]
                    mask |= (1 << j)
        else:
            # Generic radius check (slower, but N is small)
            for j, w2 in enumerate(all_words):
                dist = sum(1 for k in range(n) if w[k] != w2[k])
                if dist <= radius:
                    mask |= (1 << j)
        coverage[i] = mask
        
    full_mask = (1 << N) - 1
    
    best_solution = None
    best_size = N + 1
    
    # Number of restarts depends on time
    # Each restart is very fast (microseconds to milliseconds)
    # We can do thousands of restarts
    
    while time.time() < deadline:
        uncovered = full_mask
        solution = []
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            # Tie-breaking: randomize order of candidates to escape local minima
            # Or just pick first max. Randomizing candidates is better for diversity.
            candidates = list(range(N))
            random.shuffle(candidates)
            
            for idx in candidates:
                gain = bin(uncovered & coverage[idx]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            solution.append(best_idx)
            uncovered &= ~coverage[best_idx]
            
        # Local Search: Remove redundant codewords
        # Sort solution to make removal stable? No, just iterate.
        # We can try to remove any codeword that doesn't change the union of coverage.
        changed = True
        while changed:
            changed = False
            new_solution = []
            for i in range(len(solution)):
                # Check if removing solution[i] is safe
                # Union of coverage of all other codewords must cover coverage of solution[i]
                # Actually, simpler: compute union of all, then union of all except i.
                # If union_except_i == union_all, then i is redundant.
                
                # Optimization: 
                # Let U be union of all in solution.
                # Let U_i be union of all in solution except i.
                # If U_i == U, remove i.
                
                # To do this fast:
                # We can maintain the union incrementally, but for small N, recomputing is fine.
                # However, N is 729, solution size ~73. 
                # 73 * 73 * 1 (bit op) is fast.
                
                union_all = 0
                for idx in solution:
                    union_all |= coverage[idx]
                
                # Try removing solution[i]
                union_except_i = 0
                for j in range(len(solution)):
                    if j == i:
                        continue
                    union_except_i |= coverage[solution[j]]
                    
                if union_except_i == union_all:
                    changed = True
                    # Don't add solution[i] to new_solution
                    continue
                
                new_solution.append(solution[i])
            
            solution = new_solution
            
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Known best is 73. Lower bound 71.
            if best_size <= 73:
                # We might try to find 72 or 71, but 73 is the target.
                # Let's run a few more to see if 72 is possible, but 73 is a great result.
                # Actually, let's just keep searching until deadline to be safe.
                pass

    # Convert indices back to words
    if best_solution is None:
        # Fallback: return all words (invalid for score, but valid format)
        return [list(w) for w in all_words]
        
    return [list(all_words[idx]) for idx in best_solution]
