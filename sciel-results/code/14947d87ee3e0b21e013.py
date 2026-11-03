import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for other parameters
        return _generic_greedy(q, n, radius, deadline)

    N = q ** n
    # Precompute neighbors (ball of radius 1)
    # word is int 0..N-1
    neighbors = [None] * N
    
    # Helper to convert int to list of digits
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
        
    # Helper to convert list of digits to int
    def digits_to_int(digits, q):
        x = 0
        for d in reversed(digits):
            x = x * q + d
        return x

    # Precompute neighbors efficiently
    for w in range(N):
        neigh = [w]
        digits = int_to_digits(w, n, q)
        for i in range(n):
            orig = digits[i]
            for v in range(q):
                if v == orig: continue
                new_digits = digits[:]
                new_digits[i] = v
                neigh.append(digits_to_int(new_digits, q))
        neighbors[w] = neigh
    
    # Build reverse neighbors: for each word, which words cover it
    # neighbors[cw] = set of words covered by cw
    # reverse_neighbors[w] = set of cw such that w in neighbors[cw]
    reverse_neighbors = [set() for _ in range(N)]
    for w in range(N):
        for u in neighbors[w]:
            reverse_neighbors[u].add(w)
    
    covered = [False] * N
    cover_count = [0] * N
    code = []
    code_set = set()
    
    # Greedy Phase with sampling
    while True:
        if time.time() > deadline - 15.0: break
        if all(covered): break
        
        best_word = -1
        best_gain = -1
        
        # Optimization: Sample uncovered words to find good candidates
        uncovered_list = [w for w in range(N) if not covered[w]]
        if not uncovered_list:
            break
            
        if len(uncovered_list) > 200:
            sample = random.sample(uncovered_list, min(100, len(uncovered_list)))
        else:
            sample = uncovered_list
        
        candidates = set()
        for u in sample:
            candidates.update(reverse_neighbors[u])
        
        # Evaluate candidates
        for w in candidates:
            if w in code_set:
                continue
            gain = 0
            for x in neighbors[w]:
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
                
        if best_word == -1:
            # Fallback: pick any uncovered word
            best_word = uncovered_list[0]
            
        # Add best_word
        code.append(best_word)
        code_set.add(best_word)
        for x in neighbors[best_word]:
            if not covered[x]:
                covered[x] = True
            cover_count[x] += 1
            
        if time.time() > deadline - 15.0: break

    # Local Optimization: Redundancy Removal and Swapping
    changed = True
    while changed:
        if time.time() > deadline - 5.0: break
        changed = False
        
        # Try to remove redundant codewords
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for x in neighbors[c]:
                if cover_count[x] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for x in neighbors[c]:
                    cover_count[x] -= 1
                code.pop(i)
                code_set.remove(c)
                changed = True
                if time.time() > deadline - 5.0: break
        
        # Try swapping: replace a codeword with a better one
        if time.time() > deadline - 10.0: break
        
        # Pick a random codeword to try replacing
        if code:
            idx = random.randrange(len(code))
            old_cw = code[idx]
            
            # Words uniquely covered by old_cw
            would_uncover = [x for x in neighbors[old_cw] if cover_count[x] == 1]
            
            if would_uncover:
                # Find candidates that cover all would_uncover words
                # Start with neighbors of the first uncovered word
                first_uncovered = would_uncover[0]
                candidates = [w for w in reverse_neighbors[first_uncovered] if w not in code_set]
                
                # Filter candidates that cover all would_uncover words
                for cand in candidates:
                    covers_all = True
                    for x in would_uncover:
                        if x not in neighbors[cand]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Check if this candidate is better (covers more unique words)
                    # Count how many words would be newly uniquely covered
                    new_unique = 0
                    for x in neighbors[cand]:
                        if cover_count[x] == 0:  # Not covered at all
                            new_unique += 1
                    
                    # If new_unique > 0, this is an improvement
                    if new_unique > 0:
                        # Remove old_cw
                        for x in neighbors[old_cw]:
                            cover_count[x] -= 1
                        code.pop(idx)
                        code_set.remove(old_cw)
                        
                        # Add cand
                        code.append(cand)
                        code_set.add(cand)
                        for x in neighbors[cand]:
                            cover_count[x] += 1
                        changed = True
                        break

    # CA-based global reorganization to escape local minima
    # Each cell's state (in-code/out-code) updates based on coverage deficit
    # of its 17-neighborhood. This allows coordinated multi-swap moves.
    ca_changed = True
    ca_iterations = 0
    max_ca_iterations = 50
    while ca_changed and ca_iterations < max_ca_iterations:
        if time.time() > deadline - 3.0: break
        ca_changed = False
        ca_iterations += 1
        
        # Compute coverage deficit for each word:
        # deficit[w] = 1 if w is uncovered, 0 if covered
        # But we want to use this to decide CA updates
        
        # CA rule: A cell w (potential codeword) becomes in-code if:
        # - It's currently out-of-code AND
        # - Its neighborhood has high coverage deficit (many uncovered words)
        # A cell w becomes out-of-code if:
        # - It's currently in-code AND
        # - Its neighborhood is well-covered (all words have cover_count >= 2)
        
        # For efficiency, only consider cells near uncovered words
        uncovered_words = [w for w in range(N) if cover_count[w] == 0]
        if not uncovered_words:
            break
        
        # Build set of candidate cells: reverse neighbors of uncovered words
        candidate_cells = set()
        for u in uncovered_words:
            candidate_cells.update(reverse_neighbors[u])
        
        # Remove cells already in code
        candidate_cells -= code_set
        
        if not candidate_cells:
            break
        
        # For each candidate cell, compute the number of uncovered words it would cover
        # and the number of words that would become uncovered if we remove a codeword
        # We use a simple heuristic: add cells that cover many uncovered words,
        # remove cells that are redundant
        
        # Phase 1: Try to add cells that cover uncovered words
        added_cells = []
        for cand in list(candidate_cells):
            if time.time() > deadline - 3.0: break
            if cand in code_set:
                continue
            # Count how many uncovered words this cell covers
            covers_uncovered = sum(1 for x in neighbors[cand] if cover_count[x] == 0)
            if covers_uncovered >= 3:  # Threshold: must cover at least 3 uncovered words
                # Check if adding this cell is beneficial overall
                # (it covers uncovered words, but we need to check if it causes issues)
                added_cells.append((cand, covers_uncovered))
        
        # Sort by coverage benefit and add top cells
        added_cells.sort(key=lambda x: -x[1])
        for cand, _ in added_cells[:10]:  # Add at most 10 cells per iteration
            if cand in code_set:
                continue
            code.append(cand)
            code_set.add(cand)
            for x in neighbors[cand]:
                cover_count[x] += 1
            ca_changed = True
        
        # Phase 2: Try to remove redundant cells
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 3.0: break
            c = code[i]
            redundant = True
            for x in neighbors[c]:
                if cover_count[x] == 1:
                    redundant = False
                    break
            if redundant:
                for x in neighbors[c]:
                    cover_count[x] -= 1
                code.pop(i)
                code_set.remove(c)
                ca_changed = True

    # Final redundancy removal pass
    changed = True
    while changed:
        if time.time() > deadline - 2.0: break
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for x in neighbors[c]:
                if cover_count[x] == 1:
                    redundant = False
                    break
            if redundant:
                for x in neighbors[c]:
                    cover_count[x] -= 1
                code.pop(i)
                code_set.remove(c)
                changed = True

    # Convert back to list of lists
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
        
    return result

def _generic_greedy(q, n, radius, deadline):
    """Simple greedy fallback for other parameters."""
    N = q ** n
    neighbors = [None] * N
    
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
        
    def digits_to_int(digits, q):
        x = 0
        for d in reversed(digits):
            x = x * q + d
        return x

    for w in range(N):
        neigh = [w]
        digits = int_to_digits(w, n, q)
        if radius >= 1:
            for i in range(n):
                orig = digits[i]
                for v in range(q):
                    if v == orig: continue
                    new_digits = digits[:]
                    new_digits[i] = v
                    neigh.append(digits_to_int(new_digits, q))
        neighbors[w] = neigh
    
    covered = [False] * N
    cover_count = [0] * N
    code = []
    code_set = set()
    
    while True:
        if time.time() > deadline - 5.0: break
        if all(covered): break
        
        best_word = -1
        best_gain = -1
        
        for w in range(N):
            if w in code_set:
                continue
            gain = 0
            for x in neighbors[w]:
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
                
        if best_word == -1:
            break
            
        code.append(best_word)
        code_set.add(best_word)
        for x in neighbors[best_word]:
            if not covered[x]:
                covered[x] = True
            cover_count[x] += 1
            
    # Redundancy removal
    changed = True
    while changed:
        if time.time() > deadline - 2.0: break
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for x in neighbors[c]:
                if cover_count[x] == 1:
                    redundant = False
                    break
            if redundant:
                for x in neighbors[c]:
                    cover_count[x] -= 1
                code.pop(i)
                code_set.remove(c)
                changed = True
    
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))
    return result
