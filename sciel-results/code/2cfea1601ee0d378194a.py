import time
import random
import itertools

def construct(q, n, radius):
    # Parameters are fixed by the task, but we use them for generality
    # For this specific task: q=3, n=7, radius=1
    
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Precomputation
    # Map index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return t
        
    # Map tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors (covered words) for each codeword
    # covers[i] is a list of indices
    covers = [[] for _ in range(total_words)]
    # coverers[w] is a list of indices of codewords that cover w
    coverers = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        # self
        covers[i].append(i)
        coverers[i].append(i)
        
        # distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    j = tuple_to_idx(new_w)
                    covers[i].append(j)
                    coverers[j].append(i)
                    
    # 2. Greedy Initialization
    uncovered = set(range(total_words))
    cover_count = [0] * total_words
    selected = []
    
    # Greedy: pick codeword covering most uncovered
    while uncovered:
        best_c = -1
        best_gain = -1
        # Optimization: We can break early if we cover all remaining?
        # Not easily, but we can just iterate.
        for c in range(total_words):
            gain = 0
            for w in covers[c]:
                if w in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
            if best_gain == len(uncovered):
                break
        
        # Add best_c
        selected.append(best_c)
        for w in covers[best_c]:
            cover_count[w] += 1
            if cover_count[w] > 0:
                uncovered.discard(w)
                
    # 3. Local Search
    in_code = [False] * total_words
    for c in selected:
        in_code[c] = True
        
    best_size = len(selected)
    best_solution = selected.copy()
    
    # Helper to revert a swap/removal
    # We will modify state and revert if invalid
    
    while time.time() < deadline:
        if not selected: break
        
        # Randomly pick a codeword to try removing
        c_idx = random.randint(0, len(selected) - 1)
        c = selected[c_idx]
        
        # 1. Tentatively remove c
        for w in covers[c]:
            cover_count[w] -= 1
            
        uncovered_by_c = [w for w in covers[c] if cover_count[w] == 0]
        
        if not uncovered_by_c:
            # Pure removal successful
            selected.pop(c_idx)
            in_code[c] = False
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = selected.copy()
        else:
            # Need to swap: find c' that covers all words in uncovered_by_c
            # and c' is not already in code
            
            # Use intersection of coverers
            # Start with coverers of the first uncovered word
            # Convert to set for intersection
            candidates = set(coverers[uncovered_by_c[0]])
            
            for w in uncovered_by_c[1:]:
                candidates &= set(coverers[w])
                if not candidates:
                    break
            
            # Remove current codewords
            candidates -= set(selected)
            
            if candidates:
                # Pick a random candidate
                c_prime = random.choice(list(candidates))
                # Apply the swap: c is already removed (counts decremented).
                # Now add c_prime
                for w in covers[c_prime]:
                    cover_count[w] += 1
                # c_prime is now in code
                in_code[c_prime] = True
                # Replace c with c_prime in selected list
                selected[c_idx] = c_prime
                # Size remains same
            else:
                # Cannot swap, revert removal of c
                for w in covers[c]:
                    cover_count[w] += 1
                # in_code[c] is still True, selected still has c

    # Return the best solution found
    # If the current selected is better, use it, else best_solution
    # Actually, selected is always a valid cover. best_solution is the smallest found.
    # We should return the smallest valid cover.
    # If we found a smaller one, it's in best_solution.
    # But wait, if we did a swap, size didn't change.
    # If we did a removal, size decreased, and we updated best_solution.
    # So best_solution is the answer.
    
    return [idx_to_tuple(c) for c in best_solution]
