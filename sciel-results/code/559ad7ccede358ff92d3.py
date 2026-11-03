import sys
import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    # This task is specifically for q=3, n=8, radius=1.
    # The evaluator passes these, but we can specialize for speed if they match,
    # or use a general logic. Given the specific constraints in the prompt
    # ("Football pool problem for 8 matches... K_3(8,1)"), we assume q=3, n=8, radius=1.
    # However, to be safe and follow the interface, we handle the general case 
    # but optimize heavily for the specific parameters provided in the task description.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a simple construction if parameters differ
        if n < 4:
             return [list(c) for c in itertools.product(range(q), repeat=n)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def tuple_to_idx(t):
        return sum(t[i] * powers[i] for i in range(n))
        
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        t = idx_to_tuple(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for val in range(q):
                if val != orig:
                    t[pos] = val
                    j = tuple_to_idx(t)
                    neighbors[i].append(j)
            t[pos] = orig
            
    # Precompute covers: covers[j] is list of i such that j in neighbors[i]
    # This is the inverse of neighbors.
    covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            covers[j].append(i)
            
    # 1. Greedy Initialization
    uncovered = [True] * total_words
    uncovered_count = total_words
    code = []
    coverage_count = [0] * total_words
    
    # To speed up greedy, we can pre-sort candidates by potential coverage,
    # but standard greedy is O(N * K * |N|). N=6561, K~480, |N|=17. ~50M ops.
    # We can optimize by only checking candidates that cover at least one uncovered word.
    
    # For q=3, n=8, a simple greedy is fast enough.
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        best_c = -1
        best_gain = -1
        
        # Optimization: Only consider words that cover at least one uncovered word.
        # We can iterate through all words, but skip if gain is 0.
        # To make it faster, we can maintain a list of uncovered words and pick one,
        # then only consider its neighbors.
        # Let's pick a random uncovered word to break symmetry and guide the search.
        # Or just iterate all. Iterating all 6561 words * 17 neighbors is ~110k ops per step.
        # 480 steps * 110k = 52M ops. This is fine in Python if optimized or using NumPy.
        
        # Let's use a simple loop.
        for c in range(total_words):
            gain = 0
            for j in neighbors[c]:
                if uncovered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_gain == 0:
            break
            
        code.append(best_c)
        
        for j in neighbors[best_c]:
            coverage_count[j] += 1
            if coverage_count[j] == 1:
                uncovered[j] = False
                uncovered_count -= 1
                
    # 2. Local Search Optimization
    # We will use a loop that tries to remove codewords, then pairs, then swaps.
    
    improved = True
    # We need to be careful with time.
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Single Removal
        removed_any = False
        for c in list(code):
            is_redundant = True
            for j in neighbors[c]:
                if coverage_count[j] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                code.remove(c)
                for j in neighbors[c]:
                    coverage_count[j] -= 1
                removed_any = True
                break # Restart the pass after removal to update state cleanly
        
        if removed_any:
            improved = True
            continue
            
        # Pass 2: Pair Removal
        # Check if any pair (c1, c2) can be removed.
        # Condition: For all j in N(c1) U N(c2), coverage_count[j] >= 2 
        #            (considering that c1 and c2 are removed).
        # i.e., coverage_count[j] - (1 if j in N(c1) else 0) - (1 if j in N(c2) else 0) > 0
        
        found_pair = False
        code_len = len(code)
        # To speed up, we can skip pairs that are far apart? 
        # No, just check all. K~480. Pairs ~ 115k. 
        # Each check is O(|N(c1)| + |N(c2)|) ~ 34 ops.
        # 115k * 34 ~ 4M ops. Fast.
        
        for i in range(code_len):
            c1 = code[i]
            n1 = neighbors[c1]
            for j in range(i + 1, code_len):
                c2 = code[j]
                n2 = neighbors[c2]
                
                # Check if removing c1 and c2 leaves all covered
                valid = True
                # Check neighbors of c1
                for u in n1:
                    # coverage_count[u] includes c1 and potentially c2
                    # We need coverage_count[u] - 1 (for c1) - (1 if u in n2 else 0) > 0
                    # i.e., coverage_count[u] >= 1 + (1 if u in n2 else 0)
                    # If u is in n2, we need count >= 2. If not, count >= 1.
                    # Wait, if u is in n1, it is covered by c1.
                    # If u is also in n2, it is covered by c2.
                    # If we remove both, we need count(u) >= 2 if u in n2, else count(u) >= 1.
                    # Actually, simpler:
                    # new_count = coverage_count[u]
                    # if u in n2: new_count -= 1
                    # if new_count < 1: invalid
                    # But we also subtract c1.
                    # So: if u in n2: need coverage_count[u] >= 2. Else: need coverage_count[u] >= 1.
                    # Wait, if u in n1, it is covered by c1.
                    # If u in n2, it is covered by c2.
                    # If u in n1 AND u in n2, removing both removes 2 covers. Need count >= 2.
                    # If u in n1 AND u NOT in n2, removing c1 removes 1 cover. Need count >= 1.
                    
                    # So:
                    if u in n2: # This is O(|n2|) lookup, which is slow.
                        # Better to precompute a set for n2? Or just check membership.
                        # Since |n2|=17, membership check is fast.
                        if coverage_count[u] < 2:
                            valid = False
                            break
                    else:
                        if coverage_count[u] < 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Check neighbors of c2 that are NOT in n1
                for u in n2:
                    if u in n1:
                        continue # Already checked
                    # u is in n2 but not n1.
                    # Removing c2 removes 1 cover.
                    # Need coverage_count[u] >= 1.
                    if coverage_count[u] < 1:
                        valid = False
                        break
                        
                if valid:
                    # Remove c1 and c2
                    code.remove(c1)
                    code.remove(c2)
                    for u in n1:
                        coverage_count[u] -= 1
                    for u in n2:
                        coverage_count[u] -= 1
                    found_pair = True
                    break
                    
            if found_pair:
                break
                
        if found_pair:
            improved = True
            continue
            
        # Pass 3: Swap
        # Try to swap a codeword c with a new codeword c' to improve redundancy.
        # Goal: Increase the minimum coverage count, or just enable future removals.
        # Strategy: For each c, find c' that covers all uniquely covered words of c.
        # Among valid c', pick one that maximizes sum of coverage_count of its neighbors.
        
        swapped = False
        for idx, c in enumerate(code):
            n_c = neighbors[c]
            # Find uniquely covered words
            unique_words = []
            for u in n_c:
                if coverage_count[u] == 1:
                    unique_words.append(u)
                    
            if not unique_words:
                # Should have been removed in Pass 1
                continue
                
            # Find candidates c'
            # Candidates must be in the intersection of covers of all unique_words.
            # i.e., c' must be in covers[u] for all u in unique_words.
            
            # Pick the first unique word to generate candidates
            candidates = set(covers[unique_words[0]])
            for u in unique_words[1:]:
                candidates.intersection_update(covers[u])
                
            # Filter candidates: must not be in code
            code_set = set(code)
            valid_candidates = [c_prime for c_prime in candidates if c_prime not in code_set and c_prime != c]
            
            if not valid_candidates:
                continue
                
            # Score candidates
            best_c_prime = -1
            best_score = -1
            
            for c_prime in valid_candidates:
                # Score: Sum of coverage counts of neighbors of c_prime
                # This estimates how much "redundancy" we add.
                score = 0
                for u in neighbors[c_prime]:
                    score += coverage_count[u]
                    
                if score > best_score:
                    best_score = score
                    best_c_prime = c_prime
                    
            if best_c_prime != -1:
                # Perform swap
                code[idx] = best_c_prime
                # Update counts
                for u in n_c:
                    coverage_count[u] -= 1
                for u in neighbors[best_c_prime]:
                    coverage_count[u] += 1
                swapped = True
                break # Restart loop after swap
                
        if swapped:
            improved = True
            continue
            
        # If no improvement found in a full pass, do a random kick to escape local minimum
        if time.time() < deadline - 5:
            # Pick a random codeword
            idx = random.randint(0, len(code) - 1)
            c = code[idx]
            n_c = neighbors[c]
            unique_words = [u for u in n_c if coverage_count[u] == 1]
            
            if unique_words:
                candidates = set(covers[unique_words[0]])
                for u in unique_words[1:]:
                    candidates.intersection_update(covers[u])
                
                code_set = set(code)
                valid_candidates = [c_prime for c_prime in candidates if c_prime not in code_set and c_prime != c]
                
                if valid_candidates:
                    # Pick a random valid candidate
                    c_prime = random.choice(valid_candidates)
                    code[idx] = c_prime
                    for u in n_c:
                        coverage_count[u] -= 1
                    for u in neighbors[c_prime]:
                        coverage_count[u] += 1
                    # Do not set improved=True to avoid infinite loop of kicks without progress,
                    # but we break the current "no improvement" state.
                    # Actually, we should continue the main loop to see if this kick helps.
                    improved = True 
                    continue

    # Convert indices back to tuples
    result = [idx_to_tuple(c) for c in code]
    
    return result
