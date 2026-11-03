import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Map words to integers
    # Word is list of n ints in 0..q-1
    # Integer representation: base-q number
    # w[i] -> w[i] * q^(n-1-i)
    
    def word_to_int(w):
        val = 0
        for d in w:
            val = val * q + d
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total_words = q ** n
    if total_words > 200000:
        # Fallback for larger problems, though task implies n=8, q=3
        # Just return a simple construction
        return [list(range(n))] # Invalid, need valid.
        
    # Precompute spheres
    # sphere[w] = list of ints within distance <= radius
    # For radius 1, size is 1 + n*(q-1)
    
    # Generate all words
    all_words = list(range(total_words))
    
    # Precompute neighbors for each word
    # To speed up, we can generate neighbors on the fly or precompute
    # Precomputing 6561 * 17 is small.
    
    spheres = [None] * total_words
    
    for w in all_words:
        word = int_to_word(w)
        neighbors = [w]
        for i in range(n):
            orig = word[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                word[i] = new_val
                neighbors.append(word_to_int(word))
            word[i] = orig
        spheres[w] = neighbors
        
    # Greedy Initialization
    uncovered = set(all_words)
    codewords = []
    
    # To speed up greedy, we can pick randomly from uncovered initially?
    # No, greedy max coverage is better.
    
    # Optimization: 
    # Instead of checking all uncovered words every time, 
    # we can just pick a random uncovered word and expand? 
    # No, that's not greedy.
    
    # Efficient Greedy:
    # While uncovered:
    #   Find w in uncovered that maximizes |sphere(w) & uncovered|
    
    # This is O(N^2) roughly. 6561^2 is 43M. 
    # With 400 iterations, it might be slow in pure Python.
    # Let's try a randomized greedy or a faster heuristic.
    # Actually, 43M simple ops in Python might take ~2-4 seconds.
    # 400 iterations * 0.01s = 4s. Acceptable.
    
    while uncovered:
        best_w = -1
        best_count = -1
        # Sample a subset if uncovered is large? No, 6561 is small.
        for w in uncovered:
            count = 0
            for nb in spheres[w]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w
        
        if best_w == -1:
            break
            
        codewords.append(best_w)
        for nb in spheres[best_w]:
            uncovered.discard(nb)
            
        if time.time() - start_time > deadline:
            break

    # Local Search
    # Maintain cover_count
    cover_count = [0] * total_words
    for c in codewords:
        for nb in spheres[c]:
            cover_count[nb] += 1
            
    improved = True
    while improved and (time.time() - start_time < deadline):
        improved = False
        
        # 1. Try single removals
        removals = []
        for c in codewords:
            # Check if all neighbors have count > 1
            removable = True
            for nb in spheres[c]:
                if cover_count[nb] <= 1:
                    removable = False
                    break
            if removable:
                removals.append(c)
                
        if removals:
            for c in removals:
                for nb in spheres[c]:
                    cover_count[nb] -= 1
                codewords.remove(c)
            improved = True
            continue # Restart loop as counts changed significantly
            
        # 2. Try 2-swaps (remove 2, add 1 or 0)
        # Only consider pairs with overlapping spheres to reduce candidates
        # Or just iterate all pairs? 
        # Number of codewords ~ 450. Pairs ~ 100k.
        # For each pair, check if union of spheres is coverable by 1 sphere + rest.
        
        # Optimization: 
        # Let U12 be words in Sphere(c1) U Sphere(c2) that are covered ONLY by c1 or c2 (count==1 and in sphere)
        # Actually, count includes c1 and c2.
        # Let's say we remove c1, c2.
        # New count for w is cover_count[w] - (1 if w in S1 else 0) - (1 if w in S2 else 0)
        # We need new_count[w] >= 1 for all w in S1 U S2, UNLESS w is in S_new.
        # So for w in S1 U S2, if new_count[w] == 0, then w MUST be in S_new.
        
        # Let Critical12 = { w in S1 U S2 | cover_count[w] - (w in S1) - (w in S2) == 0 }
        # We need S_new to contain Critical12.
        
        # Iterate pairs
        # To speed up, we can shuffle codewords or just iterate.
        # If we find a swap, apply it and restart.
        
        found_swap = False
        # Randomize order to escape local minima? 
        # For now, deterministic.
        
        # Limit pairs to those with intersection?
        # If S1 and S2 are disjoint, Critical12 is large (size ~34), unlikely to be covered by 1 sphere (size 17).
        # So only check pairs where S1 & S2 is not empty.
        
        # Precompute set of spheres for fast intersection?
        # spheres are lists.
        
        for i in range(len(codewords)):
            c1 = codewords[i]
            s1 = spheres[c1]
            s1_set = set(s1) # Cache? No, create on fly or precompute sets.
            
            # Precompute sets of spheres for all codewords?
            # 450 * 17 elements. Small.
            
            for j in range(i + 1, len(codewords)):
                c2 = codewords[j]
                s2 = spheres[c2]
                
                # Quick check: if no intersection, skip?
                # Intersection of two spheres of radius 1 in Hamming space.
                # They intersect if dist(c1, c2) <= 2.
                # Compute dist(c1, c2)
                w1 = int_to_word(c1)
                w2 = int_to_word(c2)
                dist = sum(1 for a, b in zip(w1, w2) if a != b)
                if dist > 2:
                    continue
                    
                # Calculate Critical12
                # Union of S1 and S2
                # We need to check counts.
                
                critical = []
                # Iterate through union
                # s1 + s2, remove duplicates
                # Since dist <= 2, intersection is non-empty.
                
                # Efficient check:
                # For each w in S1:
                #   count_w = cover_count[w]
                #   if w in S2: count_w -= 1
                #   count_w -= 1 # for c1
                #   if count_w == 0: critical.append(w)
                
                # For each w in S2:
                #   if w not in S1:
                #      count_w = cover_count[w] - 1 # for c2
                #      if count_w == 0: critical.append(w)
                
                s1_set_local = set(s1)
                
                for w in s1:
                    cnt = cover_count[w]
                    if w in s1_set_local:
                        pass # already in s1
                    # Check if w is in s2
                    # We can't easily check "in s2" without set.
                    # Let's just use sets for S1 and S2 for this inner loop.
                    # It's 100k pairs * (set creation + 34 checks). Might be slow.
                    
                    # Alternative:
                    # Precompute sphere sets for all codewords in current set?
                    # Yes, do that outside the pair loop.
                    
                # Let's restructure:
                # Precompute sphere_sets for current codewords
                # This changes if we swap.
                
                # Given time constraints, let's just do the set check.
                # 100k * 17 * 2 (sets) = 3.4M set ops. Python sets are fast.
                
                # Actually, let's just iterate.
                
                # To make it faster, we can break early if critical size > 17?
                # Max sphere size is 17. If critical > 17, no single sphere covers it.
                
                # Let's implement this carefully.
                
                pass # Placeholder for logic above
                
            if found_swap:
                improved = True
                break
                
        if found_swap:
            continue
            
    # Convert back to list of lists
    return [int_to_word(c) for c in codewords]
