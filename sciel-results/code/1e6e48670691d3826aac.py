import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial set: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Expand to length n
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n-4):
            initial_codewords.append(h + list(tail))
            
    # We will run multiple iterations of greedy pruning with random shuffles
    best_codewords = initial_codewords[:]
    
    total_words = q ** n
    
    def to_idx(word):
        idx = 0
        for val in word:
            idx = idx * q + val
        return idx
        
    def get_ball(word):
        # word is a list of ints
        yield tuple(word)
        for i in range(n):
            orig_val = word[i]
            for new_val in range(q):
                if new_val != orig_val:
                    new_word = word.copy()
                    new_word[i] = new_val
                    yield tuple(new_word)
                    
    def prune(codewords):
        # Returns pruned codewords
        coverage_count = [0] * total_words
        for c in codewords:
            for w in get_ball(c):
                coverage_count[to_idx(w)] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to break ties/symmetry?
            # If we shuffle the list of indices, we might remove different ones.
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    return codewords
                c = codewords[i]
                is_redundant = True
                for w in get_ball(c):
                    if coverage_count[to_idx(w)] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    for w in get_ball(c):
                        coverage_count[to_idx(w)] -= 1
                    codewords.pop(i)
                    changed = True
                    # Note: popping shifts indices, but we are iterating over a shuffled list of original indices.
                    # This is problematic. If we pop i, the element at i+1 moves to i.
                    # Our `indices` list contains original indices.
                    # If we pop i, subsequent indices in `indices` that are > i are now off by 1.
                    # To handle this correctly, we should not use original indices if we modify the list.
                    # Better: collect all redundant indices, then remove them?
                    # As discussed, removing multiple at once is risky.
                    # Alternative: iterate, if redundant, mark for removal. After full pass, remove all marked?
                    # Risky.
                    # Alternative: use a set of active indices?
                    
                    # Let's restart the scan after each removal to be safe.
                    # This is O(N^2) but N is ~700, so 700^2 = 490k ops, which is fast.
                    return prune(codewords) # Recursive? No, deep recursion risk.
                    
                    # Let's just break and restart the while loop.
                    break
        
        return codewords

    # Run prune a few times with different random seeds/shuffles
    # Since prune modifies the list in place and breaks on first removal, we need to be careful.
    # Let's implement a non-recursive version that restarts the scan.
    
    def prune_robust(codewords):
        coverage_count = [0] * total_words
        for c in codewords:
            for w in get_ball(c):
                coverage_count[to_idx(w)] += 1
                
        while time.time() < deadline:
            removed_any = False
            # We need to find a redundant codeword.
            # To avoid index shifting issues, we can find one, remove it, update counts, and restart the search.
            # This is effectively a depth-first search for a removable codeword.
            
            # To add randomness, we shuffle the codewords list itself?
            # No, let's just iterate.
            for i in range(len(codewords)):
                c = codewords[i]
                is_redundant = True
                for w in get_ball(c):
                    if coverage_count[to_idx(w)] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in get_ball(c):
                        coverage_count[to_idx(w)] -= 1
                    codewords.pop(i)
                    removed_any = True
                    break # Restart scan
                    
            if not removed_any:
                break
                
        return codewords

    # Try a few random permutations of the initial set to see if we get different results
    # The tetracode structure is symmetric, so shuffling might not help much unless we break the structure.
    # However, the "tail" part is just all combinations.
    
    # Let's just run the robust prune once.
    # To improve, we can try to swap codewords.
    # If we have a set C, and we remove c, we might be able to add a new codeword c' that covers the "lost" coverage more efficiently?
    # This is more complex. Let's stick to pruning first.
    
    current = initial_codewords[:]
    pruned = prune_robust(current)
    
    if len(pruned) < len(best_codewords):
        best_codewords = pruned
        
    return best_codewords
