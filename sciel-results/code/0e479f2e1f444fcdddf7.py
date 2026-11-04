"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = q ** n
    
    # Map word tuple to index for quick access if needed, but we'll work with tuples or indices
    # Let's use indices for speed
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    # neighbors[i] is a set of indices of words within Hamming distance <= radius
    neighbors = [set() for _ in range(num_words)]
    
    for i, w in enumerate(all_words):
        # Distance 0
        neighbors[i].add(i)
        
        # Distance 1..radius
        # For efficiency, we can generate neighbors by changing positions
        # Given small n, we can just check all words? No, that's O(N^2).
        # Better: generate neighbors by mutation.
        
        # For radius 1, we just change one position.
        # For general radius, we can use BFS or just check distance if N is small.
        # N=729 is small. O(N^2) distance check is 729^2 ~ 530k ops. Very fast.
        pass
    
    # Actually, let's just compute distances on the fly or precompute properly.
    # Precomputing neighbors properly:
    for i, w in enumerate(all_words):
        for j, v in enumerate(all_words):
            if i == j:
                continue
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                neighbors[i].add(j)
                # Symmetry
                # neighbors[j].add(i) # We'll build full matrix or just use one side?
                # For greedy, we need to know which words a candidate covers.
                # So neighbors[i] should be the set of words covered by candidate i.
                
    # Wait, the loop above is O(N^2 * n). 729^2 * 6 ~ 3 million ops. This is fine in Python?
    # 3 million simple ops might take ~0.1-0.5s. Acceptable.
    
    # Let's optimize the neighbor generation slightly by breaking early if dist > radius
    neighbors = [set() for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].add(i)
        for j, v in enumerate(all_words):
            if i == j:
                continue
            dist = 0
            for k in range(n):
                if w[k] != v[k]:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                neighbors[i].add(j)
                
    uncovered = set(range(num_words))
    chosen = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_candidate = -1
        best_count = 0
        
        # Find candidate that covers most uncovered words
        # We can restrict candidates to those not yet chosen? 
        # Actually, we just pick the best from ALL words, but we only add if it improves.
        # Since we remove covered words, a previously chosen word covers 0 new words.
        # So we can iterate over all words, or just potential candidates.
        
        for i in range(num_words):
            # Optimization: if i is already chosen, skip? 
            # If i is chosen, its neighbors are already in `chosen`'s coverage? 
            # No, `uncovered` is the set of words NOT covered by any chosen word.
            # So `neighbors[i] & uncovered` will be empty for chosen i.
            # We can skip if `i` is in `chosen` set for speed, but checking intersection is fast.
            
            # Quick check: if no uncovered words in neighbors[i], skip
            # Intersection size calculation
            count = 0
            # Using set intersection
            # len(neighbors[i] & uncovered)
            # This creates a new set. 
            # For speed, we can iterate if sets are small? 
            # Neighbors size is ~13 for radius 1. 
            # So iterating over neighbors[i] and checking membership in uncovered is O(13).
            
            # Let's use iteration for small neighbor sets
            for idx in neighbors[i]:
                if idx in uncovered:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_candidate = i
                
        if best_candidate == -1 or best_count == 0:
            # Should not happen if uncovered is not empty and code is valid
            break
            
        chosen.append(best_candidate)
        # Remove covered words
        uncovered -= neighbors[best_candidate]
        
    # Convert indices back to word tuples/lists
    result = [list(all_words[i]) for i in chosen]
    
    return result
# EVOLVE-BLOCK-END
