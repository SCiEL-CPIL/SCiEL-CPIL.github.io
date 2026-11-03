import numpy as np
import itertools
import time

def construct(q, n, radius):
    # For this specific task q=3, n=6, radius=1
    # But we should handle general cases or at least the specific one robustly.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # However, the task description is specific to 6 matches, ternary, radius 1.
    # "Football pool problem for 6 matches: ternary covering code K_3(6,1)"
    # So q=3, n=6, radius=1 is fixed for this task instance.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the original direct sum if parameters change, 
        # though the task implies fixed parameters.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Precompute neighbors (sphere of radius 1)
    # For each word, neighbors are itself + 1 unit changes in each position
    # We store indices into all_words
    # Since all_words is a contiguous array of tuples flattened, we can map word to index.
    # Or simpler: just compute the actual words and check coverage.
    
    # Let's use indices for speed.
    # Map word (tuple) to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute sphere indices for each word
    spheres = []
    for i, w in enumerate(all_words):
        sphere = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    neighbor_idx = word_to_idx[tuple(w_list)]
                    sphere.append(neighbor_idx)
            w_list[pos] = orig_val
        spheres.append(sphere)
    
    covered = np.zeros(N, dtype=bool)
    code = []
    
    # Greedy Set Cover
    # To optimize, we can track the "value" of each word (number of uncovered neighbors).
    # Initially, value[i] = len(spheres[i]) = 13 for all i.
    # When we select a word c, we mark its sphere as covered.
    # For each word u, its value decreases by the number of its neighbors that are now covered (and were previously uncovered).
    # This is equivalent to: for each u, value[u] -= sum(covered[x] for x in spheres[u] if x was just marked).
    # Actually, simpler: just recompute or update carefully.
    
    # Given N=729, simple recomputation is fast enough.
    # But let's try to be smart.
    # values = np.array([len(s) for s in spheres])
    
    # Let's just use the simple loop, it's fast enough for 120s limit.
    # 73 iterations * 729 candidates * 13 neighbors = ~700k ops.
    
    while not np.all(covered):
        best_idx = -1
        max_uncovered = -1
        
        # Find the word that covers the most uncovered words
        for i in range(N):
            # If this word is already covered, it might still be a good center?
            # Actually, if a word is covered, it doesn't mean it can't be a center.
            # But usually, we pick centers from the space.
            # The score is how many *new* words it covers.
            sphere = spheres[i]
            # Count uncovered in sphere
            # Using numpy sum on boolean array slice is fast
            # But spheres[i] is a list of indices.
            # covered[sphere[i]] gives boolean array.
            count = np.sum(~covered[sphere])
            
            if count > max_uncovered:
                max_uncovered = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.append(list(all_words[best_idx]))
        
        # Mark sphere as covered
        covered[spheres[best_idx]] = True
        
    return code
