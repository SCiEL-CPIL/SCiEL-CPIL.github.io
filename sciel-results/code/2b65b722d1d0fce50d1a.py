import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Precomputation
    # Map word (tuple) to index
    # Word is tuple of n ints in 0..q-1
    # Index is base-q representation
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(q**n)]
    word_to_id = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= radius)
    # For radius 1, neighbors are the word itself + n*(q-1) words with one digit changed
    neighbors = []
    for w in all_words:
        nbs = [word_to_id[tuple(w)]]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = w.copy()
                new_w[pos] = val
                nbs.append(word_to_id[tuple(new_w)])
        neighbors.append(nbs)
    
    neighbors_np = np.array(neighbors) # Shape (q^n, ball_size)
    
    # 2. Initial Construction
    # Use the tetracode construction from the seed, or a simple greedy start.
    # The seed's 243-word code is a valid start.
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                initial_code.append(word_to_id[tuple(w)])
    else:
        # Fallback for small n: greedy
        initial_code = []
        
    # Convert to list of indices
    current_code = list(set(initial_code))
    
    # State
    coverage_count = np.zeros(q**n, dtype=np.int32)
    for c in current_code:
        coverage_count[neighbors_np[c]] += 1
        
    best_code = current_code.copy()
    best_size = len(current_code)
    
    # Helper to update coverage
    def update_coverage(code, cov):
        cov[:] = 0
        for c in code:
            cov[neighbors_np[c]] += 1
        return cov

    # 3. Local Search with Kicks
    stuck_count = 0
    max_stuck = 20
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy Deletion
        # Try to remove codewords that don't leave any word uncovered
        # We iterate until no more deletions are possible
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias? Or just iterate.
            # We need to check if min coverage in ball > 1
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                # Check if removing c keeps all neighbors covered
                # If min coverage in N[c] is > 1, we can remove c
                min_cov = np.min(coverage_count[neighbors_np[c]])
                if min_cov > 1:
                    # Remove c
                    coverage_count[neighbors_np[c]] -= 1
                    current_code.pop(i)
                    changed = True
                    improved = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code.copy()
        
        if not improved:
            stuck_count += 1
        else:
            stuck_count = 0
            
        if stuck_count > max_stuck:
            # Phase 2: Kick
            # Remove k random codewords
            k = random.randint(5, 15)
            if len(current_code) > k:
                # Pick k random indices to remove
                remove_indices = random.sample(range(len(current_code)), k)
                # Sort to remove from end to maintain indices? Or just mark.
                # Easier: create a new list
                remove_set = set(remove_indices)
                new_code = []
                for i, c in enumerate(current_code):
                    if i not in remove_set:
                        new_code.append(c)
                    else:
                        coverage_count[neighbors_np[c]] -= 1
                        
                current_code = new_code
                
                # Phase 3: Greedy Repair
                # Find uncovered words
                uncovered = np.where(coverage_count == 0)[0]
                
                while len(uncovered) > 0 and time.time() < deadline:
                    # Find word w that covers most uncovered words
                    # We can check all q^n words
                    best_w = -1
                    best_count = 0
                    # Optimization: Only check words not already in code?
                    # Actually, adding a duplicate is useless.
                    # But checking all is 2187 * 15 ops, which is fast.
                    for w_idx in range(q**n):
                        if w_idx in current_code: # Skip if already present
                            continue
                        # Count how many of its neighbors are in uncovered
                        # uncovered is an array of indices
                        # We can use a set for O(1) lookup, but numpy boolean indexing is fast
                        # Let's use a boolean array for uncovered for speed
                        # Actually, `uncovered` changes each step.
                        
                        # Simple loop might be faster than numpy overhead for small N?
                        # N=2187, Ball=15.
                        # Let's use a set for uncovered for O(1) checks in inner loop?
                        # No, let's stick to vectorized if possible.
                        
                        # Create a boolean mask of uncovered
                        # This is O(N) per word, so O(N^2) total. 2187^2 ~ 4.7M. Acceptable.
                        
                        # Better: Precompute neighbor lists as numpy arrays.
                        # For each candidate w, count intersection with uncovered set.
                        
                        # Let's maintain a set `uncov_set` for O(1) lookup
                        pass
                    
                    # Let's optimize the search for best_w
                    # We maintain `uncov_set` as a set of indices
                    # This loop is the bottleneck.
                    
                    # Re-implementation of repair loop with Set
                    if 'uncov_set' not in locals():
                        uncov_set = set(uncovered.tolist())
                    
                    best_w = -1
                    max_covered = 0
                    for w_idx in range(q**n):
                        if w_idx in current_code:
                            continue
                        # Count neighbors in uncov_set
                        cnt = 0
                        for nb in neighbors_np[w_idx]:
                            if nb in uncov_set:
                                cnt += 1
                        if cnt > max_covered:
                            max_covered = cnt
                            best_w = w_idx
                            
                    if best_w == -1 or max_covered == 0:
                        # Should not happen if code is not perfect, but break if stuck
                        break
                        
                    # Add best_w
                    current_code.append(best_w)
                    coverage_count[neighbors_np[best_w]] += 1
                    
                    # Update uncov_set
                    for nb in neighbors_np[best_w]:
                        if coverage_count[nb] == 1: # Just became covered
                            uncov_set.discard(nb)
                    
                    # Check if all covered
                    if not uncov_set:
                        break
                        
                # After repair, we have a valid code.
                # Run Phase 1 (Deletion) again immediately
                improved = True
                stuck_count = 0
                
    return [idx_to_word(c) for c in best_code]

# Test locally? No, just output.
