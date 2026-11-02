import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    N = q ** n
    
    # Generate all words as indices 0..N-1
    # idx_to_word: list of tuples
    idx_to_word = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute the ball matrix B where B[i, j] = 1 if dist(i, j) <= radius
    # For n=6, q=3, N=729. B is 729x729.
    # We can compute this by checking Hamming distance.
    # To be fast, we convert indices to base-q digits.
    words_np = np.array(idx_to_word, dtype=np.int8) # Shape (N, n)
    
    # Compute pairwise Hamming distances?
    # dist(i, j) = sum(words_np[i] != words_np[j])
    # We can do this in a vectorized way but it might be memory heavy if N is large.
    # For N=729, it's fine.
    # diff = words_np[:, np.newaxis, :] != words_np[np.newaxis, :, :]
    # dists = np.sum(diff, axis=2)
    # B = (dists <= radius).astype(np.uint8)
    
    # However, for radius=1, we can be smarter.
    # Ball of i contains i and neighbors.
    # Let's just build the list of neighbors for each i.
    # neighbors[i] is a list of indices.
    neighbors = [None] * N
    for i in range(N):
        w = idx_to_word[i]
        nbs = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                # Find index of new_w
                # Since we generated in product order, we can calculate index directly
                # Index is base-q number.
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                nbs.append(idx)
        neighbors[i] = nbs
        
    # Convert neighbors to a numpy array for fast operations?
    # Or just use the lists.
    # For greedy, we need to update covered status.
    # covered = np.zeros(N, dtype=bool)
    
    def greedy_cover(rng=None):
        covered = np.zeros(N, dtype=bool)
        code = []
        remaining = N
        
        while remaining > 0:
            # Find codeword c that maximizes number of new covers
            # new_covers[c] = sum(not covered[v] for v in neighbors[c])
            # We can compute this by iterating candidates.
            # With N=729, iterating 729 candidates and checking ~13 neighbors each is ~10k ops.
            # 73 steps * 10k = 730k ops. Very fast.
            
            best_c = -1
            best_new = 0
            
            # Optimization: We can precompute the "uncovered" mask and use it.
            # But simple loop is likely fast enough in Python for N=729.
            for c in range(N):
                # Skip if c is already covered? No, c must cover itself.
                # But if c is covered, it might still cover others.
                # However, if c is covered, adding it doesn't cover c itself.
                # But it might cover other uncovered words.
                
                new_covers = 0
                for v in neighbors[c]:
                    if not covered[v]:
                        new_covers += 1
                        if new_covers > best_new:
                            # Early break? No, we need max.
                            pass
                
                if new_covers > best_new:
                    best_new = new_covers
                    best_c = c
                    if best_new == len(neighbors[c]): # Max possible
                        # Can't do better for this c, but maybe another c has more total neighbors?
                        # No, ball size is constant for this problem (mostly).
                        # Actually ball size is constant 13.
                        # So if we find 13, it's the max.
                        break
            
            if best_c == -1:
                break # Should not happen if remaining > 0
                
            code.append(best_c)
            for v in neighbors[best_c]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
            
            if time.time() > deadline - 5:
                break
        
        return code, covered

    def local_search(code):
        # code is a list of indices
        # 1. Remove redundant
        # Maintain cover_count
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for v in neighbors[c]:
                cover_count[v] += 1
        
        improved = True
        while improved:
            improved = False
            # Check removal
            for i in range(len(code)):
                c = code[i]
                # If min cover_count in neighbors[c] is >= 2, c is redundant
                min_count = min(cover_count[v] for v in neighbors[c])
                if min_count >= 2:
                    # Remove c
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                    code.pop(i)
                    improved = True
                    break # Restart scan
            
            if not improved and time.time() < deadline - 10:
                # Try swaps?
                # Swapping is complex to maintain cover_count efficiently.
                # Given the small size, we can try random swaps and re-verify.
                # But removal is the main driver for reduction.
                # If we are stuck, we might need to restart greedy with different tie-breaking.
                pass
        
        return code

    # Multi-start
    best_code = None
    best_size = N + 1
    
    rng = np.random.default_rng(42)
    
    # First run deterministic greedy
    code, covered = greedy_cover()
    code = local_search(code)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
    
    # Random restarts
    # Perturb greedy by randomizing tie-breaking
    attempts = 0
    while time.time() < deadline - 10 and attempts < 100:
        attempts += 1
        covered = np.zeros(N, dtype=bool)
        code = []
        remaining = N
        valid = True
        
        while remaining > 0:
            # Compute scores
            # To speed up, we can just pick a random one from the top few?
            # Or just randomize the order of candidates?
            # Let's try: Pick random candidate from those with max score.
            
            # Compute scores for all candidates
            # This is the bottleneck.
            # Let's try to optimize the score calculation.
            
            scores = np.zeros(N, dtype=np.int32)
            # Vectorized score calculation:
            # scores[c] = sum(uncovered[neighbors[c]])
            # We can build a matrix A where A[c, v] = 1 if v in neighbors[c].
            # Then scores = A @ uncovered
            # Let's precompute A (ball matrix) once outside?
            # Yes, let's do that.
            
            if attempts == 0: # Precompute on first iteration? No, better outside.
                pass
            
            # For now, keep the loop but make it faster.
            # Actually, let's precompute the ball matrix B (N x N) outside the loop.
            # B[i, j] = 1 if j in neighbors[i]
            # Then scores = B @ uncovered (where uncovered is 1/0 array)
            
            # I will refactor to precompute B.
            
            break # Placeholder to break out of inner while to implement B outside
            
        # Re-structure to precompute B
        
    # Refactored approach with precomputed B
    B = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for v in neighbors[i]:
            B[i, v] = 1
    
    def greedy_cover_fast():
        covered = np.zeros(N, dtype=np.uint8) # 0/1
        code = []
        remaining = N
        
        while remaining > 0:
            # scores[c] = sum(B[c, :] * (1 - covered))
            # uncovered = 1 - covered
            # scores = B @ uncovered
            uncovered = 1 - covered
            scores = B @ uncovered
            
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            # Get candidates with max score
            candidates = np.where(scores == max_score)[0]
            
            # Random tie-breaking
            c = rng.choice(candidates)
            
            code.append(c)
            
            # Update covered
            new_covers = B[c, :] * (1 - covered)
            covered += new_covers
            remaining -= np.sum(new_covers)
            
            if time.time() > deadline - 5:
                break
        
        return code

    # Re-run main loop with fast greedy
    best_code = None
    best_size = N + 1
    
    # Initial deterministic (use fixed rng or just first candidate)
    # Let's just run the fast greedy multiple times
    
    for i in range(100):
        if time.time() > deadline - 10:
            break
            
        code = greedy_cover_fast()
        
        # Local Search (Removal)
        # Convert code to set for fast lookup?
        # Use cover_count method
        if code:
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count += B[c, :]
            
            improved = True
            while improved:
                improved = False
                for idx in range(len(code)):
                    c = code[idx]
                    # Check if min cover_count in B[c, :] is >= 2
                    # We need to check neighbors of c
                    # min_count = np.min(cover_count[B[c, :] == 1])
                    # But B[c, :] == 1 is a boolean mask.
                    mask = B[c, :] == 1
                    if np.any(mask):
                        min_count = np.min(cover_count[mask])
                        if min_count >= 2:
                            # Remove
                            cover_count -= B[c, :]
                            code.pop(idx)
                            improved = True
                            break
                            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 73:
            break

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [idx_to_word[c] for c in best_code]
