import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Problem parameters are fixed by the task: q=3, n=7, radius=1
    # But we use the arguments for generality if possible, though the logic is tuned for this case.
    if q != 3 or n != 7 or radius != 1:
        # Fallback to simple greedy if parameters differ
        return _generic_greedy(q, n, radius)

    deadline = time.time() + 110.0  # Safety margin
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Precompute neighbors (Hamming distance <= radius)
    # For efficiency, we can compute this on the fly or precompute indices.
    # Since N=2187, precomputing a list of neighbor indices for each word is fast.
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < q:
                    # Create new word
                    # Instead of creating new array, we can map to index
                    # But mapping is expensive. 
                    # Let's just store the indices by checking all words? No.
                    # Better: use a dictionary mapping tuple->index for fast lookup.
                    pass
        
    # Let's use a dictionary for index lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < q:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    idx = word_to_idx[tuple(new_word)]
                    neigh.append(idx)
        neighbors.append(neigh)
    
    # Initial Solution: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            init_indices.append(word_to_idx[tuple(word)])
    
    S = set(init_indices)
    best_S = set(S)
    best_size = len(S)
    
    # Coverage counts
    cover_count = np.zeros(N, dtype=np.int32)
    for j in S:
        for i in neighbors[j]:
            cover_count[i] += 1
            
    # Local Search
    while time.time() < deadline:
        # 1. Check validity and try to remove redundant
        removed_any = False
        redundant = []
        for j in S:
            # Check if all neighbors have count > 1
            # If count is 1 for any neighbor, j is essential
            is_redundant = True
            for i in neighbors[j]:
                if cover_count[i] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant.append(j)
        
        if redundant:
            for j in redundant:
                S.remove(j)
                for i in neighbors[j]:
                    cover_count[i] -= 1
            removed_any = True
            
        if len(S) < best_size:
            best_S = set(S)
            best_size = len(S)
        
        if not removed_any:
            # Try a swap
            # Pick a random j in S
            if S:
                j = random.choice(list(S))
                # Pick a random k not in S
                # To make it smarter, pick k that covers many words
                # For now, random from all words not in S
                # We can sample a few candidates
                best_k = None
                best_gain = -1
                # Sample 10 random candidates
                candidates = random.sample(range(N), min(10, N - len(S)))
                for k in candidates:
                    if k in S:
                        continue
                    # Estimate gain: how many words does k cover that are currently "critical" (count 1)?
                    # Or just how many words does it cover in total?
                    # Let's try to replace j with k if k covers words that j covers, and maybe more.
                    # Simple heuristic: if k covers at least as many words as j does, and some are critical, it's a candidate.
                    # Actually, let's just try the swap and see if it allows removals.
                    # But we need to be careful not to break coverage.
                    
                    # Let's just do a random swap to escape local minima
                    # If we break coverage, we will fix it in the next iteration by adding.
                    # But adding increases size. We want to keep size small.
                    
                    # Better swap heuristic:
                    # Find a k such that N(k) is a superset of N(j) \ {critical_words}
                    # This is complex.
                    
                    # Let's try: Swap j with k.
                    # Temporarily remove j, add k.
                    # Check if valid. If valid, accept.
                    # If invalid, revert.
                    
                    # To speed up, we can pre-compute if a swap is "promising".
                    # Let's just do random swaps and rely on the removal phase.
                    # If the swap breaks coverage, the next loop iteration will add a word, increasing size.
                    # This might oscillate.
                    
                    # Let's try a "kick": 
                    # 1. Remove j
                    # 2. Add k
                    # 3. If invalid, add best word to fix.
                    # 4. Then try to remove.
                    
                    # To avoid increasing size, we should only accept swaps that maintain validity.
                    # How to check validity fast?
                    # Remove j: decrement counts for neighbors of j.
                    # Add k: increment counts for neighbors of k.
                    # Check if any count is 0.
                    
                    for k in candidates:
                        if k in S:
                            continue
                        # Simulate swap
                        # We need to know if removing j and adding k keeps all counts >= 1.
                        # Words that become 0 are those in N(j) \ N(k) that had count 1.
                        # Words that become >= 1 are those in N(k) \ N(j).
                        
                        # Check critical words in N(j)
                        valid_swap = True
                        for i in neighbors[j]:
                            if cover_count[i] == 1:
                                # This word is only covered by j.
                                # So k MUST cover it.
                                if i not in neighbors[k]:
                                    valid_swap = False
                                    break
                        if valid_swap:
                            # Perform swap
                            S.remove(j)
                            for i in neighbors[j]:
                                cover_count[i] -= 1
                            S.add(k)
                            for i in neighbors[k]:
                                cover_count[i] += 1
                            break # Accept this swap and go to removal phase
                            
            if not S:
                # Should not happen, but reset if empty
                S = set(best_S)
                cover_count = np.zeros(N, dtype=np.int32)
                for j in S:
                    for i in neighbors[j]:
                        cover_count[i] += 1

    # Convert best_S to list of lists
    return [all_words[j].tolist() for j in best_S]

def _generic_greedy(q, n, radius):
    # Fallback
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    # Simple greedy
    covered = [False] * N
    S = []
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = []
        for j, v in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                neigh.append(j)
        neighbors.append(neigh)
        
    while any(not c for c in covered):
        best_idx = -1
        best_gain = -1
        for j in range(N):
            if j in S:
                continue
            gain = sum(1 for i in neighbors[j] if not covered[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        if best_idx == -1:
            break
        S.append(best_idx)
        for i in neighbors[best_idx]:
            covered[i] = True
            
    return [all_words[j] for j in S]
