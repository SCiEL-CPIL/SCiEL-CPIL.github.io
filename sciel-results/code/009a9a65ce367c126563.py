import itertools
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, we assume q=3, n=7, radius=1
    # But the function signature is general.
    # If parameters differ, we might need a fallback.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we only need to handle the specific case or a general case.
    # Given the time limit and complexity, I will specialize for q=3, n=7, r=1
    # and provide a fallback for other cases (e.g., greedy only).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy
        return _greedy_construct(q, n, radius)

    N = q ** n
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Build Ball Matrix: M[i, j] is True if word j is in ball of word i
    # M has shape (N, N)
    # For radius 1, M[i] is True for i and all neighbors.
    # We can construct this efficiently.
    # Actually, for q=3, n=7, radius=1, each ball has 1 + 7*2 = 15 words.
    # We can just iterate and set bits.
    
    M = np.zeros((N, N), dtype=bool)
    for i in range(N):
        M[i, i] = True
        w = words[i]
        for pos in range(n):
            for delta in [1, 2]: # 1 and -1 mod 3
                new_val = (w[pos] + delta) % q
                new_w = w.copy()
                new_w[pos] = new_val
                # Find index of new_w
                # Since words are generated in lexicographic order, 
                # index is base-q representation.
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                M[i, idx] = True

    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Precompute column sums of M? 
    # We want to maximize sum(M[i, uncovered])
    # This is equivalent to maximizing dot(M[i], uncovered)
    
    while uncovered.any():
        # Find best i
        # scores = M @ uncovered
        # best_i = np.argmax(scores)
        # This is O(N^2) per step. N=2187. Steps ~ 186.
        # 186 * 2187 * 2187 ~ 880M ops. Might be too slow in Python loop if not vectorized well.
        # But M @ uncovered is a matrix-vector product, which is fast in BLAS.
        
        scores = M @ uncovered
        best_i = np.argmax(scores)
        code.append(best_i)
        uncovered[ M[best_i] ] = False
        
        # Optimization: If we are getting stuck or time is up, break?
        # Greedy usually finishes fast.

    # 2. Local Search
    # Maintain cover_count: number of codewords covering each word
    cover_count = np.zeros(N, dtype=np.int32)
    for i in code:
        cover_count[ M[i] ] += 1
        
    improved = True
    start_time = time.time()
    
    while improved:
        improved = False
        
        # Pass 1: Single Removal
        # A codeword i is redundant if no word is covered ONLY by i.
        # i.e., for all j in M[i], cover_count[j] > 1 is NOT required.
        # We need: there is no j such that M[i, j] is True AND cover_count[j] == 1.
        # So, check if np.any(cover_count[M[i]] == 1) is False.
        
        for idx_in_code, i in enumerate(code):
            # Check if any word in ball i has count 1
            if not np.any(cover_count[M[i]] == 1):
                # Remove i
                code.pop(idx_in_code)
                cover_count[ M[i] ] -= 1
                improved = True
                break # Restart loop after modification
                
        if improved:
            # Check time
            if time.time() - start_time > 110:
                break
            continue

        # Pass 2: Pair Replacement
        # Try to replace i, j with k
        # Condition: For all w in (B_i U B_j) \ B_k, cover_count[w] > 1_{w in B_i} + 1_{w in B_j}
        
        len_code = len(code)
        # Randomize order? Or iterate?
        # To save time, maybe only check a subset of pairs?
        # But we need to be thorough.
        # We can vectorize the check for a fixed pair (i, j).
        
        for i_idx in range(len_code):
            i = code[i_idx]
            Bi = M[i]
            
            for j_idx in range(i_idx + 1, len_code):
                j = code[j_idx]
                Bj = M[j]
                
                # Union of balls
                # Words in Bi U Bj
                # We need to check candidates k.
                # A good candidate k is one that covers many of Bi U Bj.
                # Specifically, k should cover all "critical" parts of Bi U Bj.
                
                # Let's find the set of words that would be lost if we remove i and j, but don't add k.
                # Lost if w in (Bi U Bj) and w not in Bk.
                
                # Heuristic: Pick k that maximizes coverage of (Bi U Bj) \ (Bi \ Bj) ?
                # Actually, just iterate k? No, N is too big.
                # Candidates for k: Words that are in Bi U Bj?
                # If k is not in Bi U Bj, it doesn't help cover Bi U Bj.
                # So k must be in Bi U Bj? No, k's ball must cover Bi U Bj.
                # So k must be close to the words in Bi U Bj.
                
                # Let's restrict k to be in the union of balls of i and j?
                # If k is far, it won't cover the core of i or j.
                # Actually, k must be such that Bk covers (Bi U Bj) \ Bk.
                # This implies k is "close" to i and j.
                
                # Let's try candidates k from the union of the supports of Bi and Bj?
                # Support of Bi is the set of indices where Bi is True.
                # This set has size 15.
                # Union has size at most 30.
                # So we only check ~30 candidates for k.
                
                # Get indices of True in Bi and Bj
                # This is slow in Python loop.
                # M is dense-ish? No, sparse. 15/2187 ~ 0.7%.
                # np.nonzero is okay.
                
                # But doing this for every pair (i,j) is expensive.
                # 186^2 / 2 ~ 17000 pairs.
                # For each pair, find nonzero indices (fast), check 30 candidates (fast).
                # This should be fast enough.
                
                # Optimization: Precompute list of indices for each ball?
                # ball_indices[i] = np.nonzero(M[i])[0]
                # This is memory heavy (2187 * 15 ints) but fast access.
                
                # Let's compute ball_indices once outside the loop.
                # Wait, I'm inside the function. I should precompute before local search.
                
                # Let's refactor to precompute ball_indices.
                
                # For now, let's just do it inline but efficiently.
                
                # Actually, let's precompute `ball_indices` at the start of the function.
                
                # I will move the precomputation to the top of the function.
                
                # For this specific edit, I'll assume I can access a precomputed list.
                # Since I can only edit the function, I'll define `ball_indices` inside `construct` before the loops.
                
                # Let's restart the logic in my head to include precomputation.
                
                # I will replace the whole `construct` function body to include precomputation and optimized loops.
                
                # Wait, the directive says "Prefer several small edits".
                # But the current code is a mess of bitsets. I need to replace the core logic.
                
                # Let's do a big replace of the `construct` function.

    return [list(words[i]) for i in code]

def _greedy_construct(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Simple Greedy without bitsets for fallback
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        # Remove covered
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
    return [words[i] for i in code]
