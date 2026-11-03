import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0
    
    total_space = q ** n
    max_coverage = 1 + n * (q - 1) # 1 + 7*2 = 15 for q=3, n=7
    
    # Precompute cover masks
    # index is integer representation of word
    # word_to_idx: simple base-q conversion
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[n - 1 - i] = idx % q
            idx //= q
        return w

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover_mask[i]
    # cover_mask[i] is an integer with bits set for all words within distance <= radius of i
    cover_mask = [0] * total_space
    
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        cover_mask[i] = mask
        
    all_ones = (1 << total_space) - 1
    
    # Symmetry breaking: Fix first codeword to 0
    # This is valid because the graph is vertex-transitive.
    initial_uncovered = all_ones & ~cover_mask[0]
    initial_code = [0]
    initial_size = 1
    
    best_size = total_space # Upper bound
    best_code = None
    
    # We will use a recursive DFS with pruning
    # To avoid recursion depth issues (though depth is ~180, which is fine), 
    # and to allow time checking, we implement it carefully.
    
    # Helper: Get list of candidates that cover vertex v
    # We can precompute for each vertex v, the list of u that cover v.
    # This is the "reverse cover" or "dominated by" list.
    # Since radius is 1, u covers v if dist(u, v) <= 1.
    # This is just the neighbors of v (including v).
    # We can compute this on the fly or precompute.
    # Precomputing is faster.
    
    candidates_for_vertex = [[] for _ in range(total_space)]
    for u in range(total_space):
        mask = cover_mask[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates_for_vertex[v].append(u)
            m ^= lsb
            
    # Sort candidates for each vertex by their global coverage? 
    # No, we sort dynamically based on current uncovered set.
    
    def get_lower_bound(uncovered, current_size):
        # Returns lower bound on total size
        if uncovered == 0:
            return current_size
            
        uncovered_count = uncovered.bit_count()
        
        # Find first uncovered vertex v
        # We need to find the index of the least significant set bit
        # But "first" in terms of index? 
        # It doesn't matter which one we pick, as long as it's uncovered.
        # Let's pick the one with the smallest index to be deterministic, 
        # or just the LSB.
        v = (uncovered & -uncovered).bit_length() - 1
        
        # Find candidates u that cover v
        cands = candidates_for_vertex[v]
        
        # Find max gain among these candidates
        max_gain = 0
        for u in cands:
            gain = (cover_mask[u] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                
        # If max_gain is 0, it means v cannot be covered? 
        # This should not happen if we start with a valid cover.
        if max_gain == 0:
            return float('inf')
            
        # Lower bound logic:
        # The next codeword must be one of the candidates for v.
        # It will cover at most max_gain vertices from the current uncovered set.
        # Remaining vertices: uncovered_count - max_gain
        # Each subsequent codeword can cover at most max_coverage vertices.
        # So we need at least 1 + ceil((uncovered_count - max_gain) / max_coverage) more codewords.
        
        remaining = uncovered_count - max_gain
        if remaining <= 0:
            return current_size + 1
            
        additional = (remaining + max_coverage - 1) // max_coverage
        return current_size + 1 + additional
        
    # DFS implementation
    # We use a stack to avoid recursion limit, but recursion is cleaner.
    # Depth is ~180, so recursion is safe.
    
    def dfs(uncovered, current_code, current_size):
        nonlocal best_size, best_code
        
        if time.time() > deadline:
            return
            
        if uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
            return
            
        # Pruning
        lb = get_lower_bound(uncovered, current_size)
        if lb >= best_size:
            return
            
        # Find first uncovered vertex
        v = (uncovered & -uncovered).bit_length() - 1
        
        # Get candidates that cover v
        cands = candidates_for_vertex[v]
        
        # Score candidates by their gain on current uncovered set
        # We want to try the most effective ones first
        scored_cands = []
        for u in cands:
            gain = (cover_mask[u] & uncovered).bit_count()
            scored_cands.append((gain, u))
            
        # Sort by gain descending
        scored_cands.sort(reverse=True)
        
        for gain, u in scored_cands:
            if current_size + 1 >= best_size:
                break
                
            new_uncovered = uncovered & ~cover_mask[u]
            
            # Check if this move is worth exploring
            # Quick check: if new_uncovered is 0, we found a solution
            if new_uncovered == 0:
                if current_size + 1 < best_size:
                    best_size = current_size + 1
                    best_code = current_code + [u]
                continue
                
            # Recurse
            dfs(new_uncovered, current_code + [u], current_size + 1)
            
            # If time is up, stop
            if time.time() > deadline:
                return
                
    # Start DFS
    dfs(initial_uncovered, initial_code, initial_size)
    
    # If no solution found within time (should not happen for this size), 
    # fallback to greedy
    if best_code is None:
        # Greedy fallback
        uncovered = initial_uncovered
        code = initial_code[:]
        while uncovered:
            best_gain = -1
            best_idx = -1
            # Find any uncovered vertex
            v = (uncovered & -uncovered).bit_length() - 1
            cands = candidates_for_vertex[v]
            for u in cands:
                gain = (cover_mask[u] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = u
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~cover_mask[best_idx]
        best_code = code
        
    return [all_words[i] for i in best_code]
