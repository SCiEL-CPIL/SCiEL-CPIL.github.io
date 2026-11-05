import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        # This task is specific to q=3, n=8, radius=1
        # But we must handle the general case or at least not crash
        # The contract says: "The evaluator passes this task's alphabet size q, length n and covering radius."
        # So we can assume q=3, n=8, radius=1 for the main logic, but keep a fallback.
        
        # Fallback: Greedy Set Cover
        total_words = q ** n
        # Map word tuple to index
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        # Precompute neighbors
        neighbors = {}
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = {idx}
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.add(word_to_idx(tuple(nw)))
            neighbors[idx] = list(nbrs)

        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            # Find word covering most uncovered
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                if idx in code: # Optimization: skip if already in code, though we can just check coverage
                    continue
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    
    # 1. Precomputation
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return tuple(w)

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices within distance 1 of i
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbrs.append(word_to_idx(tuple(nw)))
        neighbors[idx] = nbrs
        
    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    start_time = time.time()
    deadline = start_time + 140  # Leave 10s for safety
    
    best_code = None
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        # Greedy Construction
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        code_set = set()
        
        # Greedy Loop
        while np.any(uncovered):
            best_idx = -1
            best_gain = -1
            candidates = []
            
            # Optimization: We can pre-compute gains and update them, 
            # but scanning 6561 * 17 is ~110k ops, which is fast enough in Python if we do it carefully.
            # However, maintaining gains is faster.
            # Let's stick to simple scan for robustness, but optimize the inner loop.
            
            # Actually, let's use a slightly smarter greedy:
            # We can precompute the "ball" of each point.
            # gain[i] = sum(uncovered[nbr] for nbr in neighbors[i])
            
            # To speed up, we can just iterate.
            for i in range(total_words):
                if i in code_set:
                    continue
                gain = 0
                for nbr in neighbors[i]:
                    if uncovered[nbr]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                break
                
            # Randomly pick from top candidates to allow restarts to differ
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            code_set.add(best_idx)
            
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False

        # Local Search: Redundancy Removal and Swaps
        # Initialize cover counts
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            for nbr in neighbors[c]:
                cover_count[nbr] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Remove redundant codewords
            # A codeword c is redundant if all its neighbors are covered by other codewords.
            # i.e., for all nbr in neighbors[c], cover_count[nbr] >= 2 (since c itself contributes 1, if others contribute >=1, total >=2).
            # Wait, if we remove c, the count drops by 1. So we need cover_count[nbr] >= 2 for all nbr in neighbors[c].
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] < 2:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    code.pop(i)
                    code_set.discard(c)
                    improved = True
                    break # Restart scan as counts changed

            # 2. If no redundancy, try a swap
            if not improved and time.time() < deadline:
                # Pick a random codeword to remove
                if code:
                    idx_c = random.randint(0, len(code) - 1)
                    c = code[idx_c]
                    
                    # Temporarily remove c
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                        
                    # Find uncovered words
                    uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
                    
                    if not uncovered_now:
                        # Should have been removed in step 1, but just in case
                        code.pop(idx_c)
                        code_set.discard(c)
                        improved = True
                        continue
                        
                    # Find candidates that cover all uncovered_now
                    # Candidates must be in intersection of neighbor_sets[u] for u in uncovered_now
                    # And not in code_set
                    # And not c itself (though c is removed from set, we can add it back? No, we want a different word)
                    
                    # Optimization: If len(uncovered_now) > 1, the intersection is likely small.
                    # Start with the set of the first uncovered word
                    if uncovered_now:
                        candidates = neighbor_sets[uncovered_now[0]].copy()
                        for u in uncovered_now[1:]:
                            candidates.intersection_update(neighbor_sets[u])
                            if not candidates:
                                break
                        
                        if candidates:
                            # Remove existing code words
                            candidates.difference_update(code_set)
                            
                            if candidates:
                                # Pick a candidate
                                # Heuristic: Pick the one that maximizes the number of neighbors with cover_count >= 2
                                # This creates more redundancy for future removals.
                                best_cp = -1
                                best_score = -1
                                
                                # Limit candidates to speed up?
                                # If too many, pick random subset?
                                # For now, iterate all.
                                for cp in candidates:
                                    score = 0
                                    for nbr in neighbors[cp]:
                                        if cover_count[nbr] >= 2:
                                            score += 1
                                    if score > best_score:
                                        best_score = score
                                        best_cp = cp
                                        
                                if best_cp != -1:
                                    # Perform swap
                                    code[idx_c] = best_cp
                                    code_set.add(best_cp)
                                    for nbr in neighbors[best_cp]:
                                        cover_count[nbr] += 1
                                    improved = True
                                    # Note: We don't break out of the while loop immediately, 
                                    # but we restart the redundancy check.
                                else:
                                    # Revert
                                    for nbr in neighbors[c]:
                                        cover_count[nbr] += 1
                            else:
                                # Revert
                                for nbr in neighbors[c]:
                                    cover_count[nbr] += 1
                        else:
                            # Revert
                            for nbr in neighbors[c]:
                                cover_count[nbr] += 1
                    else:
                        # Revert
                        for nbr in neighbors[c]:
                            cover_count[nbr] += 1

        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # If we found a very good solution (e.g., close to known best 486), we might stop early?
        # Or just keep trying to improve.
        
    if best_code is None:
        # Fallback if something went wrong
        return [list(idx_to_word(0))]
        
    return [list(idx_to_word(c)) for c in best_code]
