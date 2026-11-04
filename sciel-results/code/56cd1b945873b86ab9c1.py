import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    N = q ** n  # 2187 for q=3, n=7
    
    # Generate all words as tuples and map to indices
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Compute neighbors (ball of radius 1) for each word
    # Degree is 1 + n*(q-1) = 15 for q=3, n=7
    degree = 1 + n * (q - 1)
    neighbor_matrix = np.zeros((N, degree), dtype=np.int32)
    
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbor_matrix[i, :len(neigh)] = neigh
    
    # Precompute neighbor lists as Python lists for fast iteration in some contexts
    neighbor_lists = [neighbor_matrix[i].tolist() for i in range(N)]
    
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered_set, code, top_k=1, noise_scale=0):
        """Greedy set cover with randomization."""
        uncovered = uncovered_set
        while uncovered:
            if time.time() > deadline:
                break
            
            # Convert uncovered to boolean array
            uncovered_arr = np.zeros(N, dtype=np.uint8)
            for u in uncovered:
                uncovered_arr[u] = 1
            
            # Coverage for each candidate
            coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            
            # Add noise if specified
            if noise_scale > 0:
                coverage = coverage + np.random.normal(0, noise_scale, size=N)
            
            max_cov = coverage.max()
            if max_cov == 0:
                break
            
            # Get top-k candidates
            if top_k == 1:
                # Find all with max coverage
                best_candidates = np.where(coverage == coverage.max())[0]
            else:
                # Get top-k by coverage
                top_indices = np.argsort(coverage)[-top_k:]
                best_candidates = top_indices
            
            chosen = int(random.choice(best_candidates))
            code.append(chosen)
            
            # Update uncovered
            for u in neighbor_lists[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        return code, uncovered
    
    def prune_redundant(code):
        """Remove redundant codewords from code. Returns pruned code."""
        if not code:
            return code
        
        code = list(code)
        
        # Build cover_count: for each word u, how many codewords cover it
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbor_lists[c]:
                cover_count[u] += 1
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant: all words in B(c) are covered by at least one other codeword
                redundant = True
                for u in neighbor_lists[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                if not redundant:
                    new_code.append(c)
                else:
                    # Remove c from cover_count
                    for u in neighbor_lists[c]:
                        cover_count[u] -= 1
                    changed = True
            code = new_code
        
        return code
    
    def kick_and_repair(code, kick_size=3):
        """Perform kick-and-repair local search.
        Remove kick_size random codewords, then greedily re-add to cover gaps."""
        if len(code) <= kick_size:
            return code
        
        # Make a copy
        code = list(code)
        
        # Remove kick_size random codewords
        remove_indices = random.sample(range(len(code)), kick_size)
        remove_set = set(remove_indices)
        remaining = [code[i] for i in range(len(code)) if i not in remove_set]
        
        # Find uncovered words
        covered = np.zeros(N, dtype=np.uint8)
        for c in remaining:
            for u in neighbor_lists[c]:
                covered[u] = 1
        
        uncovered = set(np.where(covered == 0)[0].tolist())
        
        if not uncovered:
            return remaining
        
        # Greedily add codewords to cover uncovered
        max_adds = kick_size + 5
        for _ in range(max_adds):
            if not uncovered:
                break
            if time.time() > deadline:
                break
            
            uncovered_arr = np.zeros(N, dtype=np.uint8)
            for u in uncovered:
                uncovered_arr[u] = 1
            
            coverage = uncovered_arr[neighbor_matrix].sum(axis=1)
            max_cov = coverage.max()
            if max_cov == 0:
                break
            
            best_candidates = np.where(coverage == max_cov)[0]
            chosen = int(random.choice(best_candidates))
            remaining.append(chosen)
            
            for u in neighbor_lists[chosen]:
                if u in uncovered:
                    uncovered.discard(u)
        
        # Prune the result
        pruned = prune_redundant(remaining)
        return pruned
    
    def swap_improve(code):
        """Try to improve code by swapping one codeword for another.
        For each codeword c in code, try removing it and see if we can add
        a different codeword that covers the same or more uncovered words."""
        code = list(code)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Compute coverage
            covered = np.zeros(N, dtype=np.uint8)
            for c in code:
                for u in neighbor_lists[c]:
                    covered[u] = 1
            
            # For each codeword, find what it uniquely covers
            for i, c in enumerate(code):
                if time.time() > deadline:
                    break
                
                # What does c cover?
                c_neighbors = set(neighbor_lists[c])
                
                # What would be uncovered if we remove c?
                temp_covered = covered.copy()
                for u in neighbor_lists[c]:
                    # Check if u is covered by any other codeword
                    is_covered_by_other = False
                    for j, other in enumerate(code):
                        if j == i:
                            continue
                        if u in neighbor_lists[other]:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        # u would be uncovered
                        temp_covered[u] = 0
                
                uncovered_if_removed = set(np.where(temp_covered == 0)[0].tolist())
                
                if not uncovered_if_removed:
                    # c is redundant
                    code.pop(i)
                    improved = True
                    break
                elif len(uncovered_if_removed) < len(c_neighbors):
                    # Try to find a codeword that covers all of uncovered_if_removed
                    # and doesn't introduce new uncovered words
                    # For simplicity, just try to find a codeword that covers
                    # all of uncovered_if_removed
                    target_arr = np.zeros(N, dtype=np.uint8)
                    for u in uncovered_if_removed:
                        target_arr[u] = 1
                    
                    # Find codewords that cover all of uncovered_if_removed
                    covers_all = neighbor_matrix @ target_arr
                    # A codeword covers all of uncovered_if_removed if
                    # covers_all[candidate] == len(uncovered_if_removed)
                    candidates = np.where(covers_all == len(uncovered_if_removed))[0]
                    
                    if len(candidates) > 0:
                        # Pick one that minimizes the number of new words covered
                        # (to avoid making things worse)
                        best_candidate = None
                        best_new_covered = float('inf')
                        for cand in candidates:
                            cand_set = set(neighbor_lists[cand])
                            new_covered = len(cand_set - set(neighbor_lists[c]))
                            if new_covered < best_new_covered:
                                best_new_covered = new_covered
                                best_candidate = cand
                        
                        if best_candidate is not None and best_candidate not in code:
                            code[i] = best_candidate
                            improved = True
                            break
        
        return code
    
    # Main loop: multiple restarts of greedy + prune + kick-and-repair + swap
    num_restarts = 80
    for iteration in range(num_restarts):
        if time.time() > deadline:
            break
        
        uncovered = set(range(N))
        code = []
        
        # Randomize parameters for diversity
        top_k = random.choice([1, 2, 3, 5, 8])
        noise_scale = random.choice([0, 0.1, 0.5, 1.0])
        
        # Run greedy cover with randomization
        code, uncovered = greedy_cover(uncovered, code, top_k=top_k, noise_scale=noise_scale)
        
        if time.time() > deadline:
            break
        
        # Prune
        code = prune_redundant(code)
        
        # Kick-and-repair iterations
        if time.time() < deadline - 3:
            num_kicks = random.randint(2, 5)
            for _ in range(num_kicks):
                if time.time() > deadline - 2:
                    break
                kick_size = random.randint(2, 6)
                code = kick_and_repair(code, kick_size=kick_size)
        
        # Swap improvement
        if time.time() < deadline - 2:
            code = swap_improve(code)
        
        # Final prune
        code = prune_redundant(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            if best_size <= 186:
                break
    
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback: return a valid covering
        return [list(words[i]) for i in range(N)]
    
    return [list(words[c]) for c in best_code]
