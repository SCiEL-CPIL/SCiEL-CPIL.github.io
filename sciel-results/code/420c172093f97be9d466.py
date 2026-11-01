import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback for non-standard cases
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        powers = [q ** i for i in range(n)]
        neighbors_list = []
        for w in range(total_words):
            nbs = [w]
            for k in range(n):
                d_k = (w // powers[k]) % q
                for v in range(q):
                    if v != d_k:
                        w_new = w + (v - d_k) * powers[k]
                        nbs.append(w_new)
            neighbors_list.append(nbs)
        neighbors = np.array(neighbors_list, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        while uncovered_count > 0:
            scores = np.sum(~covered[neighbors], axis=1)
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[0]
            code.append(int(best_idx))
            nbs = neighbors[best_idx]
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
        result = []
        for w in code:
            digits = [0] * n
            x = w
            for i in range(n):
                digits[i] = x % q
                x //= q
            result.append(digits)
        return result
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
        
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    def greedy_with_tiebreaking(run_seed):
        """Run greedy with specific tie-breaking strategy."""
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        rng = np.random.RandomState(run_seed)
        
        while uncovered_count > 0:
            scores = np.sum(~covered[neighbors], axis=1)
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # Weighted random choice among top candidates
                # Prefer candidates that cover more unique uncovered words
                # Use a simple heuristic: pick randomly from top candidates
                best_idx = rng.choice(candidates)
                
            code.append(int(best_idx))
            nbs = neighbors[best_idx]
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
            
        return code
    
    def local_search(code, neighbors, total_words, time_limit):
        """Local search with removal and swap operations to reduce code size."""
        code = list(code)
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in code:
            coverage[neighbors[cw]] += 1
        
        # Phase 1: Try to remove codewords if possible (using coverage counts)
        improved = True
        while improved:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() > time_limit:
                    return code
                
                # Check if removing code[i] still covers all words
                nbs = neighbors[code[i]]
                # A word can only be uncovered if its coverage count is exactly 1
                if np.any(coverage[nbs] == 1):
                    continue
                
                # Safe to remove
                coverage[nbs] -= 1
                code.pop(i)
                improved = True
                break
        
        # Phase 2: Try swap operations - replace one codeword with another
        # that covers more uncovered words (or allows further removals)
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                return code
            
            # Try each codeword for replacement
            for i in range(len(code)):
                if time.time() > time_limit:
                    return code
                
                # What would be uncovered if we remove code[i]?
                old_nbs = neighbors[code[i]]
                # Temporarily remove
                coverage[old_nbs] -= 1
                uncovered_after_remove = np.sum(coverage == 0)
                
                if uncovered_after_remove == 0:
                    # Already removable, shouldn't happen after Phase 1
                    coverage[old_nbs] += 1
                    continue
                
                # Try to find a replacement codeword that covers all currently uncovered
                # words plus some already-covered ones
                # Score: number of uncovered words covered by candidate
                candidate_scores = np.sum((coverage[neighbors] == 0), axis=1)
                
                # Find best candidate that covers all uncovered words
                best_candidate = -1
                best_score = 0
                # We need candidates that cover ALL uncovered words
                # First, identify which words are uncovered
                uncovered_words = np.where(coverage == 0)[0]
                
                if len(uncovered_words) == 0:
                    coverage[old_nbs] += 1
                    continue
                
                # For each candidate, check if it covers all uncovered words
                # This is expensive, so use a heuristic: pick candidate with max coverage of uncovered
                # But we need it to cover ALL uncovered, so let's be smarter
                
                # Actually, a swap only helps if we can remove one codeword and add one
                # that covers at least as much. Since we're trying to REDUCE size,
                # swaps don't directly help unless they enable further removals.
                
                # Better approach: try to find if any single codeword can replace
                # code[i] AND allow removing another codeword.
                # This is complex, so let's try a simpler heuristic:
                # For each pair (i, j), check if removing both and adding one
                # codeword that covers the union still works.
                
                coverage[old_nbs] += 1
        
        # Phase 3: Pairwise swap - try replacing two codewords with one
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                return code
            
            for i in range(len(code)):
                if time.time() > time_limit:
                    return code
                for j in range(i + 1, len(code)):
                    if time.time() > time_limit:
                        return code
                    
                    # Try removing both code[i] and code[j]
                    nbs_i = neighbors[code[i]]
                    nbs_j = neighbors[code[j]]
                    
                    coverage[nbs_i] -= 1
                    coverage[nbs_j] -= 1
                    
                    uncovered = np.sum(coverage == 0)
                    
                    if uncovered == 0:
                        # Both can be removed!
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        break
                    
                    # Check if any single codeword can cover all currently uncovered words
                    # plus maintain coverage of others
                    uncovered_words = np.where(coverage == 0)[0]
                    
                    if len(uncovered_words) > 0:
                        # Find candidate that covers all uncovered_words
                        # and doesn't cause issues
                        # Score: covers all uncovered + minimizes "waste"
                        candidate_scores = np.sum((coverage[neighbors] == 0), axis=1)
                        
                        # We need candidates that cover ALL uncovered words
                        # Check which candidates cover all uncovered
                        # For each uncovered word, which candidates cover it?
                        # This is expensive, so use a filter:
                        # A candidate must have score == len(uncovered_words) to cover all
                        
                        candidates = np.where(candidate_scores == len(uncovered_words))[0]
                        
                        if len(candidates) > 0:
                            # Pick the one that maximizes total coverage (to enable future removals)
                            total_cov = np.sum(coverage[neighbors[candidates]] > 0, axis=1)
                            best_idx = candidates[np.argmax(total_cov)]
                            
                            # Replace: remove i and j, add best_idx
                            if best_idx not in code:
                                code.pop(j)
                                code.pop(i)
                                code.append(int(best_idx))
                                coverage[neighbors[best_idx]] += 1
                                improved = True
                                break
                            else:
                                # best_idx is already in code, just remove i and j
                                # But wait, we already decremented coverage for i and j
                                # best_idx's coverage is still there, so we're fine
                                code.pop(j)
                                code.pop(i)
                                improved = True
                                break
                    
                    # Restore coverage
                    coverage[nbs_i] += 1
                    coverage[nbs_j] += 1
                
                if improved:
                    break
        
        return code
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple rounds of greedy with different tie-breaking
    num_runs = 8
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        if run == 0:
            # Deterministic (lowest index)
            covered = np.zeros(total_words, dtype=bool)
            code = []
            uncovered_count = total_words
            while uncovered_count > 0:
                scores = np.sum(~covered[neighbors], axis=1)
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[0]
                code.append(int(best_idx))
                nbs = neighbors[best_idx]
                newly = np.sum(~covered[nbs])
                covered[nbs] = True
                uncovered_count -= newly
        else:
            # Random tie-breaking
            code = greedy_with_tiebreaking(run * 12345)
        
        # Apply local search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            code = local_search(code, neighbors, total_words, deadline - 5)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
    
    # Final polish: run local search again on the best code found
    if best_code is not None and time.time() < deadline - 3:
        polished = local_search(best_code, neighbors, total_words, deadline - 1)
        if len(polished) < best_size:
            best_code = polished
            best_size = len(polished)
    
    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert best_code (list of ints) to list of lists of digits
    result = []
    for w in best_code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
        
    return result
