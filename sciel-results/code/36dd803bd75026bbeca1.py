"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Strategy:
      1. Fast greedy set-cover initialization using vectorized numpy operations.
      2. Randomized local search:
         - Remove redundant codewords (those whose removal doesn't uncover any word).
         - Swap codewords: replace a codeword with another that covers more newly-covered words
           while maintaining coverage.
      3. Periodic redundancy removal during local search.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Precompute neighbor sets using numpy for speed
    # For radius=1, each word has 1 + n*(q-1) neighbors
    # We'll use integer representation for words
    def int_to_word(val, n, q):
        word = np.zeros(n, dtype=np.int8)
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Build neighbor lists using numpy vectorization
    # neighbors[w] = list of words within Hamming distance radius of w
    neighbors = [None] * total_words
    
    if radius == 1:
        # Fast path for radius=1
        # Each word w has neighbors: w itself + all words differing in exactly 1 position
        # Total neighbors per word: 1 + n*(q-1)
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = neigh
    else:
        # General case (not expected for this task, but handle gracefully)
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            neighbors[w] = neigh
    
    # Convert to sets for fast membership testing
    neighbors_set = [set(neigh) for neigh in neighbors]
    
    # Greedy set cover initialization
    def greedy_init():
        """Fast greedy: repeatedly pick the word covering most uncovered words."""
        covered = np.zeros(total_words, dtype=np.bool_)
        codewords = []
        uncovered_count = total_words
        uncovered = set(range(total_words))
        
        while uncovered and time.time() < deadline - 5:
            best_word = -1
            best_new_cov = -1
            
            # Sample from uncovered words to find good candidates
            if len(uncovered) > 200:
                sample_uncovered = random.sample(list(uncovered), min(100, len(uncovered)))
            else:
                sample_uncovered = list(uncovered)
            
            # Collect candidate codewords: neighbors of sampled uncovered words
            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors_set[u])
            
            if not candidates:
                break
            
            # Evaluate candidates - count how many uncovered words each covers
            for w in candidates:
                new_cov = 0
                for u in neighbors_set[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break
            
            if best_word == -1:
                # Fallback: pick any uncovered word
                u = next(iter(uncovered))
                best_word = u
            
            codewords.append(best_word)
            for u in neighbors_set[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered.discard(u)
                    uncovered_count -= 1
            
            if uncovered_count == 0:
                break
        
        return codewords
    
    # Local search optimization
    def optimize(codewords, time_limit):
        """Local search: remove redundant codewords, swap codewords to improve coverage."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors_set[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
            """Remove codewords whose removal doesn't uncover any word."""
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors_set[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors_set[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        break
            return changed
        
        # Initial redundancy removal
        remove_redundant()
        
        start_time = time.time()
        last_redundancy_check = start_time
        
        while time.time() - start_time < time_limit:
            # Periodically try to remove redundant codewords
            if time.time() - last_redundancy_check > 0.5:
                remove_redundant()
                last_redundancy_check = time.time()
                continue
            
            if not codewords:
                break
            
            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            # Words uniquely covered by old_cw
            would_uncover = [u for u in neighbors_set[old_cw] if coverage_count[u] == 1]
            
            if not would_uncover:
                # Can remove this codeword
                for u in neighbors_set[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Need to find a replacement that covers all would_uncover words
            # Candidate pool: intersection of neighbors of all would_uncover words
            first_uncovered = would_uncover[0]
            candidates = list(neighbors_set[first_uncovered])
            
            # Filter candidates that cover all would_uncover words
            valid_candidates = []
            for cand in candidates:
                if cand in cw_set:
                    continue
                valid = True
                for u in would_uncover:
                    if u not in neighbors_set[cand]:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(cand)
            
            if valid_candidates:
                # Pick the candidate that covers the most additional words
                best_cand = None
                best_gain = -1
                for cand in valid_candidates:
                    gain = 0
                    for u in neighbors_set[cand]:
                        if coverage_count[u] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None:
                    # Perform the swap
                    for u in neighbors_set[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors_set[best_cand]:
                        coverage_count[u] += 1
                    codewords[idx] = best_cand
                    cw_set.discard(old_cw)
                    cw_set.add(best_cand)
        
        return codewords
    
    # Run greedy initialization
    codewords = greedy_init()
    
    # Run local search with remaining time
    remaining_time = deadline - time.time() - 2
    if remaining_time > 1:
        codewords = optimize(codewords, remaining_time)
    
    # Final redundancy removal
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for u in neighbors_set[cw]:
            coverage_count[u] += 1
    
    i = 0
    while i < len(codewords):
        cw = codewords[i]
        can_remove = True
        for u in neighbors_set[cw]:
            if coverage_count[u] == 1:
                can_remove = False
                break
        if can_remove:
            for u in neighbors_set[cw]:
                coverage_count[u] -= 1
            codewords.pop(i)
        else:
            i += 1
    
    # Convert to list of lists for the return interface
    return [int_to_word(cw, n, q).tolist() for cw in codewords]
# EVOLVE-BLOCK-END
