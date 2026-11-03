"""Combined program: vectorized greedy + numpy local search with redundancy removal."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a fast vectorized greedy construction followed by randomized local search
    (redundancy removal + swaps) with periodic restarts.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors efficiently using numpy
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    q_powers = [q ** i for i in range(n)]
    
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1
    
    # Precompute neighbor sets as Python sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    def greedy_init(seed_offset=0):
        """Fast vectorized greedy: repeatedly pick the word covering most uncovered words."""
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]
            
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])
            
            codewords.append(best_word)
            
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1
        
        return codewords
    
    def local_search(cw, time_limit):
        """Local search: remove redundant codewords, swap codewords to improve coverage."""
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)
        
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        
        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        def try_remove_redundant():
            """Remove codewords whose removal doesn't uncover any word."""
            nonlocal best_cw, best_size
            improved = False
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
            
            return improved
        
        # Initial redundancy removal
        try_remove_redundant()
        
        start_ls = time.time()
        
        while time.time() < time_limit and time.time() < deadline - 2.0:
            # Periodically try to remove redundant codewords
            if random.random() < 0.1:
                try_remove_redundant()
                continue
            
            if not cw:
                break
            
            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(cw))
            old_cw = cw[idx]
            
            # Words uniquely covered by old_cw
            critical_mask = count[neighbors[old_cw]] == 1
            critical = neighbors[old_cw][critical_mask]
            
            if len(critical) == 0:
                # Can simply remove this codeword
                for u in neighbors[old_cw]:
                    count[u] -= 1
                in_code[old_cw] = False
                cw.pop(idx)
                
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
                continue
            
            # Need to find a replacement that covers all critical words
            # Candidate pool: intersection of neighbors of all critical words
            if len(critical) == 1:
                candidate_set = neighbor_sets[critical[0]].copy()
            else:
                candidate_set = neighbor_sets[critical[0]].copy()
                for u in critical[1:]:
                    candidate_set &= neighbor_sets[u]
                    if not candidate_set:
                        break
            
            if not candidate_set:
                continue
            
            # Filter out old_cw and already-in-code candidates
            valid_cands = [c for c in candidate_set if c != old_cw and not in_code[c]]
            
            if not valid_cands:
                continue
            
            # Choose the candidate that maximizes additional coverage
            best_cand = None
            best_gain = 0
            
            # Limit candidate evaluation for speed
            max_candidates = min(100, len(valid_cands))
            sampled_cands = random.sample(valid_cands, max_candidates) if len(valid_cands) > max_candidates else valid_cands
            
            for cand in sampled_cands:
                # Compute gain: number of words that gain redundancy
                gain = 0
                for u in neighbors[cand]:
                    if u not in neighbors[old_cw]:
                        if count[u] >= 1:
                            gain += 1
                    else:
                        # If already covered by old_cw, moving may reduce redundancy
                        if count[u] == 2:
                            gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand is not None:
                for u in neighbors[old_cw]:
                    count[u] -= 1
                for u in neighbors[best_cand]:
                    count[u] += 1
                cw[idx] = best_cand
                in_code[old_cw] = False
                in_code[best_cand] = True
        
        # Final redundancy removal
        try_remove_redundant()
        
        return best_cw
    
    # Multiple restarts with adaptive time allocation
    best_codewords = []
    best_len = float('inf')
    restart_count = 0
    
    while time.time() < deadline - 3.0:
        restart_count += 1
        
        # Greedy is fast, give it a small budget
        remaining_for_opt = deadline - 3.0 - time.time()
        if remaining_for_opt <= 5:
            break
        
        opt_time = min(remaining_for_opt - 1, 25)  # Cap per-iteration optimization
        if opt_time <= 0:
            break
        
        cw = greedy_init(seed_offset=restart_count)
        
        # Quick redundancy pass before full optimization
        count = np.zeros(total_words, dtype=np.int32)
        for ci in cw:
            for u in neighbors[ci]:
                count[u] += 1
        
        # Remove obviously redundant codewords first
        indices = list(range(len(cw)))
        random.shuffle(indices)
        for i in indices:
            ci = cw[i]
            if np.min(count[neighbors[ci]]) > 1:
                for u in neighbors[ci]:
                    count[u] -= 1
                cw.pop(i)
        
        cw = local_search(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
            # Early termination if we hit the known optimum
            if best_len <= 486:
                break
    
    # Final validation: ensure all words are covered
    if not best_codewords:
        # Fallback: use greedy if something went wrong
        best_codewords = greedy_init()
    
    # Verify coverage
    covered = np.zeros(total_words, dtype=bool)
    for cw in best_codewords:
        for u in neighbors[cw]:
            covered[u] = True
    
    if not np.all(covered):
        # Something went wrong, fall back to simple greedy
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered = set(range(total_words))
        while uncovered:
            u = next(iter(uncovered))
            # Pick the neighbor of u that covers most uncovered words
            best_w = -1
            best_cov = -1
            for w in neighbors[u]:
                cov = sum(1 for x in neighbors[w] if x in uncovered)
                if cov > best_cov:
                    best_cov = cov
                    best_w = w
            codewords.append(best_w)
            for x in neighbors[best_w]:
                if x in uncovered:
                    uncovered.discard(x)
        best_codewords = codewords
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
