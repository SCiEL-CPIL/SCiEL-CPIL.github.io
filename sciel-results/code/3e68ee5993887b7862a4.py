"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a fast greedy construction with neighborhood-based candidate sampling,
    followed by randomized local search that swaps codewords to maximize redundant
    coverage, with periodic redundancy removal.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
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
    
    # Precompute neighbors efficiently
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    # Total: 6561 * 17 = ~111537 entries, very manageable
    neighbors = [None] * total_words
    
    # Precompute word arrays for fast neighbor computation
    if radius == 1:
        # Fast path for radius=1
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
            neighbors[w] = set(neigh)
    elif radius == 2:
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
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)
    else:
        # General case
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
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)
    
    # Also build reverse neighbors: for each word, which codewords cover it
    # This is just the same as neighbors but from the codeword perspective
    # neighbors[cw] = set of words covered by cw
    
    def greedy_init():
        """Fast greedy: repeatedly pick the word covering most uncovered words.
        Uses neighborhood-based candidate sampling for speed."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # For efficiency, track uncovered words in a set
        uncovered = set(range(total_words))
        
        while uncovered:
            best_word = -1
            best_new_cov = -1
            
            # Sample from uncovered words to find a good candidate
            # The candidate should be in the neighborhood of uncovered words
            # Pick a random uncovered word and look at its neighbors as candidates
            if len(uncovered) > 500:
                # Sample some uncovered words
                sample_uncovered = random.sample(list(uncovered), min(100, len(uncovered)))
            else:
                sample_uncovered = list(uncovered)
            
            # Collect candidate codewords: neighbors of sampled uncovered words
            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])
            
            if not candidates:
                break
            
            # Evaluate candidates
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if u in uncovered:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break
            
            if best_word == -1:
                # Fallback: pick any uncovered word's neighbor
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))
            
            codewords.append(best_word)
            for u in neighbors[best_word]:
                if u in uncovered:
                    uncovered.discard(u)
            
            uncovered_count -= best_new_cov if best_new_cov > 0 else 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Local search: remove redundant codewords, swap codewords to improve coverage."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
            """Remove codewords whose removal doesn't uncover any word."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        # Initial redundancy removal
        remove_redundant()
        
        start_time = time.time()
        last_redundancy_check = start_time
        
        while time.time() - start_time < time_limit:
            # Periodically try to remove redundant codewords
            if time.time() - last_redundancy_check > 0.3:
                remove_redundant()
                last_redundancy_check = time.time()
                continue
            
            if not codewords:
                break
            
            # Pick a codeword to try swapping/removing
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            # Words uniquely covered by old_cw
            would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
            
            if not would_uncover:
                # Can remove this codeword
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Need to find a replacement that covers all would_uncover words
            # Candidate pool: intersection of neighbors of all would_uncover words
            # For efficiency, start with neighbors of the first and filter
            first_uncovered = would_uncover[0]
            candidates = list(neighbors[first_uncovered])
            random.shuffle(candidates)
            
            best_new_cw = -1
            best_score = -1
            
            # Limit candidates to speed up
            max_candidates = 50
            for new_cw in candidates[:max_candidates]:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                
                # Check if new_cw covers all would_uncover
                valid = True
                for u in would_uncover:
                    if u not in neighbors[new_cw]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check that removing old_cw doesn't create new uncovered words
                # Words in N(old) but not N(new) with coverage_count == 1 would become uncovered
                bad_count = 0
                for u in neighbors[old_cw]:
                    if u not in neighbors[new_cw]:
                        if coverage_count[u] == 1:
                            bad_count += 1
                            if bad_count > 0:
                                break
                
                if bad_count > 0:
                    continue
                
                # Score: how many new redundancies we create
                # Words in N(new) but not N(old) that are already covered
                score = 0
                for u in neighbors[new_cw]:
                    if u not in neighbors[old_cw]:
                        if coverage_count[u] > 0:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                # Perform swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
            else:
                # No good swap found. Try to add a random codeword and then remove a redundant one?
                # This increases size temporarily but may escape local minima.
                # Try with some probability to avoid too much overhead
                if random.random() < 0.1 and len(codewords) < 600:
                    # Pick a random word as potential new codeword
                    new_cw = random.randrange(total_words)
                    if new_cw not in cw_set:
                        # Add it temporarily
                        for u in neighbors[new_cw]:
                            coverage_count[u] += 1
                        codewords.append(new_cw)
                        cw_set.add(new_cw)
                        
                        # Try to remove a redundant codeword
                        removed = remove_redundant()
                        if not removed:
                            # Revert
                            for u in neighbors[new_cw]:
                                coverage_count[u] -= 1
                            codewords.pop()
                            cw_set.discard(new_cw)
        
        remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        iter_time = min(15, time_limit - (time.time() - start_time))
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 2)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
