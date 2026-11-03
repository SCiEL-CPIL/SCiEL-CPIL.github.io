"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses bitmask-based coverage tracking with randomized local search and perturbation.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    if total_words == 0:
        return [[]]
    
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute neighbors using bitmask representation
    # Each word is an integer 0..total_words-1
    # neighbors[w] is a list of word indices within Hamming distance radius
    
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
    
    # Build neighbor lists
    neighbors = []
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh_set = {w}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh_set.add(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            neigh_set.add(word_to_int(new_word, q))
        neighbors.append(sorted(neigh_set))
    
    # Build neighbor bitmask for O(1) membership
    # For large spaces, use a set instead
    use_set = total_words > 100000
    if use_set:
        neighbor_sets = [set(nb) for nb in neighbors]
    else:
        neighbor_sets = None
    
    def in_neighbors(w, v):
        if use_set:
            return v in neighbor_sets[w]
        # Fallback: linear search (only for small spaces)
        return v in neighbors[w]
    
    def greedy_init():
        """Greedy set cover initialization."""
        covered = [0] * total_words
        codewords = []
        codewords_set = set()
        uncovered_count = total_words
        
        while uncovered_count > 0:
            # Find word that covers most uncovered words
            best_w = -1
            best_new_cov = -1
            
            # Sample a subset if space is large
            if total_words > 5000:
                candidates = random.sample(range(total_words), min(200, total_words))
            else:
                candidates = range(total_words)
            
            for w in candidates:
                if w in codewords_set:
                    continue
                new_cov = 0
                for u in neighbors[w]:
                    if covered[u] == 0:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_w = w
            
            if best_w == -1 or best_new_cov <= 0:
                # Fallback: pick random uncovered word
                for w in range(total_words):
                    if covered[w] == 0:
                        best_w = w
                        break
            
            if best_w == -1:
                break
            
            codewords.append(best_w)
            codewords_set.add(best_w)
            for u in neighbors[best_w]:
                if covered[u] == 0:
                    covered[u] = 1
                    uncovered_count -= 1
        
        return codewords, codewords_set
    
    def optimize(codewords, codewords_set, time_limit):
        """Local search optimization with swaps and removals."""
        # Build coverage counts
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        start_time = time.time()
        
        def try_remove_redundant():
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
                        codewords_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        # Initial redundant removal
        try_remove_redundant()
        
        # Local search loop
        while time.time() - start_time < time_limit:
            if time.time() > deadline - 5:
                break
            
            # Try to remove redundant codewords
            if try_remove_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            # Try swap: pick a codeword, try to replace with better one
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would be uncovered if we remove old_cw
            would_uncover = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                # Safe to remove
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                codewords_set.discard(old_cw)
                continue
            
            # Find best replacement from neighbors of first uncovered word
            first_uncovered = would_uncover[0]
            potential_candidates = neighbors[first_uncovered]
            random.shuffle(potential_candidates)
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in potential_candidates[:50]:  # Limit candidates for speed
                if new_cw in codewords_set or new_cw == old_cw:
                    continue
                
                # Check if new_cw covers all would_uncover
                valid = True
                for u in would_uncover:
                    if not in_neighbors(new_cw, u):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Score: count how many words new_cw covers that old_cw doesn't
                score = 0
                for u in neighbors[new_cw]:
                    if not in_neighbors(old_cw, u):
                        if coverage_count[u] >= 1:
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
                codewords_set.discard(old_cw)
                codewords_set.add(best_new_cw)
            
            # Perturbation: occasionally make a random move to escape local minima
            if random.random() < 0.05 and len(codewords) > 0:
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                
                # Pick a random new codeword
                new_cw = random.randint(0, total_words - 1)
                if new_cw not in codewords_set and new_cw != old_cw:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    codewords_set.discard(old_cw)
                    codewords_set.add(new_cw)
        
        # Final redundant removal
        try_remove_redundant()
        return codewords, codewords_set, coverage_count
    
    # Multiple restarts
    start_time = time.time()
    best_codewords = []
    best_len = float('inf')
    
    while time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
        
        # Dynamic time allocation
        if random.random() < 0.4:
            opt_time = min(8, remaining_time - 2)
        else:
            opt_time = min(2, remaining_time - 2)
        
        if opt_time <= 0:
            continue
        
        cw, cw_set = greedy_init()
        cw, cw_set, _ = optimize(cw, cw_set, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
    
    # If we haven't found anything, do one final greedy
    if not best_codewords:
        cw, cw_set = greedy_init()
        best_codewords = cw
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
