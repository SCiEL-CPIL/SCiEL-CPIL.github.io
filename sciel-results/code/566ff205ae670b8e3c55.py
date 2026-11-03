"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct(q, n, radius):
    if n == 0:
        return [[]]

    total_words = q ** n
    if radius < 0:
        return []

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

    # Precompute neighbor lists for all words
    # For q=3, n=8, radius=1: 6561 * (1 + 8*2) = 104976 entries, very manageable
    neighbors = []
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            # For radius 2, we need all words at distance <= 2
            # This could be large, but for q=3, n=8, radius=1 is the main case
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
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors.append(neigh)

    # Build reverse neighbors: for each word, which codewords cover it
    # This is useful for finding candidates
    reverse_neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        for u in neighbors[w]:
            reverse_neighbors[u].append(w)

    def greedy_init():
        """Fast greedy construction of a feasible covering."""
        covered = np.zeros(total_words, dtype=np.uint8)
        codewords = []
        uncovered_count = total_words
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Sample a subset of uncovered words to find a good codeword
            # For efficiency, we don't check all words every time
            sample_size = min(total_words, max(100, uncovered_count // 2))
            sample = random.sample(range(total_words), sample_size)
            
            for w in sample:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                # Fallback: check all words
                for w in range(total_words):
                    new_cov = 0
                    for u in neighbors[w]:
                        if not covered[u]:
                            new_cov += 1
                    if new_cov > best_new_coverage:
                        best_new_coverage = new_cov
                        best_word = w
                if best_word == -1:
                    break
            
            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered_count -= 1
        
        return codewords

    def optimize(codewords, time_limit):
        """Optimize the covering by removing redundant codewords and performing swaps."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            """Remove codewords that don't uniquely cover any word."""
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
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            # Try to swap a codeword with a better one
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered if we remove old_cw
            would_uncover = []
            for u in neighbors[old_cw]:
                if coverage_count[u] == 1:
                    would_uncover.append(u)
            
            if len(would_uncover) == 0:
                # Can remove without swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Try to find a replacement that covers all would_uncover words
            # Candidates: words that cover at least one of the would_uncover words
            candidate_set = set()
            for u in would_uncover:
                candidate_set.update(reverse_neighbors[u])
            
            candidates = list(candidate_set)
            candidates = [c for c in candidates if c not in cw_set and c != old_cw]
            random.shuffle(candidates)
            
            best_new_cw = -1
            best_score = -1
            
            for new_cw in candidates[:50]:  # Limit candidates for speed
                # Check if new_cw covers all would_uncover
                valid = True
                for u in would_uncover:
                    if u not in neighbors[new_cw]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Score: count how many words in neighbors[new_cw] are already covered
                # (i.e., have coverage_count >= 1), as these represent redundancy
                score = 0
                for u in neighbors[new_cw]:
                    if coverage_count[u] >= 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords

    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
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
