"""Improved program using randomized local search with swap-based optimization."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses fast greedy initialization followed by iterative swap-based local search.
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
    
    # Precompute all neighbors efficiently
    neighbors = [None] * total_words
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = {w}
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = new_val
                    neigh.add(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1, orig2 = word[pos1], word[pos2]
                    for v1 in range(q):
                        if v1 == orig1:
                            continue
                        for v2 in range(q):
                            if v2 == orig2:
                                continue
                            new_word = word[:]
                            new_word[pos1] = v1
                            new_word[pos2] = v2
                            neigh.add(word_to_int(new_word, q))
        neighbors[w] = neigh
    
    def greedy_init():
        """Fast greedy: repeatedly pick the word covering the most uncovered words."""
        covered = bytearray(total_words)
        uncovered_count = total_words
        codewords = []
        
        while uncovered_count > 0:
            best_word = -1
            best_new = 0
            # Sample a subset of candidates to speed up greedy
            sample_size = min(total_words, 2000)
            candidates = random.sample(range(total_words), sample_size)
            
            for w in candidates:
                new_cov = sum(1 for u in neighbors[w] if not covered[u])
                if new_cov > best_new:
                    best_new = new_cov
                    best_word = w
            
            if best_word == -1:
                # Fallback: try all
                for w in range(total_words):
                    new_cov = sum(1 for u in neighbors[w] if not covered[u])
                    if new_cov > best_new:
                        best_new = new_cov
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
        """
        Iteratively swap codewords to improve coverage efficiency.
        Score = number of words covered exactly once (minimize this).
        """
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        unique_covered = sum(1 for c in coverage_count if c == 1)
        
        def remove_redundant():
            """Remove any codeword that doesn't uniquely cover any word."""
            removed = 0
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
                    cw = codewords[i]
                    has_unique = any(coverage_count[u] == 1 for u in neighbors[cw])
                    if not has_unique:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed += 1
                        break
            return removed
        
        remove_redundant()
        unique_covered = sum(1 for c in coverage_count if c == 1)
        
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() - start_time < time_limit:
            # Try to remove redundant codewords
            if remove_redundant() > 0:
                no_improve_count = 0
                unique_covered = sum(1 for c in coverage_count if c == 1)
                continue
            
            if len(codewords) == 0:
                break
            
            # Pick a random codeword to swap
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered if we remove old_cw
            would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
            
            if not would_uncover:
                # Safe to remove
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                unique_covered = sum(1 for c in coverage_count if c == 1)
                no_improve_count = 0
                continue
            
            # Find a replacement that covers all would_uncover
            # Candidates: neighbors of each word in would_uncover
            candidate_set = set()
            for u in would_uncover:
                candidate_set.update(neighbors[u])
            candidate_set.discard(old_cw)
            candidate_set -= cw_set
            
            if not candidate_set:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
                continue
            
            candidates = list(candidate_set)
            random.shuffle(candidates)
            
            best_new_cw = -1
            best_gain = 0
            
            for new_cw in candidates:
                # Check if new_cw covers all would_uncover
                valid = all(u in neighbors[new_cw] for u in would_uncover)
                if not valid:
                    continue
                
                # Calculate gain: how many words get additional coverage
                gain = 0
                for u in neighbors[new_cw]:
                    if coverage_count[u] > 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw
                
                if best_gain >= len(would_uncover):
                    break  # Can't do better than this
            
            if best_new_cw != -1:
                # Perform the swap
                for u in neighbors[old_cw]:
                    coverage_count[u] -= 1
                for u in neighbors[best_new_cw]:
                    coverage_count[u] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 500:
                    break
        
        remove_redundant()
        return codewords
    
    # Multiple restarts with time management
    start_time = time.time()
    time_limit = 115
    best_codewords = None
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        if remaining_time < 10:
            break
        
        # Allocate time for this iteration
        iter_time = min(30, remaining_time - 5)
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 2)
        
        if best_codewords is None or len(cw) < len(best_codewords):
            best_codewords = cw
        
        # If we've found a good solution, use remaining time for more optimization
        if best_codewords and time.time() - start_time > time_limit - 15:
            final_time = time_limit - (time.time() - start_time) - 2
            if final_time > 5:
                best_codewords = optimize(best_codewords, final_time)
            break
    
    if best_codewords is None:
        best_codewords = greedy_init()
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
