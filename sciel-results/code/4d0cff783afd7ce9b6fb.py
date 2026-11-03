"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with greedy init, redundancy removal, and swap optimization.
    Optimized for speed: precomputed neighbor sets, strict time guards.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    start_global = time.time()
    deadline = start_global + 110.0  # Safety margin for 120s limit
    
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

    # Precompute neighbors as sets for O(1) membership and iteration
    # This is critical for speed in the optimization phase.
    neighbors = [None] * total_words
    
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = {w}
        
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    word[pos] = val
                    neigh.add(word_to_int(word, q))
                word[pos] = orig_val
        
        if radius >= 2:
            # Radius 2 logic
            for pos1 in range(n):
                orig1 = word[pos1]
                for val1 in range(q):
                    if val1 == orig1:
                        continue
                    word[pos1] = val1
                    for pos2 in range(pos1 + 1, n):
                        orig2 = word[pos2]
                        for val2 in range(q):
                            if val2 == orig2:
                                continue
                            word[pos2] = val2
                            neigh.add(word_to_int(word, q))
                        word[pos2] = orig2
                word[pos1] = orig1
        
        neighbors[w] = neigh
        
    def greedy_init(time_limit):
        """
        Fast greedy: 
        1. Maintain a set of uncovered words.
        2. Sample uncovered words to find candidates.
        3. Pick candidate covering most uncovered words.
        """
        covered = bytearray(total_words)
        codewords = []
        uncovered = set(range(total_words))
        
        while uncovered and time.time() < time_limit:
            # Sample uncovered words to generate candidates
            if len(uncovered) < 100:
                sample_uncovered = list(uncovered)
            else:
                sample_size = min(50, len(uncovered))
                sample_uncovered = random.sample(list(uncovered), sample_size)
            
            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])
            
            best_word = -1
            best_gain = -1
            
            for w in candidates:
                # Calculate gain: how many currently uncovered words does w cover?
                gain = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        gain += 1
                        if gain >= best_gain: # Optimization: break early if we can't beat current best
                            # Actually, we need exact gain to compare? 
                            # If gain > best_gain, we update. If we break early, we might miss a better one?
                            # No, if we break because gain >= best_gain, we don't know if it's strictly greater.
                            # But for greedy, any improvement is good. Let's just count fully for simplicity in this small space.
                            pass
                
                # Corrected gain calculation
                gain = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    if best_gain == len(uncovered):
                        break
            
            if best_word == -1:
                best_word = next(iter(uncovered))
            
            codewords.append(best_word)
            
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = 1
                    uncovered.discard(u)
        
        return codewords
    
    def optimize(codewords, time_limit):
        """
        Local search:
        1. Remove redundant codewords.
        2. Try swaps to reduce size or maintain size with better structure.
        """
        if not codewords:
            return codewords
            
        coverage = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage[u] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
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
                        if coverage[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        remove_redundant()
        
        start_time = time.time()
        
        while time.time() < time_limit:
            # 1. Try to remove redundant
            if remove_redundant():
                continue
            
            if not codewords:
                break
            
            # 2. Try Swap
            idx = random.randrange(len(codewords))
            old_cw = codewords[idx]
            
            # Words uniquely covered by old_cw
            would_uncover = [u for u in neighbors[old_cw] if coverage[u] == 1]
            
            if not would_uncover:
                # Safe to remove
                for u in neighbors[old_cw]:
                    coverage[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find best replacement
            # Candidates must cover all words in would_uncover
            # Intersection of neighbors of all words in would_uncover
            # Start with neighbors of first word
            candidates = set(neighbors[would_uncover[0]])
            
            # Intersect with neighbors of other words in would_uncover
            for u in would_uncover[1:]:
                candidates &= neighbors[u]
                if not candidates:
                    break
                # Early exit if too many? No, intersection usually shrinks fast.
            
            candidates.discard(old_cw)
            
            if not candidates:
                continue
                
            # Pick a random candidate from the valid set
            # We could score them, but random is fast and often good enough for local search
            new_cw = random.choice(list(candidates))
            
            # Apply swap
            for u in neighbors[old_cw]:
                coverage[u] -= 1
            for u in neighbors[new_cw]:
                coverage[u] += 1
            codewords[idx] = new_cw
            cw_set.discard(old_cw)
            cw_set.add(new_cw)
        
        remove_redundant()
        return codewords

    # Multiple restarts
    best = []
    
    while time.time() < deadline - 2:
        # Budget for this iteration
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Allocate time: 2s for greedy, rest for optimize
        greedy_time = min(2.0, remaining * 0.2)
        opt_time = remaining - greedy_time - 0.5
        
        cw = greedy_init(time.time() + greedy_time)
        cw = optimize(cw, time.time() + opt_time)
        
        if not best or len(cw) < len(best):
            best = cw[:]
            
        # If we are close to deadline, stop early to allow return
        if time.time() > deadline - 5:
            break

    if not best:
        # Fallback: return a trivial valid code if something went wrong (should not happen)
        # Just return all words? No, that's too big. 
        # Return empty? No, invalid.
        # This case should be rare.
        return [[]]

    return [int_to_word(cw, n, q) for cw in best]
# EVOLVE-BLOCK-END
