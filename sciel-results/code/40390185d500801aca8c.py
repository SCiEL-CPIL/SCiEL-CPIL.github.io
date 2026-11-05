import itertools
import time
import random
import numpy as np
import sys

_START_TIME = None
_DEADLINE_FRAC = 0.90  # Stop at 90% of 120s to be safe


def _elapsed():
    return time.time() - _START_TIME


def _within_deadline(margin=0.0):
    if _START_TIME is None:
        return True
    limit = 120.0 * _DEADLINE_FRAC
    return (time.time() - _START_TIME) < (limit - margin)


def _flat(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _unflat(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _precompute_balls(q, n, radius):
    """
    Precompute the set of indices covered by each word in {0..q-1}^n.
    Returns a list of sets, where balls[i] is the set of indices covered by word i.
    """
    total = q ** n
    balls = [None] * total
    
    # Generate all words
    all_words = [tuple(_unflat(i, q, n)) for i in range(total)]
    
    for i, w in enumerate(all_words):
        covered = set()
        # Center
        covered.add(i)
        # Radius 1 neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(_flat(new_w, q, n))
        balls[i] = covered
        
    return balls


def _greedy_cover(balls, total, q, n):
    """
    Greedy set cover algorithm.
    Returns a list of indices of codewords.
    """
    uncovered = set(range(total))
    codewords = []
    
    # Precompute the size of each ball for initial sorting or just use len()
    # To speed up, we can maintain a count of uncovered elements per candidate
    # But since N=729 is small, we can just iterate.
    
    while uncovered and _within_deadline():
        best_i = -1
        best_count = 0
        
        # Optimization: Only consider candidates that cover at least one uncovered word
        # We can iterate through all candidates
        for i in range(total):
            if i in codewords:
                continue
            # Count how many uncovered words this ball covers
            # Using set intersection is O(|uncovered|) or O(|ball|)
            count = len(balls[i] & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
            if best_count == len(uncovered):
                break
                
        if best_i < 0 or best_count == 0:
            break
            
        codewords.append(best_i)
        uncovered -= balls[best_i]
        
    return codewords


def _local_search_reduce(codewords, balls, total, q, n, max_iters=1000):
    """
    Local search to reduce the number of codewords.
    1. Try to remove redundant codewords.
    2. Try to swap codewords to enable removals or directly reduce count (if swap+remove is possible).
    """
    codewords = list(codewords)
    if not codewords:
        return codewords
    
    # Coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        for idx in balls[cw]:
            coverage[idx] += 1
            
    # Verify validity
    if not (coverage >= 1).all():
        return codewords # Invalid, should not happen if greedy was correct

    improved = True
    iters = 0
    rng = random.Random(42)
    
    while improved and iters < max_iters and _within_deadline():
        improved = False
        iters += 1
        
        # Phase 1: Removal
        # Check if any codeword is redundant
        for i in range(len(codewords) - 1, -1, -1):
            if not _within_deadline():
                break
            cw = codewords[i]
            # If all words in balls[cw] have coverage > 1, we can remove cw
            # i.e., min(coverage[balls[cw]]) > 1
            # Efficient check:
            is_redundant = True
            for idx in balls[cw]:
                if coverage[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for idx in balls[cw]:
                    coverage[idx] -= 1
                del codewords[i]
                improved = True
                break # Restart loop after change
        
        # Phase 2: Swap
        # If no removal found, try to swap one codeword with another to potentially allow a removal
        # Strategy: Pick a codeword 'victim' that has low "slack" (covers many words with coverage 1)
        # Try to replace it with a candidate that covers those critical words.
        
        if not improved and len(codewords) > 0:
            # Find the most "critical" codeword (one that if removed, breaks many words)
            # Actually, we want to find a codeword that is hard to remove.
            # Let's pick a random codeword to swap out.
            # To be smarter, pick one that covers the most words with coverage == 1.
            
            best_victim = -1
            max_critical = -1
            
            for i, cw in enumerate(codewords):
                critical_count = 0
                for idx in balls[cw]:
                    if coverage[idx] == 1:
                        critical_count += 1
                if critical_count > max_critical:
                    max_critical = critical_count
                    best_victim = i
            
            if best_victim == -1:
                break # No critical words? Then we could have removed something? 
                       # If max_critical is 0, all words in balls[cw] have coverage > 1, 
                       # so Phase 1 should have caught it. 
                       # If max_critical > 0, we have a victim.
            
            victim_cw = codewords[best_victim]
            
            # Find a candidate that covers all the critical words of victim
            # Critical words = { idx in balls[victim_cw] | coverage[idx] == 1 }
            critical_words = set()
            for idx in balls[victim_cw]:
                if coverage[idx] == 1:
                    critical_words.add(idx)
            
            if not critical_words:
                # Should have been removed in Phase 1
                continue
                
            # Look for a candidate 'cand' not in codewords such that critical_words is subset of balls[cand]
            # To speed up, we can check candidates that intersect critical_words
            
            # Heuristic: Check candidates that cover the first critical word
            first_crit = next(iter(critical_words))
            
            # Precompute reverse index: which words cover index X?
            # This is expensive to build dynamically, but we can just iterate all candidates
            # N=729, so 729 checks is fine.
            
            found_swap = False
            # Randomize candidate order to escape local minima
            candidates = [i for i in range(total) if i not in codewords]
            rng.shuffle(candidates)
            
            for cand in candidates:
                if not _within_deadline():
                    break
                    
                # Check if cand covers all critical words
                if critical_words.issubset(balls[cand]):
                    # Valid swap candidate
                    # Perform swap
                    # 1. Remove victim coverage
                    for idx in balls[victim_cw]:
                        coverage[idx] -= 1
                    # 2. Add cand coverage
                    for idx in balls[cand]:
                        coverage[idx] += 1
                    # 3. Update codewords list
                    codewords[best_victim] = cand
                    
                    # Check if this swap allows a removal of another codeword
                    # Or just mark as improved and let Phase 1 handle it in next iteration
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # No swap found for this victim. 
                # To avoid getting stuck, we might need to try a different victim or accept we are stuck.
                # For now, break if we can't find any swap for the most critical victim.
                # A better strategy might be to try all victims.
                pass

    return codewords


def _construct_k3_6_1(q, n, radius):
    """
    Specialized construction for q=3, n=6, radius=1.
    """
    if q != 3 or n != 6 or radius != 1:
        return None

    total = 3 ** 6
    balls = _precompute_balls(q, n, radius)
    
    best_codewords = None
    best_len = total + 1
    
    # Try multiple random restarts of greedy + local search
    for seed in range(20):
        if not _within_deadline(margin=5.0):
            break
            
        rng = random.Random(seed)
        
        # Greedy with random tie-breaking
        uncovered = set(range(total))
        codewords = []
        
        while uncovered and _within_deadline():
            best_i = -1
            best_count = 0
            best_candidates = []
            
            for i in range(total):
                if i in codewords:
                    continue
                count = len(balls[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_candidates = [i]
                elif count == best_count and count > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            # Pick randomly from best candidates
            best_i = rng.choice(best_candidates)
            codewords.append(best_i)
            uncovered -= balls[best_i]
            
        # Local search
        codewords = _local_search_reduce(codewords, balls, total, q, n)
        
        # Verify
        coverage = np.zeros(total, dtype=np.int32)
        for cw in codewords:
            for idx in balls[cw]:
                coverage[idx] += 1
        if (coverage >= 1).all():
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords
                if best_len == 73:
                    break # Found optimal known size
    
    if best_codewords is None:
        # Fallback: Tetracode sum (81 words)
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                codewords.append(h + list(tail))
        return codewords
        
    return [list(_unflat(cw, q, n)) for cw in best_codewords]


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    # Check for the specific problem instance
    if q == 3 and n == 6 and radius == 1:
        result = _construct_k3_6_1(q, n, radius)
        if result is not None:
            return result
    
    # General case: Greedy + Local Search
    total = q ** n
    balls = _precompute_balls(q, n, radius)
    
    # Greedy
    codewords = _greedy_cover(balls, total, q, n)
    
    # Local Search
    codewords = _local_search_reduce(codewords, balls, total, q, n)
    
    # Verify
    coverage = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        for idx in balls[cw]:
            coverage[idx] += 1
            
    if (coverage >= 1).all():
        return [list(_unflat(cw, q, n)) for cw in codewords]
    else:
        # Fallback: All words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
