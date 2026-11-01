"""Improved covering code construction for K_3(7,1) using greedy + local search."""
import time
import random
import itertools
from collections import defaultdict

def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))

def _build_cover_map(codewords, n, radius):
    """Return set of all words covered by codewords."""
    covered = set()
    for cw in codewords:
        cw = tuple(cw)
        # Add cw itself
        covered.add(cw)
        # Add all words at distance 1
        for i in range(n):
            for v in range(3):
                if v != cw[i]:
                    lst = list(cw)
                    lst[i] = v
                    covered.add(tuple(lst))
    return covered

def _is_covered(codewords, n, radius, target):
    """Check if target is within radius of some codeword."""
    target = tuple(target)
    for cw in codewords:
        if _hamming_dist(target, cw) <= radius:
            return True
    return False

def _greedy_cover(q, n, radius):
    """Greedy set cover: repeatedly pick codeword covering most uncovered words."""
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []
    
    # Precompute for each potential codeword, which words it covers
    # But that's 3^7 * 3^7 which is too big. Instead, use lazy evaluation.
    
    # For efficiency, we'll use a different approach:
    # Start with a structured initial set, then greedily add
    
    # Actually for n=7, q=3, let's use a smarter greedy:
    # Partition into groups and cover each group
    
    # Better: use the fact that radius=1 means each codeword covers 1 + 7*2 = 15 words
    # Total words = 2187, so minimum ~146 codewords needed
    
    # Greedy: pick codeword that maximizes new coverage
    # To make this fast, we'll use random sampling + greedy
    
    # Initial: use the tetracode-based construction as starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Now try to improve by removing redundant codewords
    codewords = list(initial)
    uncovered = set(all_words)
    
    # Compute coverage for initial set
    for cw in codewords:
        cw_t = tuple(cw)
        uncovered.discard(cw_t)
        for i in range(n):
            for v in range(q):
                if v != cw_t[i]:
                    lst = list(cw_t)
                    lst[i] = v
                    uncovered.discard(tuple(lst))
    
    # Try to remove each codeword if it doesn't reduce coverage
    improved = True
    while improved and len(uncovered) == 0:
        improved = False
        for idx in range(len(codewords) - 1, -1, -1):
            cw = codewords[idx]
            cw_t = tuple(cw)
            # Check if removing this codeword leaves all its covered words still covered
            # by other codewords
            needs_removal_check = True
            # The words uniquely covered by this codeword (or covered by it)
            # We need to check: for each word within radius 1 of cw, is it still covered by others?
            other_cw = codewords[:idx] + codewords[idx+1:]
            # Check all words within distance 1 of cw
            candidates = [cw_t]
            for i in range(n):
                for v in range(q):
                    if v != cw_t[i]:
                        lst = list(cw_t)
                        lst[i] = v
                        candidates.append(tuple(lst))
            
            all_still_covered = True
            for w in candidates:
                if w in uncovered:
                    all_still_covered = False
                    break
                # Check if covered by other codewords
                found = False
                for ocw in other_cw:
                    if _hamming_dist(w, ocw) <= radius:
                        found = True
                        break
                if not found:
                    all_still_covered = False
                    break
            
            if all_still_covered:
                codewords.pop(idx)
                improved = True
                break
    
    return codewords

def _local_search(codewords, q, n, radius, deadline):
    """Local search with 1-for-1 removals, 2-for-1, and 3-for-2 swaps."""
    all_words = list(itertools.product(range(q), repeat=n))
    codewords = [list(cw) for cw in codewords]
    
    def compute_uncovered(cw_list):
        covered = set()
        for cw in cw_list:
            cw_t = tuple(cw)
            covered.add(cw_t)
            for i in range(n):
                for v in range(q):
                    if v != cw_t[i]:
                        lst = list(cw_t)
                        lst[i] = v
                        covered.add(tuple(lst))
        return set(all_words) - covered
    
    uncovered = compute_uncovered(codewords)
    
    # 1-for-1: try removing each codeword
    while True:
        if time.time() > deadline:
            break
        removed = False
        for idx in range(len(codewords) - 1, -1, -1):
            cw = codewords[idx]
            cw_t = tuple(cw)
            # Words covered by this codeword
            covered_by_this = set()
            covered_by_this.add(cw_t)
            for i in range(n):
                for v in range(q):
                    if v != cw_t[i]:
                        lst = list(cw_t)
                        lst[i] = v
                        covered_by_this.add(tuple(lst))
            
            # Check if all these words are still covered by remaining codewords
            others = codewords[:idx] + codewords[idx+1:]
            all_covered = True
            for w in covered_by_this:
                found = False
                for ocw in others:
                    if _hamming_dist(w, ocw) <= radius:
                        found = True
                        break
                if not found:
                    all_covered = False
                    break
            
            if all_covered:
                codewords.pop(idx)
                removed = True
                break
        
        if not removed:
            break
    
    # 2-for-1: try replacing two codewords with one
    while True:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                # Try all possible single replacement codewords
                # The replacement must cover all words that were uniquely covered by i or j
                # Simplified: try random candidate replacements
                for _ in range(5):  # Try a few random candidates
                    candidate = [random.randint(0, q-1) for _ in range(n)]
                    new_list = codewords[:i] + codewords[i+1:j] + codewords[j+1:] + [candidate]
                    unc = compute_uncovered(new_list)
                    if len(unc) == 0:
                        codewords = new_list
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # 3-for-2: try replacing three codewords with two
    while True:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                for k in range(j + 1, len(codewords)):
                    # Try random pairs of replacements
                    for _ in range(3):
                        c1 = [random.randint(0, q-1) for _ in range(n)]
                        c2 = [random.randint(0, q-1) for _ in range(n)]
                        new_list = codewords[:i] + codewords[i+1:j] + codewords[j+1:k] + codewords[k+1:] + [c1, c2]
                        unc = compute_uncovered(new_list)
                        if len(unc) == 0:
                            codewords = new_list
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with tetracode-based construction
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    best = codewords[:]
    
    # Greedy improvement: try removing redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(len(codewords) - 1, -1, -1):
            cw = codewords[idx]
            cw_t = tuple(cw)
            others = codewords[:idx] + codewords[idx+1:]
            
            # Check if removing this codeword still covers everything
            # Words within distance 1 of cw
            covered_by_this = [cw_t]
            for i in range(n):
                for v in range(q):
                    if v != cw_t[i]:
                        lst = list(cw_t)
                        lst[i] = v
                        covered_by_this.append(tuple(lst))
            
            all_covered = True
            for w in covered_by_this:
                found = False
                for ocw in others:
                    if _hamming_dist(w, ocw) <= radius:
                        found = True
                        break
                if not found:
                    all_covered = False
                    break
            
            if all_covered:
                codewords.pop(idx)
                improved = True
                break
    
    if len(codewords) < len(best):
        best = codewords[:]
    
    # Local search with randomization
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Start from current best or a perturbed version
        if restart == 0:
            current = best[:]
        else:
            current = best[:]
            # Perturb: replace some codewords with random ones
            for _ in range(min(3, len(current))):
                if random.random() < 0.5:
                    current[random.randint(0, len(current)-1)] = [random.randint(0, q-1) for _ in range(n)]
        
        current = _local_search(current, q, n, radius, deadline - 5)
        
        if len(current) < len(best):
            best = current[:]
    
    return best
