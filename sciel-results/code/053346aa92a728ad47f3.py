import time
import itertools
import numpy as np

def _tetracode_solution(q, n, radius):
    """Tetracode direct-sum construction: valid radius-1 covering for q=3."""
    if q != 3 or n < 4:
        # Fallback: full space
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]

def _ball_indices(word, q, n, radius, word_to_idx):
    """Return list of indices of words within Hamming distance <= radius from word."""
    idx = word_to_idx[word]
    result = [idx]
    if radius >= 1:
        for i in range(n):
            for d in range(q):
                if d != word[i]:
                    new_w = list(word)
                    new_w[i] = d
                    result.append(word_to_idx[tuple(new_w)])
    return result

def _prune(chosen, balls):
    """Remove redundant codewords. chosen: list of indices, balls: list of lists of indices."""
    if not chosen:
        return []
    total = len(balls)
    coverage = np.zeros(total, dtype=np.int32)
    for c in chosen:
        for j in balls[c]:
            coverage[j] += 1
    keep = [True] * len(chosen)
    any_removed = True
    while any_removed:
        any_removed = False
        for i, c in enumerate(chosen):
            if not keep[i]:
                continue
            redundant = True
            for j in balls[c]:
                if coverage[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                any_removed = True
                for j in balls[c]:
                    coverage[j] -= 1
    return [chosen[i] for i in range(len(chosen)) if keep[i]]

def _local_search(chosen, balls, total, deadline):
    """Targeted swap local search."""
    chosen = _prune(chosen, balls)
    chosen_set = set(chosen)
    
    while time.time() < deadline:
        improved = False
        chosen_set = set(chosen)
        
        coverage = np.zeros(total, dtype=np.int32)
        for c in chosen:
            for j in balls[c]:
                coverage[j] += 1
        
        for i, c in enumerate(chosen):
            if time.time() > deadline:
                break
            
            unique_pts = [j for j in balls[c] if coverage[j] == 1]
            if not unique_pts:
                continue
            
            # Find candidates that cover all unique_pts
            candidates = set(balls[unique_pts[0]])
            for uj in unique_pts[1:]:
                candidates &= set(balls[uj])
            candidates -= chosen_set
            
            for c_prime in candidates:
                tentative = list(chosen)
                tentative[i] = c_prime
                pruned = _prune(tentative, balls)
                if len(pruned) < len(chosen):
                    chosen = pruned
                    chosen_set = set(chosen)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return chosen

def construct(q, n, radius):
    deadline = time.time() + 115.0
    total = q ** n
    
    try:
        # Build word index map
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Start with tetracode direct-sum (valid for q=3, n>=4)
        initial = _tetracode_solution(q, n, radius)
        chosen = [word_to_idx[w] for w in initial]
        
        # Precompute balls for chosen codewords only (to save memory/time)
        # But we need balls for local search candidates too.
        # For n=8, q=3, total=6561. Precompute all balls.
        balls = []
        for w in all_words:
            idx = word_to_idx[w]
            nbrs = [idx]
            if radius >= 1:
                for i in range(n):
                    for d in range(q):
                        if d != w[i]:
                            nw = list(w)
                            nw[i] = d
                            nbrs.append(word_to_idx[tuple(nw)])
            balls.append(nbrs)
        
        # Prune initial
        chosen = _prune(chosen, balls)
        
        # Local search
        chosen = _local_search(chosen, balls, total, deadline)
        
        # Final prune
        chosen = _prune(chosen, balls)
        
        # Verify validity
        covered = np.zeros(total, dtype=bool)
        for c in chosen:
            for j in balls[c]:
                covered[j] = True
        if not np.all(covered):
            # Fallback to tetracode
            return _tetracode_solution(q, n, radius)
        
        return [list(all_words[i]) for i in chosen]
    
    except Exception:
        return _tetracode_solution(q, n, radius)
