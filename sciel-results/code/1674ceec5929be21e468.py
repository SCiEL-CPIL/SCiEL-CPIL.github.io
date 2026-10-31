"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    n = len(U)
    s = len(sumset)
    d = len(diffset)
    
    if s == 0 or d == 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    ratio = d / s
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / denom
    return score


def _two_block_construct(a, b, n_a, n_b):
    """
    Construct U as union of two scaled arithmetic progressions.
    A = {0, a, 2a, ..., (n_a-1)*a}
    B = {b, 2b, 3b, ..., n_b*b}  (offset to avoid overlap with A mostly)
    
    Actually, let's use:
    A = {k * a : k in range(n_a)}  -> {0, a, 2a, ...}
    B = {offset + k * b : k in range(n_b)} where offset is chosen to separate
    
    To minimize sumset collisions and maximize difference spread,
    choose a and b multiplicatively independent (e.g., a=2, b=3, or a=3, b=5).
    """
    A = set(k * a for k in range(n_a))
    # Offset B so it doesn't overlap too much with A
    # Use offset = a (or some multiple) to shift B
    offset = a * n_a  # Place B after A
    B = set(offset + k * b for k in range(n_b))
    U = A | B
    # Ensure 0 is in U
    if 0 not in U:
        U.add(0)
    return sorted(U)


def _refine_set(U, max_iterations=50, timeout=None):
    """
    Local refinement: try adding or removing elements to improve score.
    """
    U_set = set(U)
    current_score = _score_set(sorted(U_set))
    
    start_time = time.time()
    
    for _ in range(max_iterations):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        improved = False
        # Try adding elements
        max_u = max(U_set) if U_set else 0
        # Candidate positions to try
        candidates = set()
        # Try small additions
        for x in range(1, min(max_u + 20, 200)):
            if x not in U_set:
                candidates.add(x)
        # Try some larger candidates
        for x in range(max_u + 1, max_u + 10):
            candidates.add(x)
        
        best_candidate = None
        best_score = current_score
        
        for x in candidates:
            new_U = U_set | {x}
            s = _score_set(sorted(new_U))
            if s > best_score:
                best_score = s
                best_candidate = x
        
        if best_candidate is not None:
            U_set.add(best_candidate)
            current_score = best_score
            improved = True
        
        # Try removing elements (except 0)
        if not improved:
            for x in list(U_set):
                if x == 0:
                    continue
                new_U = U_set - {x}
                s = _score_set(sorted(new_U))
                if s > current_score:
                    U_set.remove(x)
                    current_score = s
                    improved = True
                    break
        
        if not improved:
            break
    
    return sorted(U_set), current_score


def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Explore two-block structures with various parameters
    # Use multiplicatively independent scaling factors
    scaling_pairs = [(2, 3), (3, 5), (2, 5), (3, 7), (5, 7), (2, 7), (3, 11), (5, 11)]
    
    for a, b in scaling_pairs:
        if time.time() > deadline:
            break
        
        for n_a in [5, 10, 15, 20, 30, 40]:
            if time.time() > deadline:
                break
            
            for n_b in [5, 10, 15, 20, 30, 40]:
                if time.time() > deadline:
                    break
                
                U = _two_block_construct(a, b, n_a, n_b)
                score = _score_set(U)
                
                if score > best_score:
                    best_score = score
                    best_U = U
    
    # Phase 2: Refine the best set found
    if time.time() < deadline:
        refined_U, refined_score = _refine_set(best_U, max_iterations=30, 
                                                timeout=max(0, deadline - time.time() - 5))
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
    
    # Phase 3: Try a few more structural variations
    if time.time() < deadline - 10:
        # Try geometric-like structures
        for base in [2, 3, 4]:
            if time.time() > deadline:
                break
            for n in [8, 12, 16, 20]:
                if time.time() > deadline:
                    break
                U = sorted(set(base ** k for k in range(n)))
                score = _score_set(U)
                if score > best_score:
                    best_score = score
                    best_U = U
    
    # Phase 4: Final refinement if time permits
    if time.time() < deadline - 5:
        refined_U, refined_score = _refine_set(best_U, max_iterations=20,
                                                timeout=max(0, deadline - time.time() - 3))
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
    
    return best_U

# EVOLVE-BLOCK-END
