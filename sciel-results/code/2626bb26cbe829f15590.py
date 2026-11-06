import time
import random
import itertools

def _check_validity(colors, n=16):
    """Check every K4 spans >= 5 distinct colours."""
    ec = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ec[(i, j)] = colors[idx]
            idx += 1
    
    # Precompute K4s for speed if called repeatedly? 
    # For n=16, 1820 K4s. 
    # Checking 1820 sets of 6 colors is fast.
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    if len({ec[(a,b)], ec[(a,c)], ec[(a,d)],
                            ec[(b,c)], ec[(b,d)], ec[(c,d)]}) < 5:
                        return False
    return True

def _get_colors_from_linear(alpha, beta, p, n=16):
    return [(alpha * i + beta * j) % p
            for i in range(n) for j in range(i + 1, n)]

def construct():
    t0 = time.time()
    deadline = t0 + 110
    
    best = None
    best_n = 999
    
    # 1. Exhaustive search for linear colorings (alpha*i + beta*j) % p
    # We check p from 13 upwards. If we find 13, we can stop immediately as it's the lower bound.
    for p in range(13, 17):
        # Optimization: alpha and beta should be coprime to p? Not strictly necessary for validity, 
        # but helps ensure diversity. We just iterate all.
        for alpha in range(1, p):
            for beta in range(1, p):
                if time.time() > deadline:
                    break
                
                colors = _get_colors_from_linear(alpha, beta, p)
                used = len(set(colors))
                
                # If used colors is already >= best_n, skip check
                if used >= best_n:
                    continue
                
                if _check_validity(colors):
                    best = colors
                    best_n = used
                    if best_n == 13:
                        return colors
        if time.time() > deadline:
            break

    # 2. If linear didn't find 13, try some random structured perturbations or random restarts for 13/14
    # Since random 13-coloring is unlikely, we try to repair a random 13-coloring or use a greedy approach.
    # However, given the time limit, we stick to the robust linear search which is likely to find the optimal 
    # if it exists in that family.
    
    # If we haven't found a solution with <= 15 colors, we should ensure we return a valid one.
    # The linear search for p=13..16 should find something.
    # If p=13 fails, p=14 might work.
    
    if best is not None:
        return best
    
    # Fallback: A known valid coloring. 
    # (i+j) % 17 uses 17 colors.
    # Let's try to construct a 15-coloring if linear failed (unlikely).
    # We can use a random search with a tight deadline for 15 colors.
    if time.time() < deadline - 10:
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            colors = [random.randint(0, 14) for _ in range(120)]
            if _check_validity(colors):
                return colors

    # Final fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
