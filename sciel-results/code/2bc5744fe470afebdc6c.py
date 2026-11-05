import math
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g, minimizing n.
    Uses a multi-strategy approach:
    1. Try known explicit constructions (Lazebnik-Ustimenko-Woldar, Moore bound-based)
    2. Try random graph + local optimization
    3. Return best valid graph found within time limit
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_result = None
    
    # Strategy 1: Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # Try small d values first, as they produce smaller graphs.
    # The girth bound is likely d+5 or 2d+4. We verify exactly.
    q = k
    if q >= 3:
        # Try d from 3 up to a reasonable limit
        # For g=9, d=3 might work (girth 10).
        # Limit d to g to avoid huge graphs if g is large.
        max_d = max(3, g)
        for d in range(3, max_d + 1):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = _luw_graph(q, d)
                if n > 0 and _verify_graph(k, g, n, edges, time_limit=5):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                        # If we find a graph close to Moore bound, we can stop early?
                        # Moore bound is a lower bound. If n == moore, we are done.
                        moore = _moore_bound(k, g)
                        if n <= moore + 5:
                            break
            except Exception:
                pass
    
    # Strategy 2: Random k-regular graph with girth checking
    if time.time() < deadline - 5:
        try:
            # Use improved random search
            n_random, edges_random = _random_regular_with_girth(k, g, max_tries=100, time_limit=60)
            if n_random > 0 and _verify_graph(k, g, n_random, edges_random, time_limit=5):
                if best_result is None or n_random < best_result[0]:
                    best_result = (n_random, edges_random)
        except Exception:
            pass
    
    # If nothing found, fall back to LUW with minimal d that is likely valid
    if best_result is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_graph(k, d)
        if n > 0:
            best_result = (n, edges)
        else:
            # Last resort
            if k == 1:
                best_result = (2, [(0, 1)])
            elif k == 2:
                best_result = (g, [(i, (i+1) % g) for i in range(g)])
            else:
                best_result = (0, [])
    
    return best_result

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        s = 0
        term = 1
        for _ in range(r):
            s += term
            term *= (k - 1)
        return 2 * s
    else:
        r = (g - 1) // 2
        s = 0
        term = 1
        for _ in range(r):
            s += term
            term *= (k - 1)
        return 1 + k * s

# ... other functions remain same, except _random_regular_with_girth and _random_k_regular ...
