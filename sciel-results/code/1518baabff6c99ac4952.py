import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def _build_structures(n):
    """
    Precompute adjacency structures for fast local search.
    Returns:
    - m: number of triples
    - tri_quads: list of lists, tri_quads[i] = list of quad indices containing triple i
    - tri_pents: list of lists, tri_pents[i] = list of pent indices containing triple i
    - quads: list of lists of triple indices
    - pents: list of lists of triple indices
    """
    triples = list(combinations(range(n), 3))
    m = len(triples)
    triples_set = {t: i for i, t in enumerate(triples)}
    
    # Build quads
    quads = []
    for q in combinations(range(n), 4):
        a,b,c,d = q
        idxs = [
            triples_set[(a,b,c)],
            triples_set[(a,b,d)],
            triples_set[(a,c,d)],
            triples_set[(b,c,d)]
        ]
        quads.append(idxs)
        
    # Build pents
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(triples_set[t])
        pents.append(idxs)
        
    # Build reverse maps
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
            
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)
            
    return m, tri_quads, tri_pents, quads, pents

def _algebraic_seed(n):
    """
    Generate a deterministic coloring based on modular arithmetic.
    We want to avoid Red K4 (4 triples all 0) and Blue K5 (10 triples all 1).
    Using (a+b+c) % 3:
    - In a set of 4 vertices, the sum of the 4 triples is 2*(sum of vertices).
    - If we color 0 if sum%3==0, we might still get K4.
    - A better heuristic: color 0 if (a+b+c) % 3 == 0, else 1.
    - Actually, let's try a few patterns and pick the one with fewest violations.
    """
    best_colors = None
    best_violations = float('inf')
    
    triples = list(combinations(range(n), 3))
    quads = []
    for q in combinations(range(n), 4):
        a,b,c,d = q
        idxs = [
            (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        ]
        quads.append(idxs)
        
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(t)
        pents.append(idxs)
        
    # Try a few simple algebraic rules
    rules = [
        lambda a,b,c: (a + b + c) % 3 == 0,
        lambda a,b,c: (a + 2*b + 3*c) % 5 == 0,
        lambda a,b,c: (a*b + b*c + c*a) % 7 == 0,
        lambda a,b,c: (a + b + c) % 2 == 0,
    ]
    
    for rule in rules:
        colors = np.array([0 if rule(a,b,c) else 1 for a,b,c in triples], dtype=np.int8)
        
        # Count violations
        red_k4 = 0
        for idxs in quads:
            # Map indices to colors
            # Need index mapping for speed in this check, but n is small enough for this one-time check?
            # No, let's just use the index mapping from _build_structures logic if we had it.
            # For simplicity in seed generation, let's just count on the fly if n is small,
            # but for n=34, C(34,4)=46376, which is fine.
            # We need to map (a,b,c) to index.
            pass
        
        # This approach is too slow if we don't have indices. Let's just use random for seeds 
        # but keep the SA loop optimized. The key improvement is the SA loop speed.
        
    # Fallback: Random seed with bias towards Red (since K4 is smaller than K5, 
    # and random 0.5/0.5 might create too many blue K5s if n is large? 
    # Actually, P(blue K5) = (1/2)^10 = 1/1024. P(red K4) = (1/2)^4 = 1/16.
    # So Red violations are much more likely in random.
    # To minimize Red K4, we should bias towards Blue?
    # If p_red is low, red K4 is very unlikely.
    # If p_red is low, blue K5 is likely.
    # We want a balance.
    return (np.random.rand(len(triples)) < 0.4).astype(np.int8)

def _sa_local_search_vectorized(colors, m, tri_quads, tri_pents, quads, pents, time_budget, temp_start=1.0):
    """
    Optimized Simulated Annealing.
    Key optimization: Use NumPy arrays for counts and vectorized updates where possible,
    or at least minimize Python overhead in the inner loop.
    """
    colors = colors.copy()
    quad_red = np.zeros(len(quads), dtype=np.int16)
    pent_blue = np.zeros(len(pents), dtype=np.int16)
    
    # Initialize counts
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1
                
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    total_v = red_k4 + blue_k5
    
    if total_v == 0:
        return colors, 0, 0

    t_end = time.time() + time_budget
    temp = temp_start
    cooling = 0.995
    
    # Pre-compute lists for faster access
    # tri_quads and tri_pents are lists of lists.
    
    while time.time() < t_end:
        # Shuffle indices
        order = np.random.permutation(m)
        
        for t in order:
            if time.time() >= t_end:
                break
            if temp < 0.01:
                break
                
            cur = colors[t]
            new = 1 - cur
            
            # Calculate delta in violations
            delta_red = 1 if new == 0 else -1
            delta_blue = 1 if new == 1 else -1
            
            dq = 0
            # Iterate over quads containing t
            for qi in tri_quads[t]:
                old = quad_red[qi]
                new_q = old + delta_red
                if old == 4: dq -= 1
                if new_q == 4: dq += 1
                
            dp = 0
            for pi in tri_pents[t]:
                old = pent_blue[pi]
                new_p = old + delta_blue
                if old == 10: dp -= 1
                if new_p == 10: dp += 1
            
            delta_v = dq + dp
            
            if delta_v < 0:
                # Accept
                colors[t] = new
                for qi in tri_quads[t]:
                    quad_red[qi] += delta_red
                for pi in tri_pents[t]:
                    pent_blue[pi] += delta_blue
                red_k4 += dq
                blue_k5 += dp
                total_v += delta_v
            else:
                # Accept with probability exp(-delta_v / temp)
                if delta_v > 0:
                    # Optimization: avoid exp if possible
                    if delta_v > temp * 5: # approx exp(-5) is small
                        pass
                    else:
                        prob = np.exp(-delta_v / temp)
                        if random.random() < prob:
                            colors[t] = new
                            for qi in tri_quads[t]:
                                quad_red[qi] += delta_red
                            for pi in tri_pents[t]:
                                pent_blue[pi] += delta_blue
                            red_k4 += dq
                            blue_k5 += dp
                            total_v += delta_v
            
        temp *= cooling
        if total_v == 0:
            return colors, 0, 0
            
    return colors, red_k4, blue_k5

def _try_n(n, time_budget):
    """
    Try to find valid coloring for n.
    """
    m, tri_quads, tri_pents, quads, pents = _build_structures(n)
    
    # Multiple restarts
    attempts = 0
    while time.time() < _START + time_budget:
        attempts += 1
        # Random init with bias
        # Bias towards blue to avoid red K4?
        # Let's try p_red = 0.3
        p_red = 0.3
        colors = (np.random.rand(m) < p_red).astype(np.int8)
        
        # Run SA
        colors, r4, b5 = _sa_local_search_vectorized(colors, m, tri_quads, tri_pents, quads, pents, 2.0)
        if r4 == 0 and b5 == 0:
            return colors.tolist()
            
    return None

def construct():
    deadline = _START + _DEADLINE - 2.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try n=34 first
    targets = [34, 33, 32, 31, 30]
    budgets = [70, 30, 20, 10, 5]
    
    for n, budget in zip(targets, budgets):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < budget:
            budget = remaining - 1.0
        if budget < 2.0:
            break
            
        colors = _try_n(n, budget)
        if colors is not None:
            best_n = n
            best_colors = colors
            break # Found valid, stop.
            
    return {'n': best_n, 'colors': best_colors}
