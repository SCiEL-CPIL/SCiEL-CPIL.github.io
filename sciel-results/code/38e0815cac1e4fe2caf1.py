"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin

def construct():
    """
    Returns a valid coloring for n vertices.
    Strategy:
    1. Use Simulated Annealing with vectorized NumPy updates to find valid colorings.
    2. Iterate n from high to low, allocating more time to larger n initially.
    3. Return the first valid coloring found (highest n).
    """
    
    def _triples_to_idx(n):
        triples = list(combinations(range(n), 3))
        m = len(triples)
        idx = {}
        for i, t in enumerate(triples):
            idx[t] = i
        return triples, idx, m

    def _build_quads(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            idxs = [t_idx(a, b, c), t_idx(a, b, d), t_idx(a, c, d), t_idx(b, c, d)]
            quads.append(idxs)
        return quads

    def _build_pents(n):
        m = len(list(combinations(range(n), 3)))
        def t_idx(a, b, c):
            cnt = 0
            for x in range(a):
                cnt += (n - x - 1) * (n - x - 2) // 2
            for y in range(a + 1, b):
                cnt += (n - y - 1)
            cnt += (c - b - 1)
            return cnt
        pents = []
        for p in combinations(range(n), 5):
            a, b, c, d, e = p
            idxs = []
            for t in combinations(p, 3):
                idxs.append(t_idx(*t))
            pents.append(idxs)
        return pents

    def _count_violations(colors, quads, pents):
        red_k4 = 0
        for idxs in quads:
            if all(colors[i] == 0 for i in idxs):
                red_k4 += 1
        blue_k5 = 0
        for idxs in pents:
            if all(colors[i] == 1 for i in idxs):
                blue_k5 += 1
        return red_k4, blue_k5

    def _simulated_annealing(n, time_budget):
        """
        Finds a valid coloring for n vertices using Simulated Annealing.
        Returns None if no valid coloring is found within time_budget.
        """
        triples, idx, m = _triples_to_idx(n)
        quads = _build_quads(n)
        pents = _build_pents(n)
        
        # Precompute incidence lists for fast delta calculation
        # tri_quads[t] = list of quad indices containing triple t
        # tri_pents[t] = list of pent indices containing triple t
        tri_quads = [[] for _ in range(m)]
        for qi, idxs in enumerate(quads):
            for i in idxs:
                tri_quads[i].append(qi)
        
        tri_pents = [[] for _ in range(m)]
        for pi, idxs in enumerate(pents):
            for i in idxs:
                tri_pents[i].append(pi)
        
        # Initial random coloring
        # Probability of red (0) should be balanced to avoid immediate violations
        # For R(4,5;3), red K4 is smaller, so red should be less frequent?
        # Actually, let's start with p_red = 0.3 or 0.4
        p_red = 0.35
        colors = (np.random.rand(m) < p_red).astype(np.int8)
        
        # State: number of red edges in each quad, blue edges in each pent
        quad_red = np.zeros(len(quads), dtype=np.int8)
        pent_blue = np.zeros(len(pents), dtype=np.int8)
        
        for qi, idxs in enumerate(quads):
            quad_red[qi] = np.sum(colors[idxs] == 0)
        for pi, idxs in enumerate(pents):
            pent_blue[pi] = np.sum(colors[idxs] == 1)
            
        red_k4 = int(np.sum(quad_red == 4))
        blue_k5 = int(np.sum(pent_blue == 10))
        total_v = red_k4 + blue_k5
        
        if total_v == 0:
            return colors.tolist()
            
        t_start = time.time()
        t_end = t_start + time_budget
        
        # Annealing parameters
        # Temperature needs to be high enough to accept moves that increase violations initially
        # Cost function: total_v. Move cost delta can be -1, 0, +1, etc.
        # Actually, changing one triple affects at most C(3,1)*C(n-3,1) quads? No.
        # A triple (a,b,c) is in quads {a,b,c,d}. There are n-3 such quads.
        # So delta_red_k4 can be in range [-3, 3] roughly? No, each quad count changes by +/- 1.
        # If a quad was 4 reds and we flip one to blue, it becomes 3 reds. Violation removed (-1).
        # If a quad was 3 reds and we flip one to red, it becomes 4 reds. Violation added (+1).
        # So delta is integer.
        
        T = 1.0
        cooling = 0.999 # Very slow cooling to allow thorough search
        best_v = total_v
        best_colors = colors.copy()
        
        # We want to minimize total_v.
        # If total_v becomes 0, we return immediately.
        
        while time.time() < t_end:
            # Pick a random triple
            t_idx = random.randint(0, m - 1)
            cur_color = colors[t_idx]
            new_color = 1 - cur_color
            
            # Calculate delta in violations
            delta_red_k4 = 0
            delta_blue_k5 = 0
            
            # Update quad_red counts
            # For each quad containing t_idx:
            # If new_color is 0 (Red): quad_red[qi] increases by 1
            # If new_color is 1 (Blue): quad_red[qi] decreases by 1
            # We need to check if any quad crosses the threshold of 4.
            
            # Optimization: Instead of iterating all quads/pents for every step in Python,
            # we can just iterate the small lists tri_quads[t_idx] and tri_pents[t_idx].
            
            delta_q = 0
            for qi in tri_quads[t_idx]:
                old_val = quad_red[qi]
                new_val = old_val + (1 if new_color == 0 else -1)
                if old_val == 4 and new_val < 4:
                    delta_q -= 1
                elif old_val < 4 and new_val == 4:
                    delta_q += 1
                # Update state
                quad_red[qi] = new_val
                
            delta_p = 0
            for pi in tri_pents[t_idx]:
                old_val = pent_blue[pi]
                new_val = old_val + (1 if new_color == 1 else -1)
                if old_val == 10 and new_val < 10:
                    delta_p -= 1
                elif old_val < 10 and new_val == 10:
                    delta_p += 1
                # Update state
                pent_blue[pi] = new_val
                
            total_delta = delta_q + delta_p
            new_total_v = total_v + total_delta
            
            # Acceptance criterion
            if new_total_v < total_v or (T > 0 and random.random() < np.exp(-total_delta / T)):
                # Accept move
                colors[t_idx] = new_color
                total_v = new_total_v
                
                if total_v == 0:
                    return colors.tolist()
                    
                if total_v < best_v:
                    best_v = total_v
                    best_colors = colors.copy()
            else:
                # Reject move (state remains unchanged)
                pass
                
            T *= cooling
            if T < 0.01:
                T = 0.01 # Floor temperature to allow some random walks? Or restart?
                # Let's restart with a new random coloring if stuck too long?
                # For now, just keep going.
                
        return None

    # Try to find a valid coloring for n from 40 down to 10
    # The known bound is 34. We should focus on 30-40.
    # Allocate time budget based on n.
    
    # We'll try a few restarts for each n if time permits.
    # But SA is stochastic.
    
    # Let's prioritize higher n.
    # Time budget allocation:
    # n=40: 10s
    # n=39: 10s
    # ...
    # n=30: 10s
    # If we find one, return immediately.
    
    # To be safe, let's try n=34 first, as it's the published construction size.
    # If we can't find 34 quickly, try 33, 32, etc.
    # Or try 40 down to 10 with fixed budgets.
    
    # Given 145s limit, let's try:
    # 5 attempts for n=40 (5s each) -> 25s
    # 5 attempts for n=39 (5s each) -> 25s
    # ...
    # This might be too slow.
    
    # Better: Try n=40 for 20s. If fail, n=39 for 20s, etc.
    # If we reach n=30 and haven't found anything, something is wrong.
    
    # Actually, the current program scores 2.0 with n=32.
    # So finding n=32 is the baseline. We want to beat it, i.e., find n=33 or 34.
    
    # Let's start from n=35 and go down.
    # If we find a valid coloring, return it.
    
    for n in range(35, 9, -1):
        if time.time() > _START + _DEADLINE - 5.0:
            break
            
        # Budget for this n
        # If n is large, give it more time.
        # Let's give 10 seconds per attempt, 2 attempts per n.
        # Total 20s per n.
        # 35, 34, 33, 32, 31, 30 -> 6 * 20 = 120s. Fits.
        
        found = None
        for attempt in range(2):
            if time.time() > _START + _DEADLINE - 5.0:
                break
            budget = 10.0
            # Add small jitter to random seed? No, np.random is global.
            found = _simulated_annealing(n, budget)
            if found is not None:
                return {'n': n, 'colors': found}
                
    # Fallback: If we didn't find anything for n>=10, return a small valid one
    # e.g. n=9
    n = 9
    triples, idx, m = _triples_to_idx(n)
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
