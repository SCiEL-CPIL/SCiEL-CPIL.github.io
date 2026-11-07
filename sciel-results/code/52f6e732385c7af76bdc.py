"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types
    # top_edges: indices of edges (i, i+1) in 0..m-1
    # bot_edges: indices of edges (i+m, i+1+m)
    # match_edges: indices of edges (i, i+m)
    
    top_edge_idx = {}
    bot_edge_idx = {}
    match_edge_idx = {}
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle
            # Normalize to (min, max) or just store
            top_edge_idx[(min(u,v), max(u,v))] = i
        elif u >= m and v >= m:
            # Bottom cycle
            bot_edge_idx[(min(u,v), max(u,v))] = i
        else:
            # Matching
            # u < m, v >= m or vice versa
            match_edge_idx[min(u, v)] = i # The vertex in 0..m-1

    # Lists for fast access
    top_signs_idx = [top_edge_idx[(i, (i+1)%m)] for i in range(m)]
    bot_signs_idx = [bot_edge_idx[(m+i, m+(i+1)%m)] for i in range(m)]
    match_signs_idx = [match_edge_idx[i] for i in range(m)]
    
    def get_top_signs(signs):
        return signs[top_signs_idx]
        
    def get_bot_signs(signs):
        return signs[bot_signs_idx]
        
    def get_match_signs(signs):
        return signs[match_signs_idx]

    def is_valid_fast(signs):
        # Check if lift is connected
        # Invalid if:
        # 1. Product of top signs == 1
        # 2. Product of bot signs == 1
        # 3. Product of square signs == 1 for all i
        # Square i involves: top_edge(i), bot_edge(i), match_edge(i), match_edge(i+1)
        
        top_prod = np.prod(get_top_signs(signs))
        if top_prod == -1:
            return True
            
        bot_prod = np.prod(get_bot_signs(signs))
        if bot_prod == -1:
            return True
            
        # Check squares
        # Q_i = s_top[i] * s_bot[i] * s_match[i] * s_match[i+1]
        # Note: Edge (i, i+1) is top[i]. Edge (i+m, i+1+m) is bot[i].
        # Matching edges connect i to i+m.
        # Square i is cycle i -> i+1 -> i+1+m -> i+m -> i
        # Edges: (i, i+1), (i+1, i+1+m), (i+1+m, i+m), (i+m, i)
        # Signs: top[i], match[i+1], bot[i], match[i]
        
        top_s = get_top_signs(signs)
        bot_s = get_bot_signs(signs)
        match_s = get_match_signs(signs)
        
        for i in range(m):
            q = top_s[i] * bot_s[i] * match_s[i] * match_s[(i+1)%m]
            if q == -1:
                return True
        return False

    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def evaluate(signs):
        if not is_valid_fast(signs):
            return float('inf')
        return spectral_radius(signs)

    # Strategy:
    # 1. If num_edges <= 15, exhaustive search.
    # 2. Else, Simulated Annealing / Random Restarts Local Search.
    
    best_signs = None
    best_radius = float('inf')
    
    start_time = time.time()
    time_limit = 55 # Leave buffer for overhead
    
    if num_edges <= 15:
        # Exhaustive search
        # Use numpy to generate all combinations?
        # 2^15 = 32768.
        # We can iterate through integers 0 to 2^num_edges - 1
        for mask in range(1 << num_edges):
            # Convert mask to signs
            # bits: 0 -> +1, 1 -> -1
            signs = np.ones(num_edges, dtype=int)
            # Efficient conversion
            # Using bit operations
            for i in range(num_edges):
                if (mask >> i) & 1:
                    signs[i] = -1
            
            rad = evaluate(signs)
            if rad < best_radius:
                best_radius = rad
                best_signs = signs.copy()
                
            if time.time() - start_time > time_limit:
                break
    else:
        # Heuristic Search
        # Multiple restarts
        rng = np.random.default_rng(42)
        
        # Initial candidates
        candidates = []
        
        # 1. All +1
        candidates.append(np.ones(num_edges, dtype=int))
        # 2. All -1
        candidates.append(np.ones(num_edges, dtype=int) * -1)
        # 3. Alternating top, +1 bot, -1 match
        # 4. Random valid
        for _ in range(100):
            s = rng.choice([-1, 1], size=num_edges)
            # Ensure validity by flipping if needed?
            # Just add, evaluate will filter
            candidates.append(s)
            
        # Evaluate initial
        for s in candidates:
            rad = evaluate(s)
            if rad < best_radius:
                best_radius = rad
                best_signs = s.copy()
                
        if best_signs is None:
            # Fallback: find any valid
            # Flip random edges until valid
            s = np.ones(num_edges, dtype=int)
            while not is_valid_fast(s):
                idx = rng.integers(0, num_edges)
                s[idx] *= -1
            best_signs = s
            best_radius = spectral_radius(best_signs)
            
        # Local Search with Random Restarts
        # Use fast validity check to allow many steps
        
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        for restart in range(50): # Limit restarts
            if time.time() - start_time > time_limit:
                break
                
            # If we found a good one, maybe do local search from it
            # Else start from random
            if restart == 0:
                curr = best_signs.copy()
                curr_rad = best_radius
            else:
                curr = rng.choice([-1, 1], size=num_edges)
                if not is_valid_fast(curr):
                    # Try to repair
                    for _ in range(1000):
                        idx = rng.integers(0, num_edges)
                        curr[idx] *= -1
                        if is_valid_fast(curr):
                            break
                if not is_valid_fast(curr):
                    continue
                curr_rad = spectral_radius(curr)
                
            improved = True
            while improved:
                if time.time() - start_time > time_limit:
                    break
                improved = False
                # Try flipping each edge
                for i in range(num_edges):
                    trial = curr.copy()
                    trial[i] *= -1
                    if not is_valid_fast(trial):
                        continue
                    trial_rad = spectral_radius(trial)
                    if trial_rad < curr_rad:
                        curr_rad = trial_rad
                        curr = trial
                        improved = True
                        if curr_rad < best_radius:
                            best_radius = curr_rad
                            best_signs = curr.copy()
                        
                # If no single flip improved, try double flips?
                # Might be too slow. Single flip is good enough with restarts.
                
        # Final check
        if best_signs is None:
            best_signs = np.ones(num_edges, dtype=int) * -1

    return best_signs
# EVOLVE-BLOCK-END
