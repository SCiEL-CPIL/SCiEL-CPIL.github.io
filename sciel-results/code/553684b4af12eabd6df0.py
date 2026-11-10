import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for return/eval overhead
    
    # We want to find the maximum n such that there exists a valid coloring.
    # Published construction has n=34.
    # Let's try to find a valid coloring for n=34.
    # If we fail, we try n=33, etc.
    
    # Precomputation is expensive, so we do it inside the search loop for specific n.
    # To save time, we first try n=34 with a decent time budget.
    
    # Strategy:
    # 1. Try n=34 for ~100s.
    # 2. If not found, try n=33 for ~20s.
    # 3. If not found, try smaller n quickly.
    # 4. Fallback to a known small valid coloring (n=9 or similar) if time runs out.
    
    # Actually, let's just try n=34 first. It's the target.
    # If we find it, we return it.
    # If we don't, we check time and try smaller n.
    
    # For n=34:
    # Number of triples: C(34, 3) = 5984
    # Number of K4s: C(34, 4) = 46376
    # Number of K5s: C(34, 5) = 278256
    
    # We will use a local search with incremental violation counting.
    # To make it fast, we use numpy for vectorized updates where possible, 
    # but the incremental update for a single flip is best done with precomputed incidence lists.
    
    # Let's define the search function for a specific n.
    
    def try_find_coloring(n, time_limit):
        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s incidence for each triple
        # k4_incidence[i] = list of K4 indices that contain triple i
        # k5_incidence[i] = list of K5 indices that contain triple i
        
        k4s = []
        for quad in combinations(range(n), 4):
            k4s.append([triple_index[tri] for tri in combinations(quad, 3)])
        num_k4s = len(k4s)
        
        k5s = []
        for quint in combinations(range(n), 5):
            k5s.append([triple_index[tri] for tri in combinations(quint, 3)])
        num_k5s = len(k5s)
        
        # Build incidence lists
        k4_incidence = [[] for _ in range(num_triples)]
        for k4_idx, edges in enumerate(k4s):
            for e in edges:
                k4_incidence[e].append(k4_idx)
                
        k5_incidence = [[] for _ in range(num_triples)]
        for k5_idx, edges in enumerate(k5s):
            for e in edges:
                k5_incidence[e].append(k5_idx)
        
        # Initialize random coloring
        # colors[i] is 0 (red) or 1 (blue)
        # We want to avoid all-red K4s and all-blue K5s.
        # Let's start with a random assignment.
        
        best_violations = float('inf')
        best_colors = None
        
        start = time.time()
        
        # We will use simulated annealing or simple hill climbing with restarts.
        # Given the time limit, simple hill climbing with random restarts might be better.
        
        # For each restart:
        # 1. Random init
        # 2. Hill climb until local minimum
        # 3. If 0 violations, return
        
        # To speed up violation counting, we maintain:
        # red_counts[k4_idx] = number of red edges in K4
        # blue_counts[k5_idx] = number of blue edges in K5
        
        # Violation count = sum(red_counts == 4) + sum(blue_counts == 10)
        
        # We can update these incrementally.
        
        max_restarts = 1000 # We'll break based on time
        
        for _ in range(max_restarts):
            if time.time() - start > time_limit:
                break
                
            # Random init
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Init counts
            red_counts = np.zeros(num_k4s, dtype=np.int32)
            blue_counts = np.zeros(num_k5s, dtype=np.int32)
            
            # Calculate initial counts
            for i in range(num_triples):
                if colors[i] == 0:
                    red_counts[k4_incidence[i]] += 1
                else:
                    blue_counts[k5_incidence[i]] += 1
            
            current_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            
            if current_violations == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            best_in_run = current_violations
            best_colors_in_run = colors.copy()
            
            # Hill climbing
            while time.time() - start < time_limit:
                # Pick random triple to flip
                idx = np.random.randint(0, num_triples)
                old_val = colors[idx]
                
                # Tentative flip
                if old_val == 0:
                    # 0 -> 1
                    # Red count decreases for K4s containing idx
                    red_counts[k4_incidence[idx]] -= 1
                    # Blue count increases for K5s containing idx
                    blue_counts[k5_incidence[idx]] += 1
                else:
                    # 1 -> 0
                    # Red count increases for K4s containing idx
                    red_counts[k4_incidence[idx]] += 1
                    # Blue count decreases for K5s containing idx
                    blue_counts[k5_incidence[idx]] -= 1
                
                colors[idx] = 1 - old_val
                
                new_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    if current_violations < best_in_run:
                        best_in_run = current_violations
                        best_colors_in_run = colors.copy()
                        if current_violations == 0:
                            return {'n': n, 'colors': colors.tolist()}
                else:
                    # Revert
                    if old_val == 0:
                        red_counts[k4_incidence[idx]] += 1
                        blue_counts[k5_incidence[idx]] -= 1
                    else:
                        red_counts[k4_incidence[idx]] -= 1
                        blue_counts[k5_incidence[idx]] += 1
                    colors[idx] = old_val
            
            if best_in_run < best_violations:
                best_violations = best_in_run
                best_colors = best_colors_in_run
        
        if best_colors is not None and best_violations == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        
        return None
    
    # Try n=34 first
    # We have 145s total. Let's give n=34 120s.
    time_for_34 = min(120.0, deadline - time.time())
    if time_for_34 > 1.0:
        sol = try_find_coloring(34, time_for_34)
        if sol:
            return sol
    
    # If not found, try smaller n with remaining time
    remaining = deadline - time.time()
    if remaining > 1.0:
        # Try n=33, 32, ...
        # Allocate time proportionally or fixed small chunks
        # Let's try n=33 for 10s, n=32 for 5s, etc.
        for n in range(33, 9, -1):
            if time.time() > deadline:
                break
            t_limit = min(10.0, deadline - time.time())
            if t_limit < 1.0:
                break
            sol = try_find_coloring(n, t_limit)
            if sol:
                return sol
    
    # Fallback: return a known valid coloring for small n
    # n=9 is known to be valid (trivially, as R(4,5;3) > 9 is easy)
    # Let's construct one for n=9.
    # Actually, we can just return n=1 with empty colors, but we want max n.
    # The task says "Maximize n". If we fail to find n=34, we should return the best we found.
    # But if we find nothing, we must return a valid coloring.
    # A trivial valid coloring for n=9:
    # Colors: 0 if (a+b+c)%3==0 else 1.
    # Let's verify this doesn't have red K4 or blue K5 for n=9.
    # It's a standard construction for small Ramsey numbers.
    
    n_fallback = 9
    triples = list(combinations(range(n_fallback), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n_fallback, 'colors': colors}

# Helper to verify if a coloring is valid (for debugging, not used in final return)
def is_valid(n, colors):
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    colors = np.array(colors)
    
    for quad in combinations(range(n), 4):
        if all(colors[triple_index[tri]] == 0 for tri in combinations(quad, 3)):
            return False
    for quint in combinations(range(n), 5):
        if all(colors[triple_index[tri]] == 1 for tri in combinations(quint, 3)):
            return False
    return True

# Main entry point
if __name__ == "__main__":
    # This block is for local testing
    result = construct()
    print(f"Found coloring for n={result['n']}")
    # print(is_valid(result['n'], result['colors']))
