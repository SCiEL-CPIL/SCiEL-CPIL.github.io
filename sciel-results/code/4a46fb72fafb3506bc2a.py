import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for return/eval overhead
    
    # Precomputation for a given n is needed inside the loop as n changes
    # But we can just do it on the fly for each n attempt
    
    best_result = None
    best_n = 0
    
    # We try n from 34 down to 10
    # We allocate time dynamically
    # If we find a solution, we return it.
    # We must ensure we don't spend all time on n=34 and fail.
    
    # Strategy: 
    # Try n=34 for 2s.
    # Try n=33 for 2s.
    # ...
    # Try n=10 for remaining time.
    
    # Actually, a smarter approach:
    # Run local search for n=10. It will likely succeed in <1s.
    # Then try n=11, etc.
    # But if we find n=10, we can keep searching for n=11 in the background? No, sequential.
    
    # Let's try to find the max n by iterating upwards?
    # If we find n=10, can we extend to n=11?
    # Extension is hard.
    
    # Let's stick to trying specific n values.
    # Prioritize higher n but with short time limits.
    
    n_candidates = list(range(34, 9, -1))
    
    # We will try each n for a short duration.
    # If we find a valid coloring, we return it.
    # If we run out of time, we return the best valid one found (which might be none, so we fall back to seed).
    
    found_solution = None
    
    for n in n_candidates:
        if time.time() > deadline:
            break
            
        # Allocate time based on n
        # Higher n gets less time initially? Or fixed small time?
        # Let's give 1 second to n=34, 1 second to n=33, etc.
        # If we reach n=10, we have plenty of time left.
        
        time_limit_for_n = 1.0
        if n <= 15:
            time_limit_for_n = 2.0
        if n <= 10:
            time_limit_for_n = 10.0 # Give more time to the most likely success
            
        if time.time() + time_limit_for_n > deadline:
            time_limit_for_n = deadline - time.time()
            if time_limit_for_n < 0.1:
                break

        solution = local_search(n, time_limit_for_n)
        if solution is not None:
            found_solution = solution
            break # We found the largest n (since we iterate down)
            
    if found_solution is None:
        # Fallback to seed
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
        
    return found_solution

def local_search(n, time_limit):
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    # K4: 4 vertices, 4 triples
    k4s = []
    for quad in combinations(range(n), 4):
        edges = []
        for tri in combinations(quad, 3):
            edges.append(triple_index[tri])
        k4s.append(edges)
        
    # K5: 5 vertices, 10 triples
    k5s = []
    for quint in combinations(range(n), 5):
        edges = []
        for tri in combinations(quint, 3):
            edges.append(triple_index[tri])
        k5s.append(edges)
        
    # State: list of 0/1 for each triple
    # Colors: 0=Red, 1=Blue
    # Violation:
    # Red K4: all 4 edges are 0
    # Blue K5: all 10 edges are 1
    
    # We want to minimize violations.
    
    # Initialize randomly
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Count violations
    def count_violations(colors):
        v = 0
        for edges in k4s:
            if all(colors[e] == 0 for e in edges):
                v += 1
        for edges in k5s:
            if all(colors[e] == 1 for e in edges):
                v += 1
        return v
        
    current_violations = count_violations(colors)
    best_colors = colors[:]
    best_violations = current_violations
    
    start = time.time()
    while time.time() - start < time_limit:
        # Pick random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_val = colors[idx]
        new_val = 1 - old_val
        
        # Calculate delta violations
        # This requires checking all K4s and K5s containing idx
        # For efficiency, we can precompute which K4s/K5s contain each triple
        # But for n=34, num_triples is ~5984.
        # Number of K4s is 46k. Each K4 has 4 edges. Total edge-K4 incidences = 4*46k = 184k.
        # Average K4s per triple = 184k / 6k = 30.
        # Number of K5s is 278k. Each has 10 edges. Total = 2.7M.
        # Average K5s per triple = 2.7M / 6k = 450.
        
        # This is too slow to recalculate delta naively in Python for every flip.
        # We need a faster way or accept slower flips.
        # Let's just recalculate full violations if n is small, or use a heuristic.
        # For n=34, full recalc is 46k + 278k checks. ~300k ops.
        # In Python, 300k ops might take 10-20ms.
        # So we can do ~50-100 flips per second.
        # 1 second = ~100 flips.
        
        # Let's try flipping and checking full violations.
        colors[idx] = new_val
        new_violations = count_violations(colors)
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
        else:
            # Revert if worse, or keep if equal?
            # Simulated annealing: accept if random() < exp(-delta/T)
            # For simplicity, revert if worse.
            colors[idx] = old_val
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None
