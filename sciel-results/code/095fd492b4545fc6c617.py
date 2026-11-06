import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a, b) > 1 
    for all distinct pairs. Maximizes cardinality.
    
    Strategy:
    1. Start with all multiples of 3 (size 85) as base solution.
    2. Build graph of candidates (numbers sharing factor with 255).
    3. Use improved Bron-Kerbosch with better pruning and local search.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Base solution: all multiples of 3 in [1, 255]
    base_solution = list(range(3, 256, 3))
    best_clique_size = len(base_solution)  # 85
    best_clique = set(base_solution)
    
    # Build candidate set: all numbers in [1, 255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Bron-Kerbosch with pivot and improved pruning
    best_clique_indices = set()
    for val in base_solution:
        best_clique_indices.add(candidates.index(val))
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_indices
        
        if time.time() > deadline:
            return
        
        # Pruning
        if len(R) + len(P) <= best_clique_size:
            return
        
        if len(R) > best_clique_size:
            best_clique_size = len(R)
            best_clique_indices = set(R)
        
        if not P:
            return
        
        # Choose pivot to maximize |P ∩ N(u)|
        pivot_set = P | X
        if not pivot_set:
            pivot_set = P
        
        best_u = -1
        best_count = -1
        for u in pivot_set:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_u = u
        
        candidates_iter = P - adj[best_u] if best_u != -1 else P
        
        for v in list(candidates_iter):
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    # Try multiple starting points
    # Start 1: R = {255}
    R = {idx_255}
    P = adj[idx_255].copy()
    X = set()
    bron_kerbosch(R, P, X)
    
    # Local search: try to add elements to the best clique found
    if best_clique_indices and time.time() < deadline:
        current_clique = set(best_clique_indices)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate not in clique
            for v in range(n):
                if v in current_clique:
                    continue
                if time.time() > deadline:
                    break
                # Check if v is adjacent to all in current_clique
                if all(u in adj[v] for u in current_clique):
                    current_clique.add(v)
                    improved = True
                    break
        
        if len(current_clique) > best_clique_size:
            best_clique_size = len(current_clique)
            best_clique_indices = set(current_clique)
    
    # Convert back to values
    if best_clique_indices:
        result = sorted([candidates[i] for i in best_clique_indices])
        if 255 in result:
            return result
    
    return base_solution
# EVOLVE-BLOCK-END
