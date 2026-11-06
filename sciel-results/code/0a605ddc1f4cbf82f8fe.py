import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Build initial graph
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def get_cliques_and_independents(adj, n):
        cliques = []
        independents = []
        # Find all K4s
        for combo in combinations(range(n), 4):
            is_clique = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                    break
            if is_clique:
                cliques.append(combo)
                
        # Find all independent sets of size 4
        for combo in combinations(range(n), 4):
            is_independent = True
            for u, v in combinations(combo, 2):
                if v in adj[u]:
                    is_independent = False
                    break
            if is_independent:
                independents.append(combo)
                
        return cliques, independents

    # Precompute for initial graph
    cliques, independents = get_cliques_and_independents(adj, n)
    
    # Try to add vertices with smarter heuristics
    attempts = 0
    consecutive_failures = 0
    max_consecutive_failures = 500
    
    while time.time() < deadline:
        attempts += 1
        
        # Generate candidate neighbor sets with different strategies
        candidate_neighbors = None
        
        # Strategy 1: Random with bias toward balanced degree
        if random.random() < 0.5:
            # Aim for approximately half the vertices as neighbors
            target_size = n // 2 + random.randint(-2, 2)
            candidates = list(range(n))
            random.shuffle(candidates)
            candidate_neighbors = set(candidates[:target_size])
        else:
            # Strategy 2: Greedy - avoid completing K4s and independent 4-sets
            candidate_neighbors = set()
            for i in range(n):
                # Check if adding i to neighbors would create a K4 with existing neighbors
                would_create_k4 = False
                for c in cliques:
                    if i in c and all(v in candidate_neighbors for v in c if v != i):
                        would_create_k4 = True
                        break
                # Check if not adding i would create an independent 4-set
                would_create_independent = False
                if not would_create_k4:
                    for ind in independents:
                        if i in ind and all(v not in candidate_neighbors for v in ind if v != i):
                            would_create_independent = True
                            break
                
                if not would_create_k4 and not would_create_independent:
                    # Add with some probability to maintain randomness
                    if random.random() < 0.5:
                        candidate_neighbors.add(i)
        
        if candidate_neighbors is None:
            candidate_neighbors = set()
        
        # Check validity
        valid = True
        # Check K4 in neighbors
        for c in cliques:
            if all(v in candidate_neighbors for v in c):
                valid = False
                break
        
        if valid:
            # Check independent set of size 4 in non-neighbors
            non_neighbors = set(range(n)) - candidate_neighbors
            for ind in independents:
                if all(v in non_neighbors for v in ind):
                    valid = False
                    break
        
        if valid:
            # Add vertex
            n += 1
            adj.append(set(candidate_neighbors))
            for i in candidate_neighbors:
                adj[i].add(n - 1)
            
            # Update cliques and independents
            cliques, independents = get_cliques_and_independents(adj, n)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            # If we've had many consecutive failures, try a different approach
            if consecutive_failures >= max_consecutive_failures:
                # Reset and try with more randomness
                consecutive_failures = 0
                max_consecutive_failures = int(max_consecutive_failures * 1.5)
    
    # Return the best graph found
    # We need to return (n, edges)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
