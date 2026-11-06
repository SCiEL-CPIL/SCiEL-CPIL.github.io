import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 280  # 280s safety margin
    
    # Fallback: Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    fallback_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find a larger graph
    # We try n from 42 down to 38
    for n in range(42, 37, -1):
        if time.time() > deadline - 20:  # Need 20s for final processing/return
            break
        
        # Try to find a valid graph for this n
        edges = _search_graph(n, deadline)
        if edges is not None:
            return n, edges
            
    return q, fallback_edges

def _search_graph(n, deadline):
    # Try multiple random restarts with different strategies
    # Allocate time budget based on n
    
    # For n=42, spend most of the time; for smaller n, less
    time_budget = {42: 120, 41: 60, 40: 40, 39: 20, 38: 15}
    budget = time_budget.get(n, 10)
    start_time = time.time()
    
    max_restarts = 50
    max_steps = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > budget:
            return None
        if time.time() > deadline - 10:
            return None
            
        # Try different initialization strategies
        strategy = restart % 3
        
        if strategy == 0:
            # Random graph with p=0.5
            adj, edges = _init_random(n, 0.5)
        elif strategy == 1:
            # Start from Paley(37) and add vertices
            adj, edges = _init_from_paley(n)
        else:
            # Random graph with p=0.3 (sparser)
            adj, edges = _init_random(n, 0.3)
        
        # Repair loop with improved strategy
        for step in range(max_steps):
            if time.time() - start_time > budget:
                return None
            if time.time() > deadline - 5:
                return None
                
            violation = _find_violation_fast(n, adj)
            
            if violation is None:
                # Valid graph found
                return list(edges)
                
            kind, vertices = violation
            
            if kind == 'clique':
                # Remove edges from the clique - try all pairs, pick one that helps most
                removed = False
                for _ in range(3):  # Try a few random pairs
                    v1, v2 = random.sample(vertices, 2)
                    if v1 > v2: v1, v2 = v2, v1
                    if (v1, v2) in edges:
                        edges.remove((v1, v2))
                        adj[v1] &= ~(1 << v2)
                        adj[v2] &= ~(1 << v1)
                        removed = True
                        break
                if not removed:
                    # Fallback: remove any edge
                    v1, v2 = random.sample(vertices, 2)
                    if v1 > v2: v1, v2 = v2, v1
                    if (v1, v2) in edges:
                        edges.remove((v1, v2))
                        adj[v1] &= ~(1 << v2)
                        adj[v2] &= ~(1 << v1)
            elif kind == 'indep':
                # Add edges to the independent set
                added = False
                for _ in range(3):  # Try a few random pairs
                    v1, v2 = random.sample(vertices, 2)
                    if v1 > v2: v1, v2 = v2, v1
                    if (v1, v2) not in edges:
                        edges.add((v1, v2))
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
                        added = True
                        break
                if not added:
                    # Fallback: add any edge
                    v1, v2 = random.sample(vertices, 2)
                    if v1 > v2: v1, v2 = v2, v1
                    if (v1, v2) not in edges:
                        edges.add((v1, v2))
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
                    
        # If we exit the step loop, this restart failed
        continue
        
    return None

def _init_random(n, p):
    """Initialize a random graph with edge probability p"""
    adj = [0] * n
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((i, j))
    return adj, edges

def _init_from_paley(n):
    """Initialize from Paley(37) and add vertices with random connections"""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    adj = [0] * n
    edges = set()
    
    # Build Paley(37) part
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((i, j))
    
    # Add new vertices with random connections
    for i in range(q, n):
        for j in range(i):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((j, i))
    
    return adj, edges

def _find_violation_fast(n, adj):
    """Find a K5 or independent 5-set, returning quickly"""
    # Randomize vertex order
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # Check each vertex for K4 in neighbors or I4 in non-neighbors
    for v in vertices:
        # Get neighbors and non-neighbors
        neighbors = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
        non_neighbors = [u for u in range(n) if u != v and not ((adj[v] >> u) & 1)]
        
        # Check for K4 in neighbors (early exit)
        k4 = _find_k4_fast(neighbors, adj)
        if k4:
            return ('clique', (v,) + tuple(k4))
            
        # Check for I4 in non-neighbors (early exit)
        indep4 = _find_indep4_fast(non_neighbors, adj)
        if indep4:
            return ('indep', (v,) + tuple(indep4))
            
    return None

def _find_k4_fast(vertices, adj):
    """Find a clique of size 4, with early termination"""
    if len(vertices) < 4:
        return None
    
    # Try combinations but exit early
    for comb in combinations(vertices, 4):
        v0, v1, v2, v3 = comb
        if ((adj[v0] >> v1) & 1) and ((adj[v0] >> v2) & 1) and ((adj[v0] >> v3) & 1) and \
           ((adj[v1] >> v2) & 1) and ((adj[v1] >> v3) & 1) and \
           ((adj[v2] >> v3) & 1):
            return comb
    return None

def _find_indep4_fast(vertices, adj):
    """Find an independent set of size 4, with early termination"""
    if len(vertices) < 4:
        return None
    
    # Try combinations but exit early
    for comb in combinations(vertices, 4):
        v0, v1, v2, v3 = comb
        if not ((adj[v0] >> v1) & 1) and not ((adj[v0] >> v2) & 1) and not ((adj[v0] >> v3) & 1) and \
           not ((adj[v1] >> v2) & 1) and not ((adj[v1] >> v3) & 1) and \
           not ((adj[v2] >> v3) & 1):
            return comb
    return None
