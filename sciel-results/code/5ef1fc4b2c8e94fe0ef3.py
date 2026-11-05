import random
import time
import numpy as np

def creates_c6(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    neighbors_u = adj[u]
    neighbors_v = adj[v]
    
    for a in neighbors_u:
        if a == v:
            continue
        for b in adj[a]:
            if b == u or b == a:
                continue
            for c in adj[b]:
                if c == u or c == a or c == b:
                    continue
                for d in adj[c]:
                    if d == u or d == a or d == b or d == c:
                        continue
                    if d in neighbors_v:
                        return True
    return False

def construct():
    n = 40
    start_time = time.time()
    timeout = 115
    
    # Initialize adjacency as list of sets for fast lookup
    adj = [set() for _ in range(n)]
    
    # Helper to sync numpy matrix
    def get_matrix():
        A = np.zeros((n, n), dtype=float)
        for u in range(n):
            for v in adj[u]:
                A[u, v] = 1.0
        return A

    # Helper to add edge
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    # Helper to remove edge
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    # 1. Initialize with a random C6-free graph
    # We want to start reasonably dense.
    # Let's try to add random edges until we hit a limit or fail.
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Random shuffle
    random.shuffle(all_edges)
    
    # Try to add ~90 edges initially
    init_target = 90
    added = 0
    for u, v in all_edges:
        if added >= init_target:
            break
        if not creates_c6(u, v, adj):
            add_edge(u, v)
            added += 1
            
    current_m = added
    
    # Pre-compute non-edges list for efficiency? 
    # It changes as we add/remove.
    
    best_m = current_m
    best_adj = [set(s) for s in adj] # Deep copy
    
    while time.time() - start_time < timeout:
        # Compute spectral properties
        A = get_matrix()
        try:
            # eigvalsh returns eigenvalues in ascending order
            # We want the largest, so index -1
            # eigh returns eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(A)
            lambda_1 = eigvals[-1]
            x = eigvecs[:, -1] # Perron vector
        except:
            break
            
        # Generate candidate non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    # Score: x[u] * x[v]
                    # Smaller score means less increase in lambda_1
                    score = x[u] * x[v]
                    candidates.append((score, u, v))
                    
        # Sort by score ascending
        candidates.sort(key=lambda t: t[0])
        
        added_any = False
        # Try to add the top candidates
        # Limit number of attempts per iteration to save time
        for score, u, v in candidates[:50]: 
            if time.time() - start_time > timeout:
                break
            if creates_c6(u, v, adj):
                continue
            # Add edge
            add_edge(u, v)
            current_m += 1
            added_any = True
            # Update best if improved
            if current_m > best_m:
                best_m = current_m
                best_adj = [set(s) for s in adj]
            # After adding one, the spectral properties change.
            # For simplicity in this loop, we might just try a few and then recompute.
            # But recomputing eig is expensive (O(n^3) ~ 64k ops, very fast for n=40).
            # Let's recompute after each successful add to keep the heuristic fresh?
            # Or just break after one add to keep the loop fast?
            # Let's try adding up to 5 edges per spectral update.
            if added_any and random.random() < 0.5:
                 break # Randomly stop early to vary behavior
            
        if not added_any:
            # Local minimum. Try a random swap.
            # Remove a random edge
            if current_m > 0:
                edges_to_remove = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if edges_to_remove:
                    u, v = random.choice(edges_to_remove)
                    remove_edge(u, v)
                    current_m -= 1
                    
                    # Try to add 1 or 2 new edges
                    # Recompute spectral
                    A = get_matrix()
                    try:
                        eigvals, eigvecs = np.linalg.eigh(A)
                        x = eigvecs[:, -1]
                    except:
                        continue
                        
                    candidates = []
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if v2 not in adj[u2]:
                                score = x[u2] * x[v2]
                                candidates.append((score, u2, v2))
                    candidates.sort(key=lambda t: t[0])
                    
                    for score, u2, v2 in candidates[:20]:
                        if creates_c6(u2, v2, adj):
                            continue
                        add_edge(u2, v2)
                        current_m += 1
                        if current_m > best_m:
                            best_m = current_m
                            best_adj = [set(s) for s in adj]
                        # Maybe try to add another?
                        if random.random() < 0.3:
                             break

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    return best_edges
