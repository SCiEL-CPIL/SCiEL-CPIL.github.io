"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave margin before 150s kill
    
    n = 40
    m = 401
    
    # Helper: compute book size for each edge
    def compute_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            # Count common neighbors
            common = 0
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[w][v]:
                        common += 1
            books[(u, v)] = common
            books[(v, u)] = common
        return books
    
    # Helper: get max book size
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    # Start with K20,20 plus one internal edge
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    
    for a in range(20):
        for b in range(20, 40):
            adj[a][b] = True
            adj[b][a] = True
            edge_set.add((a, b))
            edge_set.add((b, a))
    
    # Add one internal edge
    adj[0][1] = True
    adj[1][0] = True
    edge_set.add((0, 1))
    edge_set.add((1, 0))
    
    books = compute_books(adj, [(u, v) for u, v in edge_set if u < v])
    current_max = max_book(books)
    
    best_edges = set(edge_set)
    best_max = current_max
    
    # Local search: try to improve by swapping edges
    # Strategy: find edges with high book size, try removing them and adding low-book edges
    
    random.seed(42)
    
    # Build list of non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        iteration += 1
        
        # Find edges with high book size
        high_book_edges = [(u, v) for u, v in edge_set if u < v and books.get((u, v), 0) >= current_max]
        
        if not high_book_edges:
            break
        
        # Pick a random high-book edge to remove
        rem_u, rem_v = random.choice(high_book_edges)
        
        # Find candidate edges to add (non-edges)
        # Prefer non-edges that would have low book size if added
        candidates = []
        for u, v in all_pairs:
            if (u, v) not in edge_set and (v, u) not in edge_set:
                # Estimate book size if we add this edge
                # It would be the number of common neighbors
                common = 0
                for w in range(n):
                    if w != u and w != v:
                        if adj[u][w] and adj[w][v]:
                            common += 1
                candidates.append((common, u, v))
        
        if not candidates:
            no_improve += 1
            continue
        
        # Sort by estimated book size, try low ones first
        candidates.sort()
        
        improved = False
        for est_book, add_u, add_v in candidates[:50]:  # Try top 50 candidates
            if est_book >= current_max:
                break
            
            # Try the swap: remove (rem_u, rem_v), add (add_u, add_v)
            # First remove
            adj[rem_u][rem_v] = False
            adj[rem_v][rem_u] = False
            edge_set.discard((rem_u, rem_v))
            edge_set.discard((rem_v, rem_u))
            
            # Then add
            adj[add_u][add_v] = True
            adj[add_v][add_u] = True
            edge_set.add((add_u, add_v))
            edge_set.add((add_v, add_u))
            
            # Recompute books
            edge_list = [(u, v) for u, v in edge_set if u < v]
            books = compute_books(adj, edge_list)
            new_max = max_book(books)
            
            if new_max < current_max:
                current_max = new_max
                best_edges = set(edge_set)
                best_max = new_max
                no_improve = 0
                improved = True
                break
            else:
                # Revert
                adj[add_u][add_v] = False
                adj[add_v][add_u] = False
                edge_set.discard((add_u, add_v))
                edge_set.discard((add_v, add_u))
                
                adj[rem_u][rem_v] = True
                adj[rem_v][rem_u] = True
                edge_set.add((rem_u, rem_v))
                edge_set.add((rem_v, rem_u))
        
        if not improved:
            no_improve += 1
        
        # Occasionally do a random restart if stuck
        if no_improve > 2000 and random.random() < 0.1:
            # Try a different starting configuration
            adj = [[False] * n for _ in range(n)]
            edge_set = set()
            
            # Random balanced graph
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        if len(edge_set) // 2 < m:
                            adj[i][j] = True
                            adj[j][i] = True
                            edge_set.add((i, j))
                            edge_set.add((j, i))
            
            # Adjust to have exactly m edges
            while len(edge_set) // 2 > m:
                e = random.choice(list(edge_set))
                if e[0] < e[1]:
                    u, v = e
                else:
                    u, v = e[1], e[0]
                adj[u][v] = False
                adj[v][u] = False
                edge_set.discard((u, v))
                edge_set.discard((v, u))
            
            while len(edge_set) // 2 < m:
                u, v = random.choice(all_pairs)
                if (u, v) not in edge_set:
                    adj[u][v] = True
                    adj[v][u] = True
                    edge_set.add((u, v))
                    edge_set.add((v, u))
            
            edge_list = [(u, v) for u, v in edge_set if u < v]
            books = compute_books(adj, edge_list)
            current_max = max_book(books)
            if current_max < best_max:
                best_max = current_max
                best_edges = set(edge_set)
            no_improve = 0
    
    # Return the best result
    # Convert to list of [u, v] pairs
    result = [[u, v] for u, v in best_edges if u < v]
    
    # Ensure exactly m edges
    assert len(result) == m, f"Expected {m} edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
