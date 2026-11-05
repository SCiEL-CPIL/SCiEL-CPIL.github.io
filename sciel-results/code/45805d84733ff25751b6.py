"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book.
    Approach: Spectral optimization with eigenvalue-guided local search.
    1. Initialize with a near-regular graph (balanced degrees).
    2. Use eigenvalue spectrum to guide edge swaps that flatten A^2 on edges.
    3. Track the eigenvalue gap to identify swaps that improve spectral regularity.
    """
    n = 40
    target_edges = 401
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Initialize with a near-regular graph
    # Average degree = 2*401/40 = 20.05, so aim for degrees 20 or 21
    # Create a base regular-like structure
    
    # Start with a random graph but bias toward regularity
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Build initial graph with balanced degrees
    adj = np.zeros((n, n), dtype=np.int32)
    degrees = np.zeros(n, dtype=np.int32)
    
    # First, create a 20-regular base (40*20/2 = 400 edges)
    # Use a greedy approach to build near-regular
    remaining = target_edges
    vertices = list(range(n))
    
    # Shuffle for randomness
    random.shuffle(vertices)
    
    # Greedy: add edges to keep degrees balanced
    edge_set = set()
    for _ in range(target_edges):
        # Find pair with lowest degree sum that isn't already connected
        best_pair = None
        best_sum = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    s = degrees[i] + degrees[j]
                    if s < best_sum:
                        best_sum = s
                        best_pair = (i, j)
        if best_pair is None:
            # Fallback: random
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
            if candidates:
                best_pair = random.choice(candidates)
            else:
                break
        u, v = best_pair
        edge_set.add((u, v))
        adj[u, v] = 1
        adj[v, u] = 1
        degrees[u] += 1
        degrees[v] += 1
        remaining -= 1
    
    # Helper: compute max book and spectrum info
    def get_spectrum_info(adj):
        A2 = adj @ adj
        # Max book on edges
        max_book = 0
        edge_books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    b = int(A2[u, v])
                    edge_books[(u, v)] = b
                    if b > max_book:
                        max_book = b
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)[::-1]  # descending
        lambda_max = eigvals[0]
        lambda_2 = eigvals[1] if len(eigvals) > 1 else 0
        gap = lambda_max - lambda_2
        return max_book, edge_books, lambda_max, lambda_2, gap
    
    max_book, edge_books, lam1, lam2, gap = get_spectrum_info(adj)
    best_adj = adj.copy()
    best_max_book = max_book
    best_gap = gap
    
    # Local search with spectral guidance
    max_iterations = 3000
    for iteration in range(max_iterations):
        if max_book <= 7:
            if max_book == 6:
                break
        
        A2 = adj @ adj
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)[::-1]
        lam1 = eigvals[0]
        lam2 = eigvals[1] if len(eigvals) > 1 else 0
        current_gap = lam1 - lam2
        
        # Find high-book edges
        high_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    b = int(A2[u, v])
                    if b >= max_book:
                        high_edges.append((u, v, b))
        
        if not high_edges:
            break
        
        # Sort by book size descending, then by degree imbalance
        high_edges.sort(key=lambda x: (-x[2], abs(degrees[x[0]] - degrees[x[1]])))
        
        improved = False
        random.shuffle(high_edges)
        
        for (u, v, book_uv) in high_edges[:5]:
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            A2_temp = adj @ adj
            
            # Find candidate non-edges
            candidates = []
            for w in range(n):
                for x in range(w+1, n):
                    if not adj[w, x]:
                        # Potential book size if we add (w,x)
                        new_book = int(A2_temp[w, x])
                        # Also consider degree balance
                        deg_imbalance = abs(degrees[w] + 1 - (degrees[x] + 1))
                        # Penalize if this would create a very high book
                        if new_book < max_book:
                            # Score: prefer low book, good degree balance, and spectral improvement
                            score = new_book * 10 + deg_imbalance
                            candidates.append((w, x, new_book, score))
            
            if not candidates:
                adj[u, v] = 1
                adj[v, u] = 1
                continue
            
            candidates.sort(key=lambda t: t[3])
            
            best_candidate = None
            best_new_max = max_book + 100
            best_candidate_gap = -float('inf')
            
            # Try top 30 candidates
            for w, x, new_book, score in candidates[:30]:
                adj[w, x] = 1
                adj[x, w] = 1
                degrees[w] += 1
                degrees[x] += 1
                
                # Quick check: what's the new max book?
                A2_check = adj @ adj
                new_max = 0
                for a in range(n):
                    for b in range(a+1, n):
                        if adj[a, b]:
                            bk = int(A2_check[a, b])
                            if bk > new_max:
                                new_max = bk
                
                # Compute eigenvalue gap
                eig_check = np.linalg.eigvalsh(adj)
                eig_check = np.sort(eig_check)[::-1]
                new_gap = eig_check[0] - (eig_check[1] if len(eig_check) > 1 else 0)
                
                # Undo
                adj[w, x] = 0
                adj[x, w] = 0
                degrees[w] -= 1
                degrees[x] -= 1
                
                # Evaluate: prefer lower max_book, then better gap
                if new_max < best_new_max or (new_max == best_new_max and new_gap > best_candidate_gap):
                    best_new_max = new_max
                    best_candidate_gap = new_gap
                    best_candidate = (w, x)
            
            # Restore (u,v)
            adj[u, v] = 1
            adj[v, u] = 1
            
            if best_candidate and best_new_max < max_book:
                # Apply swap
                adj[u, v] = 0
                adj[v, u] = 0
                w, x = best_candidate
                adj[w, x] = 1
                adj[x, w] = 1
                degrees[u] -= 1
                degrees[v] -= 1
                degrees[w] += 1
                degrees[x] += 1
                max_book = best_new_max
                improved = True
                if max_book < best_max_book:
                    best_max_book = max_book
                    best_adj = adj.copy()
                break
        
        if not improved:
            # Random perturbation every 30 iterations
            if iteration % 30 == 29:
                edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                if edge_list and non_edges:
                    u, v = random.choice(edge_list)
                    w, x = random.choice(non_edges)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[w, x] = 1
                    adj[x, w] = 1
                    degrees[u] -= 1
                    degrees[v] -= 1
                    degrees[w] += 1
                    degrees[x] += 1
                    
                    mb, _, _, _, g = get_spectrum_info(adj)
                    if mb < best_max_book:
                        best_max_book = mb
                        best_adj = adj.copy()
    
    adj = best_adj
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
                
    if len(result) != target_edges:
        # Fallback: fix edge count
        if len(result) < target_edges:
            existing = set(tuple(sorted(e)) for e in result)
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in existing:
                        result.append([i, j])
                        existing.add((i, j))
                        if len(result) == target_edges:
                            break
                if len(result) == target_edges:
                    break
        elif len(result) > target_edges:
            result = result[:target_edges]
            
    return result
# EVOLVE-BLOCK-END
