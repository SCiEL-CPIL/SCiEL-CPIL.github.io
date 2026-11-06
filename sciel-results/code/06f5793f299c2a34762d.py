import numpy as np
import random
import time

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110  # Safety margin
    
    # Initialize with a random graph
    # Or start with K20,20 + random internal edges to be close to dense
    # Let's start with K20,20 and add 1 random internal edge, then search
    # Actually, random start is often better for local search to avoid local minima
    # But K20,20 is a good baseline.
    
    # Let's try starting with a random graph with m edges
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    def get_book_sizes(A):
        # A^2 gives common neighbors
        # We only care about entries where A[u,v] == 1
        A2 = A @ A
        # Mask non-edges
        # Diagonal is degree, we don't care about it for books
        # We want max of A2[u,v] where A[u,v] == 1
        # A2 is symmetric
        mask = (A == 1)
        # Set non-edges to 0
        A2_masked = A2 * mask
        # The max book size is the max value in A2_masked
        # Note: A2[u,v] is integer
        return np.max(A2_masked)
    
    current_books = get_book_sizes(A)
    best_books = current_books
    best_A = A.copy()
    
    # Local Search
    while time.time() < deadline:
        # Pick a random edge to remove
        edge_list = list(edges)
        if not edge_list:
            break
        u, v = random.choice(edge_list)
        
        # Try adding a random non-edge
        # To be efficient, pick a few random candidates for (x, y)
        improved = False
        for _ in range(20): # Try 20 random swaps per removal
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if (min(x, y), max(x, y)) in edges:
                continue
            # Also ensure we don't create a duplicate of the removed edge
            if (min(x, y), max(x, y)) == (u, v):
                continue
                
            # Perform swap
            edges.remove((u, v))
            A[u, v] = 0
            A[v, u] = 0
            
            nx, ny = min(x, y), max(x, y)
            edges.add((nx, ny))
            A[nx, ny] = 1
            A[ny, nx] = 1
            
            new_books = get_book_sizes(A)
            
            if new_books < current_books:
                current_books = new_books
                if current_books < best_books:
                    best_books = current_books
                    best_A = A.copy()
                improved = True
                break
            else:
                # Revert
                edges.remove((nx, ny))
                A[nx, ny] = 0
                A[ny, nx] = 0
                
                edges.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
        
        # If no improvement found in this batch, we could try a different strategy
        # But simple hill climbing might get stuck.
        # Let's add some randomness: accept worse moves with small probability (Simulated Annealing)
        # But for now, let's stick to strict improvement to ensure we don't degrade.
        # If stuck, restart?
        # Given 40 vertices, strict hill climbing from random start should find 7 or 8 quickly.
        
    # Reconstruct edges from best_A
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_A[u, v] == 1:
                final_edges.append([u, v])
                
    return final_edges
