import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0
    
    def compute_books(A):
        # A is n x n adjacency matrix
        # A2 = A @ A
        # A2[u, v] is number of common neighbors
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    def get_max_book(A2):
        return np.max(A2)

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                e = (min(u, v), max(u, v))
                edges.add(e)
        return edges

    def build_matrix(edges):
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    best_edges = None
    best_max_b = float('inf')
    
    # We will run multiple restarts
    # Each restart: Random init -> Greedy/Local Search -> SA
    num_restarts = 20 # Will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        edges = random_graph()
        A = build_matrix(edges)
        A2 = compute_books(A)
        max_b = get_max_book(A2)
        
        if max_b < best_max_b:
            best_max_b = max_b
            best_edges = set(edges)
            if best_max_b <= 7:
                # Theoretical lower bound reached
                return [[u, v] for u, v in best_edges]
        
        # Local Search Phase
        # Try to improve by swapping edges
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            A2 = compute_books(A)
            max_b = get_max_book(A2)
            
            if max_b <= 7:
                if max_b < best_max_b:
                    best_max_b = max_b
                    best_edges = set(edges)
                return [[u, v] for u, v in best_edges]
            
            # Find edge with max book
            # Get indices of max values
            max_val = max_b
            # We want to reduce this max_val
            # Find an edge (u,v) with book size max_val
            # np.where(A2 == max_val) gives all such edges
            candidates = np.where(A2 == max_val)
            if len(candidates[0]) == 0:
                break
            
            # Pick one candidate edge to "fix"
            # Randomly pick from candidates to avoid bias
            idx = random.randint(0, len(candidates[0]) - 1)
            u = candidates[0][idx]
            v = candidates[1][idx]
            
            # The common neighbors of u and v are the vertices w such that A[u,w]=1 and A[v,w]=1
            # We want to remove one edge (u,w) or (v,w) where w is a common neighbor
            # And add a different edge (x,y)
            
            # Get common neighbors
            common_neighbors = np.where((A[u] == 1) & (A[v] == 1))[0]
            
            # Try removing each edge (u, w) or (v, w)
            # And adding a random non-edge
            found_improvement = False
            
            # To speed up, don't try all w, just a few random ones
            random.shuffle(common_neighbors)
            for w in common_neighbors[:5]:
                # Option 1: Remove (u, w)
                # Option 2: Remove (v, w)
                
                for rem_u, rem_v in [(u, w), (v, w)]:
                    if rem_u == rem_v: continue
                    rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                    
                    # Try adding a random non-edge
                    # Sample a few random non-edges
                    for _ in range(10):
                        add_u = random.randint(0, n - 1)
                        add_v = random.randint(0, n - 1)
                        if add_u == add_v: continue
                        add_edge = (min(add_u, add_v), max(add_u, add_v))
                        
                        if add_edge in edges: continue
                        if add_edge == rem_edge: continue
                        
                        # Apply swap
                        A[rem_u, rem_v] = 0
                        A[rem_v, rem_u] = 0
                        A[add_u, add_v] = 1
                        A[add_v, add_u] = 1
                        
                        A2_new = compute_books(A)
                        max_new = get_max_book(A2_new)
                        
                        # Check if improvement
                        # We want max_new < max_b
                        # Or if max_new == max_b, check sum of squares or similar
                        # For now, just strict improvement in max
                        if max_new < max_b:
                            # Accept
                            edges.discard(rem_edge)
                            edges.add(add_edge)
                            max_b = max_new
                            improved = True
                            found_improvement = True
                            # Don't revert, keep the change
                            break
                        else:
                            # Revert
                            A[rem_u, rem_v] = 1
                            A[rem_v, rem_u] = 1
                            A[add_u, add_v] = 0
                            A[add_v, add_u] = 0
                    
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            if not found_improvement:
                # No improvement found in this iteration of local search
                # Try Simulated Annealing steps to escape
                T = 1.0
                for _ in range(50):
                    if time.time() > deadline - 10: break
                    
                    # Random swap
                    rem_edge = random.choice(list(edges))
                    rem_u, rem_v = rem_edge
                    
                    added = False
                    for _ in range(20):
                        add_u = random.randint(0, n - 1)
                        add_v = random.randint(0, n - 1)
                        if add_u == add_v: continue
                        add_edge = (min(add_u, add_v), max(add_u, add_v))
                        if add_edge not in edges:
                            added = True
                            break
                    if not added: continue
                    
                    A[rem_u, rem_v] = 0
                    A[rem_v, rem_u] = 0
                    A[add_u, add_v] = 1
                    A[add_v, add_u] = 1
                    
                    A2_new = compute_books(A)
                    max_new = get_max_book(A2_new)
                    
                    delta = max_new - max_b
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Accept
                        edges.discard(rem_edge)
                        edges.add(add_edge)
                        max_b = max_new
                        if max_b < best_max_b:
                            best_max_b = max_b
                            best_edges = set(edges)
                            if best_max_b <= 7:
                                return [[u, v] for u, v in best_edges]
                    else:
                        # Revert
                        A[rem_u, rem_v] = 1
                        A[rem_v, rem_u] = 1
                        A[add_u, add_v] = 0
                        A[add_v, add_u] = 0
                    T *= 0.99

    return [[u, v] for u, v in best_edges]
