import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    # Use numpy for fast matrix operations
    A = np.zeros((n, n), dtype=np.int32)
    
    def compute_books_numpy(A):
        """Compute book sizes using matrix multiplication."""
        A2 = A @ A
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    books[(i,j)] = A2[i,j]
        return books
    
    def get_max_book(books):
        return max(books.values()) if books else 0
    
    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())
    
    def regular_like_init(n, m):
        """
        Initialize with a random regular-like structure.
        Target degree ~ 2*m/n = 20.05, so aim for degree 20 or 21.
        Use a random pairing approach to create a regular graph.
        """
        A = np.zeros((n, n), dtype=np.int32)
        edge_set = set()
        
        # Target: 401 edges, 40 vertices
        # Average degree = 2*401/40 = 20.05
        # Create a 20-regular graph first (400 edges), then add 1 more edge
        
        # Generate a random 20-regular graph using configuration model
        # Each vertex needs degree 20
        # Total half-edges = 40*20 = 800
        # We'll create a random pairing
        
        # Build adjacency list for configuration model
        half_edges = []
        for v in range(n):
            for _ in range(20):
                half_edges.append(v)
        
        random.shuffle(half_edges)
        
        # Pair up half-edges
        edge_list = []
        i = 0
        while i < len(half_edges) - 1:
            u = half_edges[i]
            v = half_edges[i+1]
            i += 2
            if u != v and (u, v) not in edge_list and (v, u) not in edge_list:
                edge_list.append((u, v))
        
        # If we don't have exactly 400 edges, adjust
        # Add or remove edges to get to 401
        while len(edge_list) < 401:
            # Add a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (u, v) not in edge_list and (v, u) not in edge_list:
                edge_list.append((u, v))
        
        # If too many, remove some
        while len(edge_list) > 401:
            edge_list.pop(random.randint(0, len(edge_list)-1))
        
        # Build adjacency matrix
        for u, v in edge_list:
            A[u, v] = 1
            A[v, u] = 1
            edge_set.add((min(u,v), max(u,v)))
        
        return A, edge_set
    
    def spectral_local_search(A, edge_set, max_time=130.0):
        """
        Local search that minimizes the maximum book size.
        Uses numpy for fast book computation.
        """
        books = compute_books_numpy(A)
        current_max = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        best_max = current_max
        best_sum_sq = current_sum_sq
        best_A = A.copy()
        best_edge_set = set(edge_set)
        
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < max_time and T > T_min:
            # Find edges to consider for removal
            # Prefer high-book edges
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            
            if random.random() < 0.7:
                top_k = max(1, len(sorted_books) // 5)
                e_remove = random.choice(sorted_books[:top_k])[0]
            else:
                e_remove = random.choice(list(edge_set))
            
            u, v = e_remove
            
            # Find candidate edges to add
            # Sample and pick the one that minimizes the increase in max book
            candidates = []
            for _ in range(30):
                a, b = random.choice(all_possible)
                if (a, b) not in edge_set and (a, b) != e_remove:
                    candidates.append((a, b))
            
            if not candidates:
                T *= cooling
                continue
            
            # Evaluate each candidate by computing the new max book
            best_cand = None
            best_new_max = 100
            best_new_sum_sq = 1000000
            
            for a, b in candidates:
                # Tentatively apply the swap
                A[u, v] = 0
                A[v, u] = 0
                A[a, b] = 1
                A[b, a] = 1
                
                new_books = compute_books_numpy(A)
                new_max = get_max_book(new_books)
                new_sum_sq = get_sum_sq_books(new_books)
                
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[a, b] = 0
                A[b, a] = 0
                
                if new_max < best_new_max or (new_max == best_new_max and new_sum_sq < best_new_sum_sq):
                    best_new_max = new_max
                    best_new_sum_sq = new_sum_sq
                    best_cand = (a, b)
            
            if best_cand is None:
                T *= cooling
                continue
            
            a, b = best_cand
            
            # Apply the best candidate
            A[u, v] = 0
            A[v, u] = 0
            A[a, b] = 1
            A[b, a] = 1
            
            edge_set.remove(e_remove)
            edge_set.add((a, b))
            
            new_books = compute_books_numpy(A)
            new_max = get_max_book(new_books)
            new_sum_sq = get_sum_sq_books(new_books)
            
            # Acceptance criterion
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max:
                if new_sum_sq < current_sum_sq:
                    accept = True
                else:
                    if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 200)):
                        accept = True
            else:
                if random.random() < math.exp(-(new_max - current_max) / T):
                    accept = True
            
            if accept:
                current_max = new_max
                current_sum_sq = new_sum_sq
                books = new_books
                
                if current_max < best_max or (current_max == best_max and current_sum_sq < best_sum_sq):
                    best_max = current_max
                    best_sum_sq = current_sum_sq
                    best_A = A.copy()
                    best_edge_set = set(edge_set)
                    
                    if best_max <= 7:
                        return best_edge_set
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[a, b] = 0
                A[b, a] = 0
                
                edge_set.remove((a, b))
                edge_set.add(e_remove)
            
            T *= cooling
        
        return best_edge_set
    
    # Initialize with regular-like structure
    A, edge_set = regular_like_init(n, m)
    
    # Run spectral local search
    best_edge_set = spectral_local_search(A, edge_set, max_time=135.0)
    
    return [list(e) for e in best_edge_set]
