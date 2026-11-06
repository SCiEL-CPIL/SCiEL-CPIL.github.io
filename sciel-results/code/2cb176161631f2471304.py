import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 40
    target_edges = 401
    
    def get_books(A):
        # A is 0/1 matrix
        # Common neighbors
        B = np.dot(A, A)
        # Book size is B[i,j] for edges A[i,j]=1
        # We need max of B where A=1
        # Mask out non-edges
        # Note: B[i,i] is degree, we don't care about diagonal
        # Create a matrix of book sizes, 0 for non-edges
        book_sizes = B * A
        # Find max
        max_book = np.max(book_sizes)
        return max_book, book_sizes

    def score_graph(A):
        max_book, book_sizes = get_books(A)
        # Count how many edges have max_book
        count_max = np.sum(book_sizes == max_book)
        # Sum of all book sizes (secondary metric)
        sum_books = np.sum(book_sizes)
        return max_book, count_max, sum_books

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    best_score = None
    best_A = None
    
    for trial in range(100): # Multiple restarts
        if time.time() > deadline - 5:
            break
            
        # Random Init
        random.shuffle(all_pairs)
        edges = all_pairs[:target_edges]
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        cur_max, cur_cnt, cur_sum = score_graph(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
            
        if cur_max <= 7:
            break
            
        # Local Search
        stuck = 0
        while time.time() < deadline - 5:
            # Find max edges
            _, book_sizes = get_books(A)
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and book_sizes[i, j] == cur_max:
                        max_edges.append((i, j))
                        
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            # Try swaps
            # We want to remove (u,v) and add (x,y)
            # To be fast, we can pre-filter (x,y)
            # (x,y) should not create a book size > cur_max (ideally < cur_max)
            # Actually, if we remove (u,v), the max might drop.
            
            improved = False
            # Sample random non-edges
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            random.shuffle(non_edges)
            
            # Try first 100 random non-edges
            for x, y in non_edges[:100]:
                # Perform swap
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                new_max, new_cnt, new_sum = score_graph(A)
                new_score = (new_max, new_cnt, new_sum)
                
                if new_score < cur_score:
                    cur_max, cur_cnt, cur_sum = new_score
                    cur_score = new_score
                    improved = True
                    if best_score is None or cur_score < best_score:
                        best_score = cur_score
                        best_A = A.copy()
                    break
                else:
                    # Revert
                    A[u, v] = 1
                    A[v, u] = 1
                    A[x, y] = 0
                    A[y, x] = 0
                    
            if improved:
                stuck = 0
                if cur_max <= 7:
                    break
            else:
                stuck += 1
                if stuck > 50:
                    break
                    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        # Fix count
        existing = set(tuple(e) for e in edges)
        if len(edges) > target_edges:
            # Remove random ones
            while len(edges) > target_edges:
                idx = random.randint(0, len(edges)-1)
                e = edges.pop(idx)
                existing.discard(tuple(e))
                existing.discard(tuple(reversed(e)))
        else:
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
                    
    return edges
