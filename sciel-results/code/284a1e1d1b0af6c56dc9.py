import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_set = set(all_pairs)
    
    def calc_all_books_fast(A):
        """
        Fast computation of book sizes using matrix operations.
        Book size of edge (u,v) = number of common neighbors = (A @ A)[u,v]
        but we need to be careful: (A@A)[u,v] counts paths of length 2,
        which for symmetric A with 0 diagonal equals the number of common neighbors.
        Returns (max_book, count_max, sum_books) considering only edges.
        """
        # A is 0/1 symmetric matrix with 0 diagonal
        # (A @ A)[i][j] = sum_k A[i][k] * A[k][j] = number of common neighbors of i and j
        AA = A @ A
        # Extract book sizes for edges only
        edge_books = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edge_books[i, j] = AA[i, j]
                    edge_books[j, i] = AA[i, j]
        
        # Get only upper triangle edge values
        edge_vals = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edge_vals.append(int(AA[i, j]))
        
        if not edge_vals:
            return 0, 0, 0
        
        max_b = max(edge_vals)
        count_max = sum(1 for b in edge_vals if b == max_b)
        sum_books = sum(edge_vals)
        return max_b, count_max, sum_books
    
    def calc_books_vectorized(A):
        """
        Even faster: compute all book sizes at once using numpy.
        Returns a 2D array of book sizes and a list of (edge, book) for edges.
        """
        AA = A @ A
        # For each edge (i,j), book size = AA[i,j]
        edge_books_list = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edge_books_list.append((i, j, int(AA[i, j])))
        
        if not edge_books_list:
            return 0, 0, 0, []
        
        books = [b for _, _, b in edge_books_list]
        max_b = max(books)
        count_max = sum(1 for b in books if b == max_b)
        sum_books = sum(books)
        return max_b, count_max, sum_books, edge_books_list
    
    def score_tuple(A):
        """Compute the score tuple (max_book, count_max, sum_books)."""
        max_b, count_max, sum_books, _ = calc_books_vectorized(A)
        return (max_b, count_max, sum_books)
    
    # Initialize with multiple random trials
    best_score = None
    best_A = None
    
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        
        # Create a random initial graph
        random.shuffle(all_pairs)
        initial_edges = all_pairs[:target_edges]
        
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in initial_edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Simulated annealing local search
        cur_max, cur_cnt, cur_sum, _ = calc_books_vectorized(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
        
        # Simulated annealing parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        iterations = 0
        stuck = 0
        
        while time.time() < deadline - 5 and T > T_min:
            iterations += 1
            
            # Get current book info
            cur_max, cur_cnt, cur_sum, edge_books_list = calc_books_vectorized(A)
            
            if cur_max == 0:
                break
            
            # Find edges with max book size
            max_edges = [(i, j) for i, j, b in edge_books_list if b == cur_max]
            
            if not max_edges:
                break
            
            # Choose a max-book edge to potentially remove
            u, v = random.choice(max_edges)
            
            # Find candidate non-edges to add
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            
            if not non_edges:
                break
            
            # Try multiple swaps
            random.shuffle(non_edges)
            improved = False
            
            # Number of swap attempts per iteration - scale with temperature
            num_attempts = min(len(non_edges), max(20, int(100 * T / 10.0)))
            
            for x, y in non_edges[:num_attempts]:
                # Perform swap
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                new_max, new_cnt, new_sum, _ = calc_books_vectorized(A)
                new_score = (new_max, new_cnt, new_sum)
                
                # Compute delta
                if cur_score[0] != new_score[0]:
                    delta = new_score[0] - cur_score[0]
                elif cur_score[1] != new_score[1]:
                    delta = (new_score[1] - cur_score[1]) / 1000.0
                else:
                    delta = (new_score[2] - cur_score[2]) / 100000.0
                
                # Accept if improved or with probability exp(-delta/T)
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / max(T, 1e-10))
                
                if accept:
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
            else:
                stuck += 1
                if stuck > 500:
                    break
            
            T *= cooling
        
        # If we've found score 7, we can stop early
        if best_score is not None and best_score[0] <= 7:
            break
    
    # Final polish: try to reduce max book further with more aggressive search
    if best_A is not None and best_score[0] > 7 and time.time() < deadline - 5:
        A = best_A.copy()
        cur_max, cur_cnt, cur_sum, _ = calc_books_vectorized(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        
        # More intensive local search with lower temperature
        T = 5.0
        T_min = 0.05
        cooling = 0.998
        stuck = 0
        
        while time.time() < deadline - 2 and T > T_min:
            cur_max, cur_cnt, cur_sum, edge_books_list = calc_books_vectorized(A)
            
            if cur_max == 0:
                break
            
            max_edges = [(i, j) for i, j, b in edge_books_list if b >= cur_max - 1]
            
            if not max_edges:
                break
            
            u, v = random.choice(max_edges)
            
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            
            if not non_edges:
                break
            
            random.shuffle(non_edges)
            improved = False
            
            for x, y in non_edges[:200]:
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                new_max, new_cnt, new_sum, _ = calc_books_vectorized(A)
                new_score = (new_max, new_cnt, new_sum)
                
                if cur_score[0] != new_score[0]:
                    delta = new_score[0] - cur_score[0]
                elif cur_score[1] != new_score[1]:
                    delta = (new_score[1] - cur_score[1]) / 1000.0
                else:
                    delta = (new_score[2] - cur_score[2]) / 100000.0
                
                if delta < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / max(T, 1e-10))
                
                if accept:
                    cur_max, cur_cnt, cur_sum = new_score
                    cur_score = new_score
                    improved = True
                    if best_score is None or cur_score < best_score:
                        best_score = cur_score
                        best_A = A.copy()
                    break
                else:
                    A[u, v] = 1
                    A[v, u] = 1
                    A[x, y] = 0
                    A[y, x] = 0
            
            if improved:
                stuck = 0
            else:
                stuck += 1
                if stuck > 300:
                    break
            
            T *= cooling
    
    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            existing = set(tuple(e) for e in edges)
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
    
    return edges
