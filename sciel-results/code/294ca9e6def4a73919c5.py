import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 40
    target_edges = 401
    
    def calc_books(A):
        # A is n x n binary adjacency matrix
        # Book size for edge (u,v) = number of common neighbors = (A @ A)[u,v]
        M = A @ A
        masked = np.where(A > 0, M, 0)
        max_b = int(masked.max())
        count_max = int(np.sum(masked == max_b)) // 2
        sum_books = int(np.sum(masked)) // 2
        return max_b, count_max, sum_books, masked
    
    def incremental_swap_score(A, book_matrix, remove_u, remove_v, add_x, add_y):
        """
        Compute the new max book after removing edge (remove_u, remove_v) 
        and adding edge (add_x, add_y), using incremental updates.
        
        Only books involving vertices {remove_u, remove_v, add_x, add_y} change.
        """
        # Vertices whose books might change
        affected = set([remove_u, remove_v, add_x, add_y])
        
        # Temporarily apply the swap
        A[remove_u, remove_v] = 0
        A[remove_v, remove_u] = 0
        A[add_x, add_y] = 1
        A[add_y, add_x] = 1
        
        # Recompute books only for affected edges
        # An edge (i,j) is affected if i or j is in affected set
        new_max = 0
        new_count_max = 0
        new_sum = 0
        
        # We need to recompute books for all edges that have at least one endpoint in affected
        # But we also need to check if the global max might have been on an unaffected edge
        # that had the old max value. So we track the old max and whether it's still present.
        
        # Strategy: recompute books for all edges incident to affected vertices,
        # and keep the old book values for other edges.
        
        # Get old books for all edges
        old_books = book_matrix.copy()
        
        # For edges where both endpoints are in affected, or one is in affected,
        # recompute the book size
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 0:
                    continue
                if i in affected or j in affected:
                    # Recompute book size for edge (i,j)
                    common = np.sum(A[i, :] & A[j, :])
                    old_books[i, j] = common
                    old_books[j, i] = common
        
        # Now compute max, count, sum from updated books
        masked = np.where(A > 0, old_books, 0)
        new_max = int(masked.max())
        new_count_max = int(np.sum(masked == new_max)) // 2
        new_sum = int(np.sum(masked)) // 2
        
        # Revert
        A[remove_u, remove_v] = 1
        A[remove_v, remove_u] = 1
        A[add_x, add_y] = 0
        A[add_y, add_x] = 0
        
        return new_max, new_count_max, new_sum, old_books
    
    def bipartite_plus_init():
        """
        Start with K_{20,20} (400 edges) + 1 intra-partition edge.
        The single intra-edge has book size 20 (all 20 vertices in the other partition
        are common neighbors). We'll optimize from here.
        """
        A = np.zeros((n, n), dtype=np.int8)
        half = n // 2
        left = list(range(half))
        right = list(range(half, n))
        
        for u in left:
            for v in right:
                A[u, v] = 1
                A[v, u] = 1
        
        # Add one edge within left
        u, v = random.sample(left, 2)
        A[u, v] = 1
        A[v, u] = 1
        
        return A
    
    def greedy_init():
        """
        Greedy construction: add edges one at a time, choosing the edge
        that minimizes the resulting maximum book size.
        """
        A = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while edge_count < target_edges:
            best_edge = None
            best_book = float('inf')
            
            # Sample candidates for speed
            candidates = random.sample(all_edges, min(len(all_edges), 100))
            
            for u, v in candidates:
                if A[u, v] == 1:
                    continue
                common = np.sum(A[u, :] & A[v, :])
                if common < best_book:
                    best_book = common
                    best_edge = (u, v)
            
            if best_edge is None:
                for u, v in all_edges:
                    if A[u, v] == 0:
                        common = np.sum(A[u, :] & A[v, :])
                        if common < best_book:
                            best_book = common
                            best_edge = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            A[u, v] = 1
            A[v, u] = 1
            edge_count += 1
            
            if time.time() > deadline - 30:
                break
        
        # Fill remaining randomly
        if edge_count < target_edges:
            existing = set()
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1:
                        existing.add((i, j))
            
            for u, v in random.sample(all_edges, len(all_edges)):
                if edge_count >= target_edges:
                    break
                if (u, v) not in existing:
                    A[u, v] = 1
                    A[v, u] = 1
                    existing.add((u, v))
                    edge_count += 1
        
        return A
    
    # Try multiple initialization strategies
    best_score = None
    best_A = None
    
    # Strategy 1: Bipartite + 1
    A = bipartite_plus_init()
    cur_max, cur_cnt, cur_sum, masked = calc_books(A)
    cur_score = (cur_max, cur_cnt, cur_sum)
    if best_score is None or cur_score < best_score:
        best_score = cur_score
        best_A = A.copy()
    
    # Strategy 2: Greedy
    if time.time() < deadline - 30:
        A = greedy_init()
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
    
    # Now do local search from the best initialization
    A = best_A.copy()
    cur_max, cur_cnt, cur_sum, masked = calc_books(A)
    current_score = (cur_max, cur_cnt, cur_sum)
    
    stuck_counter = 0
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Evaluate current
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        current_score = (cur_max, cur_cnt, cur_sum)
        
        # Update best
        if (cur_max < best_score[0]) or \
           (cur_max == best_score[0] and cur_cnt < best_score[1]) or \
           (cur_max == best_score[0] and cur_cnt == best_score[1] and cur_sum < best_score[2]):
            best_score = current_score
            best_A = A.copy()
            stuck_counter = 0
            no_improve_counter = 0
            
            if best_score[0] <= 7:
                break
        else:
            stuck_counter += 1
            no_improve_counter += 1
        
        # If stuck for a long time, perform a random kick
        if stuck_counter > 50:
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
            non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            if edges_list and non_edges_list:
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                stuck_counter = 0
                continue
        
        # Generate candidates: focus on edges with high book size
        max_idxs = np.argwhere(masked == cur_max)
        max_edges = [(u, v) for u, v in max_idxs if u < v]
        
        if not max_edges:
            break
        
        # Pick a random edge from those with max book size
        u, v = random.choice(max_edges)
        
        best_swap_score = None
        best_swap = None
        best_books = None
        
        # Try swapping: remove edge (u,v), add edge (x,y)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        random.shuffle(non_edges)
        
        # Try up to 100 swaps with incremental scoring
        for x, y in non_edges[:100]:
            if (x, y) == (u, v) or (x, y) == (v, u):
                continue
            
            new_max, new_cnt, new_sum, new_books = incremental_swap_score(
                A, masked, u, v, x, y
            )
            new_score = (new_max, new_cnt, new_sum)
            
            if best_swap_score is None or \
               (new_score[0] < best_swap_score[0]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] < best_swap_score[1]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] == best_swap_score[1] and new_score[2] < best_swap_score[2]):
                best_swap_score = new_score
                best_swap = (x, y)
                best_books = new_books
            
            if time.time() > deadline - 10:
                break
        
        if best_swap is not None:
            x, y = best_swap
            if (best_swap_score[0] < cur_max) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] < cur_cnt) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] == cur_cnt and best_swap_score[2] < cur_sum):
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                # Update masked with the new books
                masked = best_books
            else:
                stuck_counter += 1
        
        # If no improvement for many iterations, try a restart with perturbation
        if no_improve_counter > 200:
            A = best_A.copy()
            for _ in range(3):
                edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
                non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
                if edges_list and non_edges_list:
                    u, v = random.choice(edges_list)
                    x, y = random.choice(non_edges_list)
                    A[u, v] = 0
                    A[v, u] = 0
                    A[x, y] = 1
                    A[y, x] = 1
            no_improve_counter = 0
            stuck_counter = 0
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            existing = set(tuple(e) for e in edges)
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
