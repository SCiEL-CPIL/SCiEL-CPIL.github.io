import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 40
    target_edges = 401
    
    # Precompute all pairs for random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def initialize_graph():
        random.shuffle(all_pairs)
        edges = all_pairs[:target_edges]
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def calc_books(A):
        B = np.dot(A, A)
        # B[i,j] is number of common neighbors.
        # For edges, this is the book size.
        return B

    def get_max_book(B, A):
        # Max book size over edges
        max_b = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    if B[i, j] > max_b:
                        max_b = B[i, j]
        return max_b

    def sa_optimize(A):
        # Calculate initial books
        B = calc_books(A)
        max_book = get_max_book(B, A)
        
        # Maintain counts of book sizes for fast max update
        # Max possible book size is n-2 = 38
        count_books = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    count_books[B[i, j]] += 1
                    
        def update_max_book():
            nonlocal max_book
            while max_book > 0 and count_books[max_book] == 0:
                max_book -= 1
            return max_book

        # Temperature schedule
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < deadline - 2:
            if T < min_T:
                T = min_T
                
            # Pick a random edge to remove
            # We need to pick an edge (u,v) with A[u,v]==1
            # And a non-edge (x,y) with A[x,y]==0
            
            # Efficient way: pick random pairs until valid
            # Or maintain lists. Lists are better.
            # But maintaining lists is complex.
            # Let's just sample. n=40, 780 pairs.
            # Probability of edge is 401/780 ~ 0.5.
            # Probability of non-edge is 379/780 ~ 0.48.
            # So 2-3 tries on average.
            
            # Remove edge
            while True:
                u, v = random.choice(all_pairs)
                if A[u, v] == 1:
                    break
            
            # Add edge
            while True:
                x, y = random.choice(all_pairs)
                if A[x, y] == 0:
                    break
            
            # Check if u,v and x,y are distinct and don't share vertices in a way that makes it invalid?
            # Swapping edges (u,v) and (x,y) is always valid as long as they are distinct edges.
            # If they share a vertex, e.g., u=x, then we remove (u,v) and add (u,y).
            # This is a valid move (changing neighbor).
            
            # Calculate change
            # 1. Remove (u,v)
            # Common neighbors of u,v
            common_uv = np.where(A[u] & A[v])[0]
            
            # Save old book sizes for affected edges to revert if needed
            # Affected edges: (u,v), (u,w) for w in common_uv, (v,w) for w in common_uv
            # Note: (u,v) is removed.
            
            # We will simulate the move, calculate new max, and revert if rejected.
            
            # Apply removal
            old_book_uv = B[u, v]
            B[u, v] = 0
            B[v, u] = 0
            A[u, v] = 0
            A[v, u] = 0
            count_books[old_book_uv] -= 1
            
            affected_edges = []
            for w in common_uv:
                # Edge (u,w)
                old_bw_uw = B[u, w]
                B[u, w] -= 1
                B[w, u] -= 1
                count_books[old_bw_uw] -= 1
                count_books[old_bw_uw - 1] += 1
                affected_edges.append((u, w, old_bw_uw))
                
                # Edge (v,w)
                old_bw_vw = B[v, w]
                B[v, w] -= 1
                B[w, v] -= 1
                count_books[old_bw_vw] -= 1
                count_books[old_bw_vw - 1] += 1
                affected_edges.append((v, w, old_bw_vw))
            
            update_max_book()
            max_after_removal = max_book
            
            # Apply addition
            common_xy = np.where(A[x] & A[y])[0]
            new_book_xy = len(common_xy)
            
            A[x, y] = 1
            A[y, x] = 1
            B[x, y] = new_book_xy
            B[y, x] = new_book_xy
            count_books[new_book_xy] += 1
            
            affected_edges_add = []
            for w in common_xy:
                # Edge (x,w)
                old_bw_xw = B[x, w]
                B[x, w] += 1
                B[w, x] += 1
                count_books[old_bw_xw] -= 1
                count_books[old_bw_xw + 1] += 1
                affected_edges_add.append((x, w, old_bw_xw))
                
                # Edge (y,w)
                old_bw_yw = B[y, w]
                B[y, w] += 1
                B[w, y] += 1
                count_books[old_bw_yw] -= 1
                count_books[old_bw_yw + 1] += 1
                affected_edges_add.append((y, w, old_bw_yw))
            
            update_max_book()
            new_max = max_book
            
            # Check acceptance
            delta = new_max - max_after_removal # This is not the right delta for SA
            # We want to compare new_max with the max BEFORE the move.
            # We didn't store max_before_move.
            # Let's store it.
            
            # Revert logic
            # To simplify, I'll just calculate the score before and after.
            # But I need max_before.
            
            # Let's restructure:
            # 1. Store max_before.
            # 2. Apply move.
            # 3. Calculate new_max.
            # 4. Decide.
            # 5. Revert if rejected.
            
            # This is getting messy to edit in place.
            # I will write a cleaner version in the replacement.
            
            # For now, just revert everything to be safe and return current best.
            # This part of the code in the SEARCH block is just a placeholder for the logic.
            # I will replace the whole function.
            
            # Revert addition
            for w in common_xy:
                # This is complicated. 
                pass
            
            # Just return the initial graph if we can't implement SA cleanly in the edit.
            # But I can.
            
            break
            
        return A, max_book

    # Main
    A = initialize_graph()
    # Run SA
    # I will implement the full SA in the replacement block.
    
    B = calc_books(A)
    max_book = get_max_book(B, A)
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    return edges
