import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # We represent the graph as an adjacency matrix for fast triangle counting.
    # A[u, v] = 1 if edge exists, 0 otherwise.
    # Book size of edge (u, v) is (A @ A)[u, v].
    
    def get_books(A):
        # Returns the matrix of book sizes for all pairs.
        # We only care about edges, but computing full matrix is fast (40x40).
        return A @ A

    def max_book(A):
        books = get_books(A)
        # Mask non-edges to 0
        masked = np.where(A > 0, books, 0)
        return int(masked.max())

    # --- Phase 1: Greedy Construction ---
    # Start with empty graph. Add edges one by one.
    # For each candidate edge, calculate the new max book if added.
    # Choose the edge that results in the smallest max book.
    # If ties, choose randomly among the best to avoid bias.
    
    A = np.zeros((n, n), dtype=np.int8)
    current_max = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # We need to add target_edges edges.
    # Naive greedy: For each of the 401 steps, try all remaining edges?
    # 780 edges * 401 steps * O(n^3) matrix mult is too slow if done naively.
    # However, updating A @ A incrementally is O(n) per edge addition.
    # Let's use incremental update.
    # B = A @ A. If we add edge (u, v):
    # A[u,v] += 1, A[v,u] += 1
    # B[u,v] += deg(u) + deg(v) - 1 (common neighbors + each other? No.)
    # Actually, B[u,v] = sum_k A[u,k]A[k,v].
    # When we add (u,v):
    # For any k, if A[u,k] and A[k,v] become 1, B increases.
    # This is complex to update incrementally for all pairs.
    # Given n=40, A @ A is very fast in NumPy. 
    # 401 steps. Each step we might check a subset of candidates.
    # Let's try a heuristic: check a random subset of 50 candidates per step.
    
    added_edges = []
    remaining_edges = all_edges.copy()
    random.shuffle(remaining_edges)
    
    # Optimization: Instead of checking all 780, we check a random sample of 100.
    # If we find an edge that keeps max_book <= current_max, we take it.
    # Otherwise, we take the one with the smallest resulting max_book.
    
    for _ in range(target_edges):
        if time.time() > deadline - 30: # Reserve time for repair
            break
            
        # Sample candidates
        sample_size = min(100, len(remaining_edges))
        candidates = random.sample(remaining_edges, sample_size)
        
        best_edge = None
        best_score = float('inf')
        best_books = None
        
        for u, v in candidates:
            # Tentatively add
            A[u, v] = 1
            A[v, u] = 1
            
            # Compute new max book
            books = A @ A
            masked = np.where(A > 0, books, 0)
            new_max = int(masked.max())
            
            # Revert
            A[u, v] = 0
            A[v, u] = 0
            
            if new_max < best_score:
                best_score = new_max
                best_edge = (u, v)
                best_books = books # We can't keep this, need to recompute after final add
                
        if best_edge is None:
            # Should not happen
            best_edge = remaining_edges[0]
            best_score = max_book(A)
            
        # Add the best edge
        u, v = best_edge
        A[u, v] = 1
        A[v, u] = 1
        remaining_edges.remove(best_edge)
        added_edges.append(best_edge)
        
    # If we didn't add enough edges due to time, fill with random remaining
    while len(added_edges) < target_edges:
        if not remaining_edges:
            break
        e = remaining_edges.pop()
        A[e[0], e[1]] = 1
        A[e[1], e[0]] = 1
        added_edges.append(e)

    current_max = max_book(A)
    best_max = current_max
    best_A = A.copy()
    
    # --- Phase 2: Targeted Repair (Book Breaking) ---
    # Strategy: Identify the edge (u, v) with the maximum book size.
    # Try to swap it with a non-edge (x, y) such that the new max book is lower.
    # Specifically, we want to remove (u, v) and add (x, y) such that
    # the new max book < current_max.
    # If we can't lower the max, we try to keep it same but improve second max, etc.
    # Simplest: Local search on the max book.
    
    # Precompute list of non-edges for faster sampling?
    # Or just random sampling.
    
    temp = 1.0
    # We have ~80 seconds left.
    # Each iteration: find max edge O(1) if we have books matrix.
    # Find swap O(1) sampling.
    # Check O(n^3) matrix mult.
    # 40x40 matrix mult is ~64k ops. 10k iterations = 640M ops. Might be tight in Python/NumPy overhead.
    # Let's optimize: Only recompute books if the move is promising?
    # No, we need exact max.
    
    # Better: Use incremental update for the specific edge being removed/added?
    # Too complex. Let's rely on NumPy speed for 40x40.
    
    while time.time() < deadline:
        # Find the edge with the largest book
        books = A @ A
        masked_books = np.where(A > 0, books, 0)
        max_val = int(masked_books.max())
        
        if max_val < best_max:
            best_max = max_val
            best_A = A.copy()
            
        # If max_val is very small (e.g., <= 2), we are likely near optimal.
        # The lower bound is ~6.6, so integer min is 7.
        # If we reach 7, we can stop early or keep searching for 6 (impossible by theorem? 40/6 = 6.66, so min integer is 7).
        # Wait, theorem says score >= 40/6. So 7 is the theoretical minimum integer.
        if best_max <= 7:
            break

        # Find all edges with book size == max_val
        # To diversify, pick one randomly
        # Create a list of (u,v) where A[u,v]=1 and books[u,v]==max_val
        # For efficiency, just pick a random edge and check if it's a max edge?
        # No, better to find one specifically.
        
        # Vectorized find
        # Flatten A and masked_books
        # This is O(n^2) per step. 1600 ops. Fine.
        
        # Get indices of max values in masked_books where A is 1
        # Since masked_books is 0 for non-edges, we just find max in masked_books
        # But there might be multiple.
        
        # Let's just pick a random edge from the set of max-book edges.
        # To do this efficiently:
        # 1. Get all indices where masked_books == max_val
        # 2. Filter to upper triangle (or just any u!=v)
        
        idxs = np.argwhere(masked_books == max_val)
        # Filter for u < v to avoid duplicates and diagonal (though diag is 0 in books for simple graph? 
        # A @ A diagonal is degree. But A[u,u]=0 so books[u,u]=0? 
        # Wait. A is 0/1. A[u,u]=0. 
        # (A@A)[u,u] = sum_k A[u,k]A[k,u] = degree(u).
        # But we masked with A>0. A[u,u]=0, so masked_books[u,u]=0.
        # So idxs will only contain off-diagonal edges.
        
        if len(idxs) == 0:
            # Should not happen if max_val > 0
            break
            
        # Pick a random one
        u, v = idxs[random.randint(0, len(idxs)-1)]
        
        # We want to remove (u, v) and add a non-edge (x, y).
        # We want the new max book to be < max_val.
        # Let's try a few random non-edges.
        
        improved = False
        # Try up to 20 random non-edges
        for _ in range(20):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or A[x, y] == 1:
                continue
            if (x, y) == (u, v) or (x, y) == (v, u):
                continue
                
            # Tentative move
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_books = A @ A
            new_masked = np.where(A > 0, new_books, 0)
            new_max = int(new_masked.max())
            
            if new_max < max_val:
                # Accept
                improved = True
                break
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
        
        if not improved:
            # If we couldn't improve by removing a max-edge, 
            # maybe we should try removing a non-max edge to allow a different structure?
            # Or accept a move that keeps max same but reduces sum of books?
            # For now, let's try a random move that doesn't increase max.
            # This is Simulated Annealing with T=0 (greedy descent).
            
            # Pick random edge to remove
            # Pick random non-edge to add
            # If new_max <= max_val, accept.
            
            # To avoid getting stuck, let's bias the removal towards high-book edges,
            # but allow removing lower-book edges if it allows adding an edge that lowers the global max?
            # Actually, if we remove a low-book edge, the max-book edges remain.
            # Adding a new edge might increase books of existing high-book edges if it shares neighbors.
            # So removing a high-book edge is usually better.
            
            # Let's try a different strategy: 
            # Randomly pick an edge (u,v). 
            # If its book is high, try to swap.
            # If its book is low, try to swap only if it doesn't increase max.
            
            # For now, if no improvement found, just do a random perturbation 
            # that keeps max <= current max.
            
            # Find a random edge
            # We can reuse the idxs logic but pick randomly from ALL edges
            # To be fast, just random sample
            found_move = False
            for _ in range(10):
                ru = random.randint(0, n-1)
                rv = random.randint(0, n-1)
                if ru == rv or A[ru, rv] == 0:
                    continue
                    
                rx = random.randint(0, n-1)
                ry = random.randint(0, n-1)
                if rx == ry or A[rx, ry] == 1:
                    continue
                if (rx, ry) == (ru, rv) or (rx, ry) == (rv, ru):
                    continue
                    
                A[ru, rv] = 0
                A[rv, ru] = 0
                A[rx, ry] = 1
                A[ry, rx] = 1
                
                nb = A @ A
                nm = np.where(A > 0, nb, 0)
                nmax = int(nm.max())
                
                if nmax <= max_val:
                    found_move = True
                    break
                else:
                    # Revert
                    A[ru, rv] = 1
                    A[rv, ru] = 1
                    A[rx, ry] = 0
                    A[ry, rx] = 0
            
            if not found_move:
                # Stuck. Do nothing or break?
                # Let's break to save time if we are stuck.
                # Or continue trying different max edges.
                pass

    # Final check and extraction
    final_A = best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if final_A[i, j] == 1:
                edges.append([i, j])
                
    # Ensure exactly 401 edges
    # If best_A has fewer/more (shouldn't happen), pad/trim
    if len(edges) != target_edges:
        # This is a fallback, shouldn't trigger
        pass
        
    return edges
