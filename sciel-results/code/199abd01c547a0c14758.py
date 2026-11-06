import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    n = 40
    target_edges = 401
    
    def build_books(A):
        """Compute full book size matrix from adjacency matrix."""
        B = A.astype(np.int32) @ A.astype(np.int32)
        # Mask non-edges to 0
        return np.where(A > 0, B, 0)
    
    def get_max_book(books):
        return int(books.max())
    
    def score_tuple(books):
        max_b = int(books.max())
        count_max = int(np.sum(books == max_b))
        sum_books = int(books.sum())
        return (max_b, count_max, sum_books)
    
    def update_books_on_add(A, books, u, v):
        """Incrementally update books when adding edge (u,v). Returns new books."""
        new_books = books.copy()
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        new_books = np.where(A > 0, new_books, 0)
        return new_books
    
    def update_books_on_remove(A, books, u, v):
        """Incrementally update books when removing edge (u,v)."""
        new_books = books.copy()
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        new_books = np.where(A > 0, new_books, 0)
        return new_books

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    best_score = None
    best_A = None
    best_books = None
    
    for trial in range(50): # Increase trials
        if time.time() > deadline - 10:
            break
        
        # Random Init
        random.shuffle(all_pairs)
        edges = all_pairs[:target_edges]
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        books = build_books(A)
        cur_score = score_tuple(books)
        cur_max = cur_score[0]
        
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
            best_books = books.copy()
            
        if cur_max <= 7:
            # Found optimal, break early
            break
            
        # Local Search with incremental updates
        stuck = 0
        max_iterations = 5000 # Increase iterations
        # Temperature for Simulated Annealing
        temp = 5.0
        cooling = 0.999
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find edges with maximum book size
            max_val = int(books.max())
            if max_val <= 7:
                break
            
            # Get all max-book edges
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and books[i, j] == max_val:
                        max_edges.append((i, j))
                        
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            # Get common neighbors of u and v
            common_neighbors = []
            for k in range(n):
                if k != u and k != v and A[u, k] == 1 and A[v, k] == 1:
                    common_neighbors.append(k)
            
            # Build candidate non-edges
            # We want to add an edge (x,y) that has low book size (few common neighbors)
            # and ideally doesn't increase the book size of other edges too much.
            
            candidates = []
            
            # Generate a pool of random non-edges
            # And some structured ones
            pool_size = 100
            for _ in range(pool_size):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and A[x, y] == 0:
                    x, y = min(x, y), max(x, y)
                    if x != u or y != v:
                        candidates.append((x, y))
            
            # Also add edges between non-neighbors of u and v to be safe?
            # Or just rely on random.
            
            random.shuffle(candidates)
            
            improved = False
            best_candidate = None
            best_new_score = None
            best_A_temp = None
            best_books_temp = None
            
            # Evaluate top candidates
            for x, y in candidates[:20]:
                if A[x, y] == 1:
                    continue
                    
                # Perform swap: remove (u,v), add (x,y)
                A[u, v] = 0
                A[v, u] = 0
                books_temp = update_books_on_remove(A, books, u, v)
                
                A[x, y] = 1
                A[y, x] = 1
                books_temp = update_books_on_add(A, books_temp, x, y)
                
                new_score = score_tuple(books_temp)
                
                # Accept if better, or with probability exp(-delta/temp) if worse (SA)
                if new_score < cur_score:
                    if best_new_score is None or new_score < best_new_score:
                        best_new_score = new_score
                        best_candidate = (x, y)
                        best_A_temp = A.copy()
                        best_books_temp = books_temp.copy()
                else:
                    # Calculate delta for SA
                    # Use a scalar metric: max_book * 100 + count_max
                    curr_scalar = cur_score[0] * 100 + cur_score[1]
                    new_scalar = new_score[0] * 100 + new_score[1]
                    delta = new_scalar - curr_scalar
                    if delta < 0:
                        if best_new_score is None or new_score < best_new_score:
                            best_new_score = new_score
                            best_candidate = (x, y)
                            best_A_temp = A.copy()
                            best_books_temp = books_temp.copy()
                    else:
                        # Probability of accepting worse move
                        if random.random() < np.exp(-delta / temp):
                            if best_new_score is None or new_score < best_new_score:
                                best_new_score = new_score
                                best_candidate = (x, y)
                                best_A_temp = A.copy()
                                best_books_temp = books_temp.copy()
                
                # Revert A and books for next candidate
                A[x, y] = 0
                A[y, x] = 0
                # Note: books_temp is a copy, so A is the only state to revert
                # But we modified A in place. We need to revert A to original state.
                # Original state had (u,v) removed.
                # We added (x,y). Now we remove (x,y).
                # We still need to restore (u,v) before the next candidate?
                # No, the loop structure is:
                # 1. Remove (u,v)
                # 2. Add (x,y)
                # 3. Evaluate
                # 4. Remove (x,y)
                # 5. Next candidate (x', y')
                #    Add (x', y')
                #    ...
                # After loop, if improved, we keep the changes.
                # If not, we must restore (u,v).
                
            if best_candidate is not None:
                x, y = best_candidate
                # We already have best_A_temp and best_books_temp
                A = best_A_temp
                books = best_books_temp
                cur_score = best_new_score
                cur_max = cur_score[0]
                improved = True
                
                if best_score is None or cur_score < best_score:
                    best_score = cur_score
                    best_A = A.copy()
                    best_books = books.copy()
            else:
                # Revert the removal of (u,v)
                A[u, v] = 1
                A[v, u] = 1
                books = update_books_on_add(A, books, u, v)
                stuck += 1
                if stuck > 100:
                    break
                    
            temp *= cooling
    
    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        existing = set(tuple(e) for e in edges)
        if len(edges) > target_edges:
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
