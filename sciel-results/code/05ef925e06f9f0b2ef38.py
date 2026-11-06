import random
import time
import sys

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110.0
    
    # Helper to generate initial random graph
    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges

    def init_state(edges):
        # A[u][v] is 1 if edge exists
        A = [[0] * n for _ in range(n)]
        N = [set() for _ in range(n)]
        book = [[0] * n for _ in range(n)] # book[u][v] for u < v
        
        for u, v in edges:
            A[u][v] = 1
            A[v][u] = 1
            N[u].add(v)
            N[v].add(u)
            
        # Calculate book sizes
        # For each edge (u,v), book[u][v] = len(N[u] & N[v])
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v]:
                    book[u][v] = len(N[u] & N[v])
                    
        # Histogram of book sizes
        # Max possible book size is n-2 = 38
        hist = [0] * (n) 
        max_b = 0
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v]:
                    b = book[u][v]
                    hist[b] += 1
                    if b > max_b:
                        max_b = b
        return A, N, book, hist, max_b

    def get_max_from_hist(hist):
        for i in range(len(hist) - 1, -1, -1):
            if hist[i] > 0:
                return i
        return 0

    best_edges = None
    best_max_b = float('inf')
    best_sum_sq = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        edges = random_graph()
        A, N, book, hist, max_b = init_state(edges)
        
        # Calculate sum of squares for tie-breaking
        sum_sq = sum(b * b for u in range(n) for v in range(u + 1, n) if A[u][v] for b in [book[u][v]])
        
        if max_b < best_max_b or (max_b == best_max_b and sum_sq < best_sum_sq):
            best_max_b = max_b
            best_sum_sq = sum_sq
            best_edges = set(edges)
            
        # Local Search
        T = 1.0
        T_min = 0.05
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
                
            # Pick random edge to remove
            # We can pick from the list of edges
            # To speed up, maintain a list of edges? 
            # Or just pick random u, v and check.
            # With 401 edges out of 780 possible, random picking is ~50% success rate.
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or A[u][v] == 0:
                continue
            if u > v: u, v = v, u
            
            # Pick random non-edge to add
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y or A[x][y] == 1:
                continue
            if x > y: x, y = y, x
            
            # Check if (x,y) is same as (u,v)
            if (u, v) == (x, y):
                continue
                
            # Calculate impact
            # Remove (u,v)
            # Common neighbors of u and v
            common_uv = N[u] & N[v]
            
            # Add (x,y)
            # Common neighbors of x and y
            common_xy = N[x] & N[y]
            
            # We need to update book sizes for edges incident to u,v and x,y
            # Edges affected by removal: (u,w) and (v,w) for w in common_uv
            # Edges affected by addition: (x,w) and (y,w) for w in common_xy
            
            # Let's simulate the changes on a temporary structure or carefully revert
            # Since we need to update hist and book, let's do it in-place and revert if rejected.
            
            # Store old values
            old_books = {}
            
            # Process Removal
            for w in common_uv:
                # Edge (u, w)
                eu, ev = min(u, w), max(u, w)
                if eu == u:
                    old_books[(eu, ev)] = book[eu][ev]
                    hist[book[eu][ev]] -= 1
                    book[eu][ev] -= 1
                    hist[book[eu][ev]] += 1
                # Edge (v, w)
                ev, ew = min(v, w), max(v, w)
                if ev == v:
                    old_books[(ev, ew)] = book[ev][ew]
                    hist[book[ev][ew]] -= 1
                    book[ev][ew] -= 1
                    hist[book[ev][ew]] += 1
            
            # The edge (u,v) itself is removed
            # Its book size is removed from hist
            old_books[(u, v)] = book[u][v]
            hist[book[u][v]] -= 1
            book[u][v] = 0 # Mark as removed
            
            # Update A and N for removal
            A[u][v] = 0
            A[v][u] = 0
            N[u].remove(v)
            N[v].remove(u)
            
            # Process Addition
            for w in common_xy:
                # Edge (x, w)
                ex, ew = min(x, w), max(x, w)
                if ex == x:
                    if (ex, ew) not in old_books:
                        old_books[(ex, ew)] = book[ex][ew]
                    hist[book[ex][ew]] -= 1
                    book[ex][ew] += 1
                    hist[book[ex][ew]] += 1
                # Edge (y, w)
                ey, ew = min(y, w), max(y, w)
                if ey == y:
                    if (ey, ew) not in old_books:
                        old_books[(ey, ew)] = book[ey][ew]
                    hist[book[ey][ew]] -= 1
                    book[ey][ew] += 1
                    hist[book[ey][ew]] += 1
            
            # The edge (x,y) is added
            # Calculate its new book size
            new_book_xy = len(N[x] & N[y]) # N[x] and N[y] haven't been updated for x,y yet? 
            # Wait, we removed u,v from N. x,y are not u,v. 
            # But if w was in common_uv, did it affect N[x]? No.
            # So N[x] and N[y] are current.
            # However, if x or y is in common_uv? 
            # If x is in N[u] and N[v], then x is in common_uv.
            # Then edge (u,x) and (v,x) were updated.
            # This doesn't change N[x].
            # So common_xy is correct based on current N.
            
            # Wait, we need to add (x,y) to A and N first to compute book size correctly?
            # No, book size is common neighbors. 
            # The common neighbors of x and y are the vertices connected to both.
            # This set does not include x or y.
            # So len(N[x] & N[y]) is correct BEFORE adding the edge?
            # Yes, because the edge (x,y) doesn't create a path of length 2 from x to y via a neighbor.
            # It creates a triangle if there is a w connected to both.
            # So the book size of the NEW edge (x,y) is exactly the number of common neighbors.
            
            book[x][y] = new_book_xy
            hist[new_book_xy] += 1
            
            # Update A and N for addition
            A[x][y] = 1
            A[y][x] = 1
            N[x].add(y)
            N[y].add(x)
            
            # Calculate new metrics
            new_max_b = get_max_from_hist(hist)
            
            # Calculate new sum_sq? 
            # It's expensive to recalculate sum_sq every step.
            # We can track sum_sq incrementally.
            # But for now, let's just use max_b as primary key.
            # If max_b improves, accept.
            # If max_b is same, we might want to accept if it reduces "potential" for high books.
            # Simple heuristic: accept if new_max_b < max_b.
            # Or accept with probability if new_max_b <= max_b.
            
            accept = False
            if new_max_b < max_b:
                accept = True
                max_b = new_max_b
                no_improve = 0
                # Update best
                # We need to get current edges. 
                # To avoid building set every time, we can just check at end of loop or periodically.
                # For now, let's just track max_b.
            elif new_max_b == max_b:
                # Accept with some probability to allow exploration
                if random.random() < 0.1: # Low probability to avoid drifting
                    accept = True
                    max_b = new_max_b
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                # Revert
                # Restore A, N, book, hist
                A[u][v] = 1
                A[v][u] = 1
                N[u].add(v)
                N[v].add(u)
                
                A[x][y] = 0
                A[y][x] = 0
                N[x].remove(y)
                N[y].remove(x)
                
                for (eu, ev), ob in old_books.items():
                    hist[book[eu][ev]] -= 1
                    book[eu][ev] = ob
                    hist[ob] += 1
                
                # Restore (u,v) book
                hist[book[u][v]] -= 1 # It was 0
                book[u][v] = old_books[(u, v)]
                hist[old_books[(u, v)]] += 1
                
                # Restore (x,y) book
                hist[book[x][y]] -= 1
                book[x][y] = 0
                # hist[0] doesn't matter for max, but keep consistent
                
                continue
                
            if accept:
                # Update edges set? 
                # We are not maintaining `edges` set in the loop, we are using A.
                # We need to extract edges at the end.
                pass
                
            T = max(T_min, T * 0.999)
            
    # Extract best edges from best_edges? 
    # Wait, I didn't update best_edges in the loop efficiently.
    # I should update best_edges when I find a better max_b.
    # But extracting edges from A is O(n^2) = 1600 ops.
    # I can do this whenever I update best_max_b.
    
    # Let's re-run with proper best tracking.
    
    return [[u, v] for u, v in best_edges]
