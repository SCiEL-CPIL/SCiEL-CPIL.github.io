import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for overhead
    
    n = 40
    m = 401
    
    best_edges = None
    best_score = float('inf')
    
    def init_random():
        """Generate a random graph with n vertices and m edges."""
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
        return list(edges)

    def init_turan_plus_one():
        """K20,20 plus one edge inside a part."""
        edges = set()
        for i in range(20):
            for j in range(20, 40):
                edges.add((i, j))
        edges.add((0, 1))
        return list(edges)
        
    def init_greedy():
        """Greedy construction trying to keep degrees balanced."""
        edges = set()
        degrees = [0] * n
        # Start with a random subset to avoid symmetry breaking issues? 
        # Or just pick random pairs but prefer those that don't increase max degree too much?
        # Let's just use random for simplicity, but maybe shuffle vertices?
        # Actually, a random graph is a good start for local search.
        return init_random()

    def evaluate(edges):
        """Calculate max book size for a given edge list."""
        # Build adjacency
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        max_book = 0
        for u, v in edges:
            # Count common neighbors
            count = 0
            for k in range(n):
                if adj[u][k] and adj[v][k]:
                    count += 1
            if count > max_book:
                max_book = count
        return max_book

    def local_search(initial_edges, time_limit):
        """
        Perform local search starting from initial_edges.
        Returns best edges and score found during this run.
        """
        edges = set(initial_edges)
        # Precompute adjacency and book sizes
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Calculate initial book sizes
        book = {}
        max_book = 0
        for u, v in edges:
            b = 0
            for k in range(n):
                if adj[u][k] and adj[v][k]:
                    b += 1
            book[(u,v)] = b
            if b > max_book:
                max_book = b
        
        # Sum of squares for tie-breaking
        sum_sq = sum(b*b for b in book.values())
        
        best_edges = list(edges)
        best_score = max_book
        best_sum_sq = sum_sq
        
        temp = 1.0
        cooling = 0.9999
        steps = 0
        last_time = time.time()
        
        while time.time() < start_time + time_limit:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Select random edge to remove
            u, v = random.choice(list(edges))
            
            # Select random non-edge to add
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if (x, y) not in edges:
                    break
            
            # Identify affected vertices
            aff_verts = {u, v, x, y}
            
            # Identify affected edges (edges in current E incident to aff_verts)
            # Note: (u,v) is in E. (x,y) is not.
            # After swap: (u,v) out, (x,y) in.
            # Affected edges are those incident to u, v, x, y.
            
            # Store old book sizes for affected edges to rollback if needed
            # But we can just recalculate.
            
            # Perform swap
            edges.remove((u, v))
            edges.add((x, y))
            
            adj[u][v] = 0
            adj[v][u] = 0
            adj[x][y] = 1
            adj[y][x] = 1
            
            # Recalculate book sizes for edges incident to aff_verts
            # We need to update `book` dict.
            # Edges to check: all (a,b) in edges such that a in aff_verts or b in aff_verts.
            # Also, the book size of (u,v) is gone.
            # The book size of (x,y) is new.
            
            # To be efficient, we iterate through neighbors of u, v, x, y.
            # However, `book` is a dict. We can just delete old entries and add new ones.
            
            # Find all edges incident to aff_verts in the NEW edge set
            new_aff_edges = set()
            for vert in aff_verts:
                for k in range(n):
                    if adj[vert][k]:
                        a, b = vert, k
                        if a > b: a, b = b, a
                        new_aff_edges.add((a, b))
                        
            # Update book sizes for these edges
            # First, remove old book sizes for edges that are NO LONGER affected or were removed
            # Actually, simpler: just recalculate for new_aff_edges.
            # But we need to know the old max and sum_sq.
            
            # Let's maintain a histogram of book sizes to find max quickly?
            # With ~400 edges, finding max is O(E). 400 ops is trivial.
            # Let's just recalculate max and sum_sq from scratch? 
            # No, that's O(E * V) which is 400*40 = 16000 ops. 
            # 16000 ops * 10000 steps = 160M ops. Might be too slow in Python.
            
            # Better: Update only affected books.
            # 1. Get current max and sum_sq.
            # 2. For each edge in new_aff_edges:
            #    old_book = book.get((a,b), -1)
            #    if old_book != -1:
            #        sum_sq -= old_book**2
            #        # We need to handle max carefully. 
            #        # If old_book was the max, the new max might be lower.
            #        # We can't easily know the new max without a histogram or full scan.
            
            # Given the small size, let's use a histogram of book sizes.
            # Max book size is bounded by n-2 = 38.
            # hist[b] = number of edges with book size b.
            
            # Re-initialize hist for this run?
            # Let's rebuild hist from `book` at start.
            
            # Let's restart the local search logic with a histogram.
            
            pass # Placeholder for the actual loop implementation below

    # Implement a cleaner local search with histogram
    
    def local_search_v2(initial_edges, time_limit):
        edges = set(initial_edges)
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        book = {}
        hist = [0] * (n + 1)
        max_book = 0
        sum_sq = 0
        
        for u, v in edges:
            b = 0
            for k in range(n):
                if adj[u][k] and adj[v][k]:
                    b += 1
            book[(u,v)] = b
            hist[b] += 1
            if b > max_book:
                max_book = b
            sum_sq += b*b
            
        best_edges = list(edges)
        best_score = max_book
        best_sum_sq = sum_sq
        best_hist = hist[:]
        
        temp = 1.0
        # Simple acceptance: accept if better, or with prob exp(-delta/T) if worse?
        # Since objective is discrete, let's try:
        # Accept if (new_max < old_max) OR (new_max == old_max and new_sum_sq < old_sum_sq)
        # Else reject. This is a greedy hill climber.
        # To escape, we can allow worsening with small prob.
        
        steps = 0
        while time.time() < start_time + time_limit:
            if time.time() > deadline:
                break
                
            u, v = random.choice(list(edges))
            
            # Pick non-edge
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if (x, y) not in edges:
                    break
            
            aff_verts = [u, v, x, y]
            
            # Collect edges incident to aff_verts in current graph
            # We need to update book sizes for:
            # 1. Edges incident to u, v, x, y that remain in E.
            # 2. The new edge (x,y).
            # 3. The removed edge (u,v) is gone.
            
            # Identify set of edges to recalculate
            recalc_edges = set()
            for vert in aff_verts:
                for k in range(n):
                    if adj[vert][k]:
                        a, b = vert, k
                        if a > b: a, b = b, a
                        recalc_edges.add((a, b))
                        
            # Also, if (u,v) was in recalc_edges, it will be removed.
            # If (x,y) is in recalc_edges (it shouldn't be, as it's a non-edge), it will be added.
            # Wait, (x,y) is not in edges, so it's not in recalc_edges derived from adj.
            # We must explicitly add (x,y) to recalc_edges after adding it.
            
            # Store old values
            old_books = {}
            for e in recalc_edges:
                if e in book:
                    old_books[e] = book[e]
            
            # Perform swap
            edges.remove((u, v))
            edges.add((x, y))
            adj[u][v] = 0
            adj[v][u] = 0
            adj[x][y] = 1
            adj[y][x] = 1
            
            # Update recalc_edges to include (x,y) and exclude (u,v) if present
            # (u,v) was in edges, so it was in recalc_edges.
            # (x,y) is now in edges.
            
            new_books = {}
            for e in recalc_edges:
                a, b = e
                # Calculate new book size
                cnt = 0
                for k in range(n):
                    if adj[a][k] and adj[b][k]:
                        cnt += 1
                new_books[e] = cnt
                
            # Add (x,y) to recalc if not already there (it wasn't, as it was non-edge)
            # Actually, if x or y was in aff_verts, (x,y) might have been considered?
            # No, recalc_edges was built from adj BEFORE swap.
            # So (x,y) is not in recalc_edges.
            # We need to calculate book for (x,y) too.
            a, b = x, y
            if a > b: a, b = b, a
            e_xy = (a, b)
            cnt = 0
            for k in range(n):
                if adj[a][k] and adj[b][k]:
                    cnt += 1
            new_books[e_xy] = cnt
            
            # Remove (u,v) from consideration if it was in old_books
            e_uv = (u, v)
            if u > v: e_uv = (v, u)
            
            # Update hist, sum_sq, max_book
            # Remove old contributions
            for e, b_old in old_books.items():
                if e == e_uv:
                    continue # This edge is removed
                hist[b_old] -= 1
                sum_sq -= b_old*b_old
                
            # Add new contributions
            for e, b_new in new_books.items():
                hist[b_new] += 1
                sum_sq += b_new*b_new
                book[e] = b_new
                
            # Clean up book dict for removed edge
            if e_uv in book:
                del book[e_uv]
                
            # Update max_book
            # Find max index in hist
            new_max = 0
            for i in range(n, -1, -1):
                if hist[i] > 0:
                    new_max = i
                    break
                    
            # Acceptance
            accepted = False
            if new_max < max_book:
                accepted = True
            elif new_max == max_book:
                if sum_sq < best_sum_sq: # Use best_sum_sq as reference? Or current?
                    # Let's compare to current state's sum_sq
                    # We stored sum_sq as current.
                    pass
                # Let's just accept if sum_sq is lower than current sum_sq
                # We need current sum_sq before update.
                # Let's restructure: keep old_sum_sq.
                
            # This logic is getting messy with in-place updates.
            # Let's do a "dry run" calculation for the new state's metrics before committing.
            
            pass # Restart with cleaner state management

    # Let's write a robust version of local search
    
    def run_search(init_edges, duration):
        edges = set(init_edges)
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        book = {}
        hist = [0] * (n + 1)
        
        def calc_books():
            nonlocal max_book, sum_sq
            max_book = 0
            sum_sq = 0
            for i in range(n+1):
                hist[i] = 0
            for u, v in edges:
                b = 0
                for k in range(n):
                    if adj[u][k] and adj[v][k]:
                        b += 1
                book[(u,v)] = b
                hist[b] += 1
                sum_sq += b*b
                if b > max_book:
                    max_book = b
                    
        max_book = 0
        sum_sq = 0
        calc_books()
        
        best_edges = list(edges)
        best_score = max_book
        best_sum_sq = sum_sq
        
        end_time = time.time() + duration
        steps = 0
        
        while time.time() < end_time:
            if time.time() > deadline:
                break
                
            # Random move
            u, v = random.choice(list(edges))
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if (x, y) not in edges:
                    break
                    
            # Calculate delta
            # Affected vertices: u, v, x, y
            aff = {u, v, x, y}
            
            # Edges to check: all edges incident to aff
            # Plus the new edge (x,y)
            # Minus the old edge (u,v)
            
            # Let's just recalculate the entire book vector? 
            # 400 edges * 40 nodes = 16000 ops.
            # If we do 10000 steps, 160M ops. Python might take ~10-20s per 100M ops?
            # So 10000 steps might take 20s.
            # We have 115s. So ~50000 steps possible.
            # This is feasible.
            
            # To be safer, let's do incremental updates properly.
            
            # 1. Identify affected edges in OLD graph
            old_aff_edges = set()
            for vert in aff:
                for k in range(n):
                    if adj[vert][k]:
                        a, b = vert, k
                        if a > b: a, b = b, a
                        old_aff_edges.add((a, b))
                        
            # 2. Identify affected edges in NEW graph
            # Remove (u,v), Add (x,y)
            # Update adj temporarily? No, let's just calculate new book sizes for:
            # - All edges in old_aff_edges except (u,v)
            # - The new edge (x,y)
            # - Any edge that becomes incident to u,v,x,y due to the swap?
            #   No, only u,v,x,y neighbors change.
            #   So the set of edges whose book size CAN change is exactly:
            #   Edges incident to u, v, x, y in the NEW graph.
            
            # Let's compute new book sizes for these edges.
            # We need to know the new adj state.
            # Let's simulate adj changes on a copy? 
            # adj is 40x40. Copying is cheap.
            
            adj_new = [row[:] for row in adj]
            adj_new[u][v] = 0
            adj_new[v][u] = 0
            adj_new[x][y] = 1
            adj_new[y][x] = 1
            
            new_aff_edges = set()
            for vert in aff:
                for k in range(n):
                    if adj_new[vert][k]:
                        a, b = vert, k
                        if a > b: a, b = b, a
                        new_aff_edges.add((a, b))
                        
            # Calculate new book sizes for new_aff_edges
            new_books = {}
            for a, b in new_aff_edges:
                cnt = 0
                for k in range(n):
                    if adj_new[a][k] and adj_new[b][k]:
                        cnt += 1
                new_books[(a,b)] = cnt
                
            # Calculate old books for these edges (from current `book` dict)
            # Note: (u,v) is in old_aff_edges but not in new_aff_edges (removed).
            # (x,y) is in new_aff_edges but not in old_aff_edges (added).
            
            # Compute new max and sum_sq
            # We need to update hist.
            # Let's create a new hist from current hist.
            new_hist = hist[:]
            
            # Subtract old books for edges that are changing or removed
            # Edges to subtract:
            # 1. (u,v) if it was in book.
            # 2. Any edge in (old_aff_edges U new_aff_edges) that is in book.
            
            all_changed_edges = old_aff_edges.union(new_aff_edges)
            
            valid_old_edges = set()
            for e in all_changed_edges:
                if e in book:
                    valid_old_edges.add(e)
                    
            # Subtract
            for e in valid_old_edges:
                b = book[e]
                new_hist[b] -= 1
                
            # Add new books
            for e, b in new_books.items():
                new_hist[b] += 1
                
            # Find new max
            new_max = 0
            for i in range(n, -1, -1):
                if new_hist[i] > 0:
                    new_max = i
                    break
                    
            # Calculate new sum_sq
            # old_sum_sq - sum(b^2 for old changed) + sum(b^2 for new)
            # We need to be careful not to double count or miss.
            # The edges in `new_books` are the ONLY ones with changed book sizes.
            # The edges in `valid_old_edges` are the ones we subtracted.
            # Are they the same set?
            # `new_books` keys are `new_aff_edges`.
            # `valid_old_edges` are `all_changed_edges` that existed.
            # If an edge exists in both, it's updated.
            # If it was in old but not new (removed), we subtracted it.
            # If it was in new but not old (added), we added it.
            # If it was in neither, it's not in either set.
            
            # So:
            # new_sum_sq = sum_sq
            # for e in valid_old_edges: new_sum_sq -= book[e]**2
            # for e, b in new_books.items(): new_sum_sq += b**2
            
            new_sum_sq = sum_sq
            for e in valid_old_edges:
                new_sum_sq -= book[e]**2
            for e, b in new_books.items():
                new_sum_sq += b**2
                
            # Acceptance
            if new_max < max_book:
                # Apply
                edges = set()
                for e in edges:
                    if e != (u,v) and e != (v,u): # Handle orientation
                        pass
                # Simpler:
                edges.remove((u,v))
                edges.add((x,y))
                adj = adj_new
                # Update book dict
                for e in valid_old_edges:
                    if e in book:
                        del book[e]
                for e, b in new_books.items():
                    book[e] = b
                hist = new_hist
                max_book = new_max
                sum_sq = new_sum_sq
                
                if max_book < best_score:
                    best_score = max_book
                    best_edges = list(edges)
                    best_sum_sq = sum_sq
                    
            elif new_max == max_book and new_sum_sq < sum_sq:
                # Accept to improve distribution
                edges.remove((u,v))
                edges.add((x,y))
                adj = adj_new
                for e in valid_old_edges:
                    if e in book:
                        del book[e]
                for e, b in new_books.items():
                    book[e] = b
                hist = new_hist
                sum_sq = new_sum_sq
                
                if max_book < best_score or (max_book == best_score and sum_sq < best_sum_sq):
                    best_score = max_book
                    best_edges = list(edges)
                    best_sum_sq = sum_sq
            else:
                # Reject, do nothing (state unchanged)
                pass
                
            steps += 1
            if steps % 1000 == 0:
                # Optional: check time
                pass
                
        return best_edges, best_score

    # Run multiple restarts with different initializations
    # Allocation of time:
    # 1. Turan + 1 (baseline)
    # 2. Random
    # 3. Random
    # 4. Random
    
    inits = [
        init_turan_plus_one(),
        init_random(),
        init_random(),
        init_random()
    ]
    
    time_per_run = 25.0
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        edges, score = run_search(init, time_per_run)
        if score < best_score:
            best_score = score
            best_edges = edges
            
    if best_edges is None:
        best_edges = init_turan_plus_one()
        
    # Convert to list of lists
    return [[u, v] for u, v in best_edges]
