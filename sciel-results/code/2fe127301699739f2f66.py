import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # Initial random graph
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42) # Fixed seed for reproducibility in development, but we might want random start?
    # Actually, using multiple random restarts is better.
    
    best_score = float('inf')
    best_edges = None
    
    def get_max_book(adj, edges, book):
        max_b = 0
        for u, v in edges:
            if book[u][v] > max_b:
                max_b = book[u][v]
        return max_b

    def calculate_books(adj, edges):
        book = [[0]*n for _ in range(n)]
        for u, v in edges:
            common = adj[u] & adj[v]
            book[u][v] = len(common)
            book[v][u] = len(common)
        return book

    def simulate_annealing(initial_edges, initial_adj, initial_book, max_time):
        edges = set(initial_edges)
        adj = {i: set(initial_adj[i]) for i in range(n)}
        book = [row[:] for row in initial_book]
        
        current_max = get_max_book(adj, edges, book)
        best_local_max = current_max
        best_local_edges = set(edges)
        
        T = 10.0
        cooling = 0.999
        
        # Precompute non-edges for faster sampling? 
        # We can just sample random pairs and check if in edges.
        
        while time.time() < max_time:
            # Pick a random edge to remove
            u, v = random.choice(list(edges))
            
            # Pick a random non-edge to add
            # Try a few times to find a non-edge
            x, y = random.choice(all_pairs)
            if (x, y) in edges or (x, y) == (u, v):
                continue
            # Ensure x,y distinct and not same as u,v
            if x == y:
                continue
            
            # Perform swap: remove (u,v), add (x,y)
            
            # 1. Remove (u,v)
            # Update book sizes for neighbors of u and v
            # For any w in N(u), if w is in N(v), book[u][w] decreases by 1
            # For any w in N(v), if w is in N(u), book[v][w] decreases by 1
            
            # We need to be careful not to double count or modify while iterating
            # Let's collect changes
            
            changes = []
            
            # Neighbors of u
            for w in list(adj[u]):
                if w == v:
                    continue
                if w in adj[v]:
                    # w is common neighbor of u and v? No, w is neighbor of u and v.
                    # The edge is (u, w). Common neighbors of u and w included v.
                    # So book[u][w] decreases by 1.
                    changes.append((u, w, -1))
                    changes.append((w, u, -1))
                # Also, edge (v, w) is affected?
                # w is neighbor of v. Is w neighbor of u? Yes, we are iterating N(u).
                # So w is in N(u) and N(v).
                # Edge (v, w): Common neighbors of v and w included u.
                # So book[v][w] decreases by 1.
                changes.append((v, w, -1))
                changes.append((w, v, -1))
            
            # Wait, the logic above is slightly flawed.
            # If w is in N(u) and N(v), then:
            # 1. Edge (u, w) had v as a common neighbor. Now it doesn't. book[u][w] -= 1.
            # 2. Edge (v, w) had u as a common neighbor. Now it doesn't. book[v][w] -= 1.
            # This is correct.
            
            # Apply changes
            for a, b, delta in changes:
                book[a][b] += delta
                book[b][a] += delta
            
            # Remove edge (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove((u, v))
            book[u][v] = 0
            book[v][u] = 0
            
            # 2. Add (x,y)
            # New book size for (x,y)
            common_xy = adj[x] & adj[y]
            new_book_xy = len(common_xy)
            book[x][y] = new_book_xy
            book[y][x] = new_book_xy
            
            # Update neighbors of x and y
            for w in list(adj[x]):
                if w == y:
                    continue
                if w in adj[y]:
                    # w is common neighbor of x and y? No, w is in N(x) and N(y).
                    # Edge (x, w): Common neighbors of x and w included y.
                    # So book[x][w] += 1.
                    book[x][w] += 1
                    book[w][x] += 1
                    # Edge (y, w): Common neighbors of y and w included x.
                    # So book[y][w] += 1.
                    book[y][w] += 1
                    book[w][y] += 1
            
            adj[x].add(y)
            adj[y].add(x)
            edges.add((x, y))
            
            # Calculate new max book
            # We only need to check edges incident to u, v, x, y
            # And the new edge (x,y)
            # And potentially the old max edge if it was affected?
            # Actually, book sizes only increased or decreased for edges incident to u, v, x, y.
            # The global max can only change if:
            # 1. The old max edge was affected and its book size decreased.
            # 2. A new edge (or affected edge) has a book size > old max.
            
            # Let's just recalculate max for affected edges to be safe and fast.
            # Affected edges are those incident to u, v, x, y.
            
            max_candidate = 0
            # Check all edges incident to u, v, x, y
            affected_nodes = {u, v, x, y}
            for node in affected_nodes:
                for neighbor in adj[node]:
                    if (min(node, neighbor), max(node, neighbor)) in edges:
                        if book[node][neighbor] > max_candidate:
                            max_candidate = book[node][neighbor]
            
            # Also check if the previous global max is still valid?
            # If the previous max edge was NOT affected, its book size is unchanged.
            # So new_max = max(previous_max, max_candidate) IF the previous max edge wasn't affected.
            # But if the previous max edge WAS affected, we need to know its new value.
            # It's easier to just keep a global max variable and update it.
            
            # Let's maintain current_max.
            # If the edge with current_max was affected, we might have reduced it.
            # If we added an edge with book > current_max, we increase it.
            
            # To be robust, let's just scan the affected edges and the previous max edge.
            # But we don't track which edge had the max.
            
            # Simple approach: 
            # new_max = max(
            #    max(book[a][b] for a in affected_nodes for b in adj[a] if (min(a,b),max(a,b)) in edges),
            #    # We need to know if the old max is still there.
            # )
            
            # Let's just do a full scan of 401 edges. It's fast enough (401 ops).
            new_max = 0
            for eu, ev in edges:
                if book[eu][ev] > new_max:
                    new_max = book[eu][ev]
            
            delta = new_max - current_max
            
            # Acceptance
            if delta < 0:
                current_max = new_max
                if current_max < best_local_max:
                    best_local_max = current_max
                    best_local_edges = set(edges)
            elif delta == 0:
                # Accept neutral moves to explore
                if random.random() < 0.5:
                    current_max = new_max
            else:
                # Worse move
                if random.random() < math.exp(-delta / T):
                    current_max = new_max
                else:
                    # Revert
                    # Remove (x,y)
                    # This is tedious to write inline. Let's use a helper or just restart if revert is complex?
                    # No, we must revert to continue SA.
                    
                    # Revert Add (x,y)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    edges.remove((x, y))
                    
                    # Revert Book Updates for (x,y) addition
                    # We need to undo the +1s
                    for w in list(adj[x]):
                        if w == y:
                            continue
                        if w in adj[y]:
                            book[x][w] -= 1
                            book[w][x] -= 1
                            book[y][w] -= 1
                            book[w][y] -= 1
                    book[x][y] = 0
                    book[y][x] = 0
                    
                    # Revert Remove (u,v)
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add((u, v))
                    
                    # Revert Book Updates for (u,v) removal
                    # We need to undo the -1s
                    for w in list(adj[u]):
                        if w == v:
                            continue
                        if w in adj[v]:
                            book[u][w] += 1
                            book[w][u] += 1
                            book[v][w] += 1
                            book[w][v] += 1
                    # Restore book[u][v]
                    common_uv = adj[u] & adj[v]
                    book[u][v] = len(common_uv)
                    book[v][u] = len(common_uv)
            
            T *= cooling
            if T < 0.01:
                T = 0.01 # Floor temperature
        
        return best_local_edges, best_local_max

    # Run multiple restarts with different seeds
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(seed)
        
        # Initialize random graph
        random.shuffle(all_pairs)
        init_edges = set(all_pairs[:target_edges])
        
        # Build adj
        init_adj = {i: set() for i in range(n)}
        for u, v in init_edges:
            init_adj[u].add(v)
            init_adj[v].add(u)
        
        # Calculate initial books
        init_book = calculate_books(init_adj, init_edges)
        
        # Run SA for a short duration
        # Divide remaining time among restarts
        # Or just run SA until a small time slice is used.
        # Let's give each restart 1 second? No, 100 restarts * 1s = 100s.
        # Let's give each restart 1.4s.
        
        time_limit_for_restart = min(1.4, deadline - time.time() - 1)
        if time_limit_for_restart <= 0:
            break
            
        res_edges, res_score = simulate_annealing(init_edges, init_adj, init_book, time_limit_for_restart)
        
        if res_score < best_score:
            best_score = res_score
            best_edges = res_edges
            
        if best_score <= 7:
            break

    if best_edges is None:
        # Fallback to random
        random.shuffle(all_pairs)
        best_edges = set(all_pairs[:target_edges])
        
    return [[u, v] for u, v in best_edges]
