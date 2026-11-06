import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 40
    target_edges = 401

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    def init_graph():
        """Initialize with a greedy construction minimizing max book size."""
        A = np.zeros((n, n), dtype=np.int8)
        edges = set()
        
        # Precompute potential edges
        candidates = list(all_pairs)
        random.shuffle(candidates)
        
        # Greedy add: pick edge that minimizes new max book size
        # To speed up, we can just pick random edges first to get a base, then refine
        # But a good start is important.
        
        # Strategy: Start with a 20-regular graph (400 edges) and add 1 edge.
        # A 20-regular graph on 40 vertices can be constructed by taking two 20-cycles or similar.
        # Simpler: Random 20-regular graph.
        
        # Let's try a different init: 
        # Split into 2 sets of 20. 
        # Connect each node in L to 10 random nodes in R. (200 edges, degree 10)
        # Connect each node in R to 10 random nodes in L. (200 edges, degree 10)
        # This is a 10-regular bipartite graph. 200 edges.
        # We need 401. 
        # Add 201 more edges.
        # This seems too sparse.
        
        # Let's stick to the 20-regular idea.
        # Generate a random 20-regular graph on 40 nodes.
        # Use a simple method: 
        # For each node, shuffle the other 39 nodes and pick the first 20.
        # This might create duplicates or self-loops if not careful, but with n=40, k=20, it's dense.
        # Actually, a 20-regular graph on 40 nodes is the complement of a 19-regular graph.
        # Let's just use a random graph with m=401 and run SA from there? 
        # No, random start is bad.
        
        # Better Init:
        # Start with complete bipartite K_{20,20} (400 edges).
        # Book size for any edge is 19.
        # We need to add 1 edge.
        # If we add an edge within L, say (0,1), then:
        #   Book(0,1) = 20 (all R nodes are common neighbors).
        #   Book(0,v) for v in R: Common neighbors are R\{v} (19 nodes) + 1 (node 1). Total 20.
        #   Book(1,v) for v in R: Common neighbors are R\{v} (19 nodes) + 0 (node 0). Total 20.
        #   Other edges remain 19.
        #   Max book = 20.
        
        # Can we do better?
        # If we start with a non-bipartite regular graph?
        # A 20-regular graph on 40 vertices.
        # If it's strongly regular, books are constant.
        # Let's try to construct a 20-regular graph that is NOT bipartite.
        # E.g. Take K_{20,20} and remove a perfect matching (10 edges) and add 10 edges within L?
        # This is getting complicated.
        
        # Let's use a randomized greedy approach for initialization.
        # Start with 0 edges.
        # Add edges one by one.
        # At each step, pick the edge (u,v) that results in the smallest increase in the current max book size.
        # To make it faster, we can pick from a subset of candidates.
        
        # Given time constraints, a random 401-edge graph is a valid start, but SA needs to escape local optima.
        # Let's try: Start with K_{20,20} (400 edges) + 1 random edge.
        # Then use SA to improve.
        
        A = np.zeros((n, n), dtype=np.int8)
        left = list(range(20))
        right = list(range(20, 40))
        
        # Add all bipartite edges
        for u in left:
            for v in right:
                A[u, v] = 1
                A[v, u] = 1
        
        # Add 1 random edge within left or right
        # To minimize initial max book, maybe add an edge that doesn't create too many triangles?
        # In K_{20,20}, any internal edge creates 20 triangles.
        # Is there a way to have max book < 20?
        # Theorem says >= 7.
        # Let's try a different structure.
        # What if we don't use K_{20,20}?
        # What if we use a 20-regular graph that is "more random"?
        
        # Let's generate a random 20-regular graph.
        # Method: 
        # 1. Create a list of 20 copies of each vertex.
        # 2. Shuffle.
        # 3. Pair them up.
        # This creates a 20-regular multigraph. We need simple graph.
        # With n=40, k=20, it's very dense.
        
        # Let's just use the K_{20,20} + 1 edge start. It's fast and valid.
        # The SA will hopefully find a better configuration by swapping edges.
        
        # Pick a random edge to add
        # To make it slightly better, pick an edge that connects vertices with low degree?
        # In K_{20,20}, all degrees are 20.
        
        # Let's add edge (0, 1)
        A[0, 1] = 1
        A[1, 0] = 1
        
        return A

    def get_books(A):
        """Calculate book sizes for all edges."""
        common = A.astype(np.int32) @ A.astype(np.int32).T
        books = {}
        max_b = 0
        sum_b = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    b = int(common[i, j])
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
                    sum_b += b
                else:
                    books[(i, j)] = 0
        return books, max_b, sum_b

    def evaluate_move(A, books, max_b, sum_b, u, v, x, y):
        """
        Evaluate removing (u,v) and adding (x,y).
        Returns (new_max_b, new_sum_b, new_books) or None if invalid.
        """
        if A[u, v] == 0 or A[x, y] == 1:
            return None
        
        # Identify affected edges
        # Removing (u,v):
        # - (u,v) book becomes 0
        # - (u,w) for w in N(u)\{v}: book decreases by 1 if w in N(v)
        # - (v,w) for w in N(v)\{u}: book decreases by 1 if w in N(u)
        
        # Adding (x,y):
        # - (x,y) book becomes common_neighbors(x,y)
        # - (x,w) for w in N(x): book increases by 1 if w in N(y)
        # - (y,w) for w in N(y): book increases by 1 if w in N(x)
        
        # We need to update the 'books' dict.
        # To avoid copying the whole dict, we can calculate the new max/sum directly
        # and return the updated dict if we accept the move.
        # For now, let's calculate the change in max and sum.
        
        # Helper to get neighbors
        # Using A directly is faster than maintaining neighbor lists if we update A.
        
        # Let's simulate the change on a copy of A? 
        # Copying 40x40 int8 array is cheap.
        
        new_A = A.copy()
        new_A[u, v] = 0
        new_A[v, u] = 0
        new_A[x, y] = 1
        new_A[y, x] = 1
        
        # We only need to recompute books for edges incident to u, v, x, y.
        # And (u,v) itself, and (x,y) itself.
        
        # List of edges to update:
        update_set = set()
        update_set.add((min(u,v), max(u,v)))
        update_set.add((min(x,y), max(x,y)))
        
        for w in range(n):
            if w == u or w == v: continue
            if new_A[u, w] == 1:
                update_set.add((min(u,w), max(u,w)))
            if new_A[v, w] == 1:
                update_set.add((min(v,w), max(v,w)))
            
            if w == x or w == y: continue
            if new_A[x, w] == 1:
                update_set.add((min(x,w), max(x,w)))
            if new_A[y, w] == 1:
                update_set.add((min(y,w), max(y,w)))
        
        # Calculate new books for these edges
        new_books = books.copy()
        new_max_b = 0
        new_sum_b = 0
        
        # We need to iterate over all edges to find new max/sum? 
        # That's O(N^2) per move. 1600 edges. 
        # If we do 10000 moves, that's 16M ops. Fast enough in Python? 
        # 16M simple ops might take 1-2 seconds. 
        # 115s budget. 
        # Let's try to optimize.
        
        # Instead of copying books, we can just compute the new max and sum for the affected edges
        # and keep track of the global max/sum?
        # No, because the max might be in an unaffected edge.
        # But if the current max was in an affected edge, and we reduce it, the new max might be the second highest.
        # This requires knowing the distribution.
        
        # Simpler: Just recompute max and sum from the updated books dict.
        # The dict has 780 entries.
        # Iterating 780 entries is fast.
        
        for e in update_set:
            a, b = e
            if new_A[a, b] == 1:
                # Compute common neighbors
                # np.sum(new_A[a, :] * new_A[b, :])
                # This is O(N).
                cnt = int(np.dot(new_A[a, :], new_A[b, :]))
                new_books[e] = cnt
            else:
                new_books[e] = 0
        
        # Now compute new_max and new_sum from new_books
        # This is O(E) = O(1600)
        for b_val in new_books.values():
            if b_val > new_max_b:
                new_max_b = b_val
            new_sum_b += b_val
        
        return new_max_b, new_sum_b, new_books, new_A

    # Main Loop
    A = init_graph()
    books, max_b, sum_b = get_books(A)
    
    best_max_b = max_b
    best_books = books.copy()
    best_A = A.copy()
    best_sum_b = sum_b
    
    T = 10.0
    T_min = 0.01
    alpha = 0.995
    
    # Precompute all pairs for random selection
    # We need to pick (u,v) from edges and (x,y) from non-edges.
    
    current_max_b = max_b
    current_sum_b = sum_b
    current_books = books
    current_A = A
    
    steps = 0
    no_improve_steps = 0
    
    while time.time() < deadline - 5.0:
        if current_max_b < best_max_b or (current_max_b == best_max_b and current_sum_b < best_sum_b):
            best_max_b = current_max_b
            best_sum_b = current_sum_b
            best_books = current_books.copy()
            best_A = current_A.copy()
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            
        # Pick a random edge to remove
        # Get list of edges
        edges_list = [k for k, v in current_books.items() if v > 0]
        if not edges_list:
            break
            
        u, v = random.choice(edges_list)
        
        # Pick a random non-edge to add
        # To speed up, we can pick from a precomputed list of non-edges?
        # Or just pick random pairs and check.
        # With 780 total pairs, 401 edges, 379 non-edges.
        # Random sampling is fine.
        
        x, y = None, None
        attempts = 0
        while attempts < 10:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y: continue
            if x > y: x, y = y, x
            if current_A[x, y] == 0 and (x, y) != (u, v):
                break
            attempts += 1
            
        if x is None:
            continue
            
        # Evaluate move
        result = evaluate_move(current_A, current_books, current_max_b, current_sum_b, u, v, x, y)
        if result is None:
            continue
            
        new_max_b, new_sum_b, new_books, new_A = result
        
        # Acceptance probability
        # Objective: minimize max_b, then sum_b
        # Delta E = (new_max_b - current_max_b) + 0.01 * (new_sum_b - current_sum_b)
        delta_max = new_max_b - current_max_b
        delta_sum = new_sum_b - current_sum_b
        delta_E = delta_max + 0.01 * delta_sum
        
        if delta_E < 0:
            # Accept
            current_A = new_A
            current_books = new_books
            current_max_b = new_max_b
            current_sum_b = new_sum_b
        else:
            # Accept with probability exp(-delta_E / T)
            if random.random() < np.exp(-delta_E / T):
                current_A = new_A
                current_books = new_books
                current_max_b = new_max_b
                current_sum_b = new_sum_b
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally restart with higher T if stuck?
            if no_improve_steps > 1000:
                T = 5.0
                no_improve_steps = 0
                
        steps += 1

    # Extract edges from best_A
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
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
