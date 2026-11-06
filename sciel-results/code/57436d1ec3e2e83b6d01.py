"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Use numpy adjacency matrix for fast book computation
    # book(e) = |N(u) ∩ N(v)| = (A @ A)[u,v] - (1 if edge exists else 0)
    # Actually, for edge (u,v), book = number of w such that w~u and w~v
    # This equals (A^2)[u,v] - 1 (since A[u,v]=1, the path u-v contributes 1)
    
    def compute_all_books(A):
        """Compute book size for all edges using matrix multiplication."""
        A2 = A @ A
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    books[(i,j)] = int(A2[i,j]) - 1
        return books
    
    def max_book_from_books(books):
        """Return the maximum book size."""
        mx = 0
        for e, b in books.items():
            if b > mx:
                mx = b
        return mx
    
    def compute_max_book(A):
        """Compute max book size from adjacency matrix."""
        A2 = A @ A
        mx = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    b = int(A2[i,j]) - 1
                    if b > mx:
                        mx = b
        return mx
    
    def get_books_for_affected(A, vertices):
        """Compute books for edges incident to given vertices."""
        A2 = A @ A
        books = {}
        for u in vertices:
            for v in range(n):
                if u != v and A[u,v] == 1:
                    key = (min(u,v), max(u,v))
                    books[key] = int(A2[u,v]) - 1
        return books
    
    # Initial construction: K20,20 plus one intra-part edge
    # This gives a good starting point with max book = 19 (for the intra edge)
    # or 20 (for cross edges? let's check)
    # For K20,20: cross edge (u in 0..19, v in 20..39) has book = |N(u)∩N(v)| = 19 (the other 19 in u's part)
    # Wait, N(u) for u in left = all 20 right vertices. N(v) for v in right = all 20 left vertices.
    # N(u) ∩ N(v) = empty? No, u and v are in different parts.
    # Common neighbors of (u,v) where u in left, v in right: none, since left vertices only connect to right and vice versa.
    # So book = 0 for all cross edges in pure K20,20.
    # For the intra edge (0,1) where both in left: N(0) = {20..39} ∪ {1}, N(1) = {20..39} ∪ {0}
    # Common neighbors = {20..39}, size 20. So book = 20.
    # Max book = 20.
    
    # We want to minimize max book. The theoretical lower bound is 40/6 ≈ 6.67, so >= 7.
    # Let's start from K20,20 + one edge and use simulated annealing.
    
    # Build initial adjacency matrix
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(20):
        for j in range(20, 40):
            A[i,j] = 1
            A[j,i] = 1
    # Add one intra edge
    A[0,1] = 1
    A[1,0] = 1
    
    # Verify edge count
    edge_count = int(np.sum(A)) // 2
    assert edge_count == target_edges, f"Edge count {edge_count} != {target_edges}"
    
    # Compute initial max book
    current_max = compute_max_book(A)
    best_A = A.copy()
    best_max = current_max
    
    # Simulated annealing
    # Move: remove one edge, add one different edge
    # Temperature schedule: start high, cool down
    
    T_initial = 5.0
    T_final = 0.1
    cooling_rate = 0.9995
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    
    T = T_initial
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < DEADLINE:
        if no_improve_count > max_no_improve and T < 1.0:
            break
            
        # Pick a random edge to remove
        # Bias towards edges with high book size
        A2 = A @ A
        # Build list of (book_size, edge) for sampling
        edge_books = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    b = int(A2[i,j]) - 1
                    edge_books.append((b, i, j))
        
        if not edge_books:
            break
        
        # Sample an edge to remove, weighted by book size (higher book = more likely to be removed)
        # But we also want to explore, so use a mix
        if random.random() < 0.7:
            # Sample proportional to book size + 1
            weights = [b + 1 for b, _, _ in edge_books]
            total_w = sum(weights)
            r = random.random() * total_w
            cumulative = 0
            for idx, (b, i, j) in enumerate(edge_books):
                cumulative += weights[idx]
                if r <= cumulative:
                    remove_u, remove_v = i, j
                    remove_book = b
                    break
            else:
                remove_u, remove_v, remove_book = edge_books[-1][1], edge_books[-1][2], edge_books[-1][0]
        else:
            # Random edge
            b, remove_u, remove_v = random.choice(edge_books)
            remove_book = b
        
        # Pick a random non-edge to add
        # Try a few candidates
        added = False
        for _ in range(30):
            a = random.randint(0, n-1)
            b_idx = random.randint(0, n-1)
            if a == b_idx:
                continue
            ka, kb = min(a, b_idx), max(a, b_idx)
            if A[ka, kb] == 1:
                continue
            # This is a valid non-edge to add
            
            # Apply the move: remove (remove_u, remove_v), add (ka, kb)
            A[remove_u, remove_v] = 0
            A[remove_v, remove_u] = 0
            A[ka, kb] = 1
            A[kb, ka] = 1
            
            # Compute new max book - only need to check affected edges
            # Affected vertices: remove_u, remove_v, ka, kb
            affected_vertices = {remove_u, remove_v, ka, kb}
            A2_new = A @ A
            
            new_max = 0
            # Check all edges incident to affected vertices
            for u in affected_vertices:
                for v in range(n):
                    if u != v and A[u,v] == 1:
                        key = (min(u,v), max(u,v))
                        b_val = int(A2_new[u,v]) - 1
                        if b_val > new_max:
                            new_max = b_val
            
            # Also check if the overall max might be higher elsewhere
            # For safety, do a full check every 100 iterations
            if _ == 29 or random.random() < 0.1:
                full_max = compute_max_book(A)
                if full_max > new_max:
                    new_max = full_max
            
            # Simulated annealing acceptance
            delta = new_max - current_max
            if delta <= 0:
                # Accept
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_A = A.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                added = True
                break
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    current_max = new_max
                    no_improve_count += 1
                    added = True
                    break
                else:
                    # Revert
                    A[remove_u, remove_v] = 1
                    A[remove_v, remove_u] = 1
                    A[ka, kb] = 0
                    A[kb, ka] = 0
        
        if not added:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    # Final phase: local search to polish
    if time.time() - start_time < DEADLINE - 5:
        A = best_A.copy()
        current_max = compute_max_book(A)
        
        # Greedy local search: try to improve the worst edge
        for iteration in range(10000):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            A2 = A @ A
            # Find the worst edge
            worst_edge = None
            worst_book = -1
            for i in range(n):
                for j in range(i+1, n):
                    if A[i,j] == 1:
                        b = int(A2[i,j]) - 1
                        if b > worst_book:
                            worst_book = b
                            worst_edge = (i, j)
            
            if worst_book <= 0:
                break
            
            u, v = worst_edge
            improved = False
            
            # Try removing one of the edges in the book
            # Common neighbors of (u,v)
            common = []
            for w in range(n):
                if w != u and w != v and A[u,w] == 1 and A[v,w] == 1:
                    common.append(w)
            
            for w in common[:5]:  # Try up to 5 common neighbors
                # Try removing (u,w) or (v,w)
                for remove_edge in [(u, w), (v, w)]:
                    ru, rv = min(remove_edge), max(remove_edge)
                    if A[ru, rv] != 1:
                        continue
                    
                    # Try adding a random non-edge
                    for _ in range(20):
                        a = random.randint(0, n-1)
                        b_idx = random.randint(0, n-1)
                        if a == b_idx:
                            continue
                        ka, kb = min(a, b_idx), max(a, b_idx)
                        if A[ka, kb] == 1:
                            continue
                        
                        # Apply
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1
                        
                        A2_new = A @ A
                        new_max = compute_max_book(A)
                        
                        if new_max < worst_book:
                            # Accept
                            current_max = new_max
                            if new_max < best_max:
                                best_max = new_max
                                best_A = A.copy()
                            improved = True
                            break
                        else:
                            # Revert
                            A[ru, rv] = 1
                            A[rv, ru] = 1
                            A[ka, kb] = 0
                            A[kb, ka] = 0
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i,j] == 1:
                edges.append([i, j])
    
    # Verify
    assert len(edges) == target_edges, f"Got {len(edges)} edges, expected {target_edges}"
    assert len(set((min(e[0],e[1]), max(e[0],e[1])) for e in edges)) == target_edges, "Duplicate edges"
    
    return edges
# EVOLVE-BLOCK-END
