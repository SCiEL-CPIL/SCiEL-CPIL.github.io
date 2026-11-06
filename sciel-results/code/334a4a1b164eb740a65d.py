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

    def compute_max_book(A):
        A2 = A @ A
        mx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    if b > mx:
                        mx = b
        return mx

    def get_worst_edges(A, A2, k=10):
        """Return up to k edges with the largest book, sorted descending by book."""
        edge_books = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    edge_books.append((b, i, j))
        edge_books.sort(reverse=True)
        return edge_books[:k]

    def check_affected_max(A, A2, vertices):
        """Compute max book among edges incident to any vertex in 'vertices'."""
        mx = 0
        for u in vertices:
            for v in range(n):
                if u != v and A[u, v] == 1:
                    b = int(A2[u, v]) - 1
                    if b > mx:
                        mx = b
        return mx

    random.seed(42)

    # ---- Phase 0: Structured construction from K20,20 with balanced intra-edges ----
    # Start with K20,20 (vertices 0-19 vs 20-39), then replace some cross edges
    # with intra-partition edges to reduce the book of the single intra edge.
    # Key: distribute intra-edges so no vertex has too many intra-neighbors.
    
    A = np.zeros((n, n), dtype=np.int32)
    
    # Start with full K20,20
    for i in range(20):
        for j in range(20, 40):
            A[i, j] = 1
            A[j, i] = 1
    
    # We have 400 edges. Need 1 more. But a single intra edge has book 20.
    # Instead, let's remove some cross edges and add intra edges in a balanced way.
    # Strategy: remove k cross edges, add k intra edges (net +1 edge total means
    # we need to be careful). Actually we need exactly 401 edges.
    # 
    # Better approach: start with K20,20, then do a sequence of "edge swaps"
    # where we remove a cross edge (u,v) with u<20, v>=20 and add an intra edge.
    # This keeps edge count at 400. Then add 1 more intra edge at the end.
    # 
    # The book of a cross edge (u,v) = |N(u) ∩ N(v)|. In K20,20, N(u) = all 20 
    # vertices in other part, N(v) = all 20 in other part. So common neighbors = 0.
    # Book of intra edge (u,v) within part A: N(u) ∩ N(v) = the 20 cross neighbors 
    # that are common. In pure K20,20 + 1 intra, that's 20.
    #
    # To reduce: if we remove cross edge (u, w) where w is a common neighbor, 
    # then book of intra edge (u,v) decreases by 1 for each such removal.
    #
    # Plan: Remove ~7 cross edges incident to vertices involved in intra edges,
    # and add ~7 intra edges distributed across both partitions.
    
    # Let's use a greedy approach: start with K20,20, then iteratively
    # swap cross edges for intra edges to minimize max book.
    
    # First, let's try a specific structured construction:
    # Divide each part into groups and create a "block" structure.
    
    # Alternative: Use a random restart approach with many trials of
    # structured construction + local search.
    
    best_A = None
    best_max = 999
    
    for trial in range(30):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        A = np.zeros((n, n), dtype=np.int32)
        
        # Start with K20,20
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        
        # Now we'll do random edge swaps: remove a cross edge, add an intra edge
        # Do this m times, then add 1 more edge to reach 401.
        # We want to balance the intra-edge distribution.
        
        m = random.randint(5, 15)  # number of swaps
        
        cross_edges = [(i, j) for i in range(20) for j in range(20, 40)]
        random.shuffle(cross_edges)
        
        for swap_idx in range(m):
            if swap_idx >= len(cross_edges):
                break
            
            ru, rv = cross_edges[swap_idx]
            
            # Remove cross edge (ru, rv)
            A[ru, rv] = 0
            A[rv, ru] = 0
            
            # Add an intra edge: pick from same partition
            if random.random() < 0.5:
                # Intra in partition 0 (vertices 0-19)
                a = random.randint(0, 19)
                b = random.randint(0, 19)
                if a == b or A[a, b] == 1:
                    # Try a few times
                    for _ in range(10):
                        a = random.randint(0, 19)
                        b = random.randint(0, 19)
                        if a != b and A[a, b] == 0:
                            break
                    else:
                        # Revert
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        continue
            else:
                # Intra in partition 1 (vertices 20-39)
                a = random.randint(20, 39)
                b = random.randint(20, 39)
                if a == b or A[a, b] == 1:
                    for _ in range(10):
                        a = random.randint(20, 39)
                        b = random.randint(20, 39)
                        if a != b and A[a, b] == 0:
                            break
                    else:
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        continue
            
            A[a, b] = 1
            A[b, a] = 1
        
        # Now we have 400 edges. Add 1 more edge (prefer intra to balance)
        # Find a non-edge that minimizes the increase in max book
        A2 = A @ A
        current_max = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    if b > current_max:
                        current_max = b
        
        # Try adding each possible non-edge and pick the one with smallest max book
        best_add_score = 999
        best_add = None
        all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
        random.shuffle(all_nonedges)
        
        for a, b in all_nonedges[:200]:  # Limit to 200 candidates for speed
            A[a, b] = 1
            A[b, a] = 1
            A2_t = A @ A
            # Quick check: max book among edges incident to a or b
            new_max = 0
            for v in range(n):
                if v != a and A[a, v] == 1:
                    bk = int(A2_t[a, v]) - 1
                    if bk > new_max:
                        new_max = bk
                if v != b and A[b, v] == 1:
                    bk = int(A2_t[b, v]) - 1
                    if bk > new_max:
                        new_max = bk
            A[a, b] = 0
            A[b, a] = 0
            
            if new_max < best_add_score:
                best_add_score = new_max
                best_add = (a, b)
        
        if best_add:
            a, b = best_add
            A[a, b] = 1
            A[b, a] = 1
        
        # Verify edge count
        edge_count = sum(1 for i in range(n) for j in range(i+1, n) if A[i,j] == 1)
        if edge_count != 401:
            continue
        
        # Local search to improve
        A2 = A @ A
        cur_max = compute_max_book(A)
        
        # Quick local search
        for ls_iter in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            A2 = A @ A
            worst = get_worst_edges(A, A2, k=3)
            if not worst or worst[0][0] <= 6:
                break
            
            b_val, u, v = worst[0]
            
            # Try to reduce book of (u,v) by swapping
            improved = False
            common = np.where((A[u] == 1) & (A[v] == 1))[0]
            common = [w for w in common if w != u and w != v]
            
            if not common:
                continue
            
            for w in random.sample(common, min(len(common), 10)):
                for ru, rv in [(u, w), (v, w)]:
                    if A[ru, rv] != 1:
                        continue
                    
                    # Find best replacement edge
                    best_rep = None
                    best_rep_max = b_val
                    
                    # Sample non-edges
                    for _ in range(30):
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        if a == b:
                            continue
                        ka, kb = min(a, b), max(a, b)
                        if A[ka, kb] == 1:
                            continue
                        
                        # Test swap
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1
                        
                        A2_t = A @ A
                        new_max = check_affected_max(A, A2_t, {ru, rv, ka, kb})
                        
                        # Revert
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        A[ka, kb] = 0
                        A[kb, ka] = 0
                        
                        if new_max < best_rep_max:
                            best_rep_max = new_max
                            best_rep = (ka, kb)
                    
                    if best_rep and best_rep_max < b_val:
                        ka, kb = best_rep
                        A[ru, rv] = 0
                        A[rv, ru] = 0
                        A[ka, kb] = 1
                        A[kb, ka] = 1
                        cur_max = best_rep_max
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if cur_max < best_max:
            best_max = cur_max
            best_A = A.copy()
        
        if best_max <= 7:
            break
    
    # If we didn't find anything good, fall back to random graph + annealing
    if best_A is None:
        A = np.zeros((n, n), dtype=np.int32)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        best_A = A
        best_max = compute_max_book(A)
    
    final_A = best_A
    
    # Extract edges
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_A[i, j] == 1:
                edge_list.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edge_list) != target_edges:
        # Fix: add or remove edges as needed
        if len(edge_list) < target_edges:
            all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                           if final_A[i, j] == 0]
            random.shuffle(all_nonedges)
            for u, v in all_nonedges:
                if len(edge_list) >= target_edges:
                    break
                final_A[u, v] = 1
                final_A[v, u] = 1
                edge_list.append([u, v])
        else:
            # Remove extra edges
            edge_list = edge_list[:target_edges]
            final_A = np.zeros((n, n), dtype=np.int32)
            for u, v in edge_list:
                final_A[u, v] = 1
                final_A[v, u] = 1
    
    assert len(edge_list) == target_edges, f"Edge count {len(edge_list)} != {target_edges}"
    
    return edge_list
# EVOLVE-BLOCK-END
