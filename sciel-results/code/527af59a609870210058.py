"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct a graph with 31 vertices and 91 edges that minimizes the number of C4s.
    
    Strategy: 
    1. Try multiple polarity-based constructions to find a 90-edge C4-free base
    2. Add the edge that creates the fewest new C4s
    3. Perform local search (edge swaps) to minimize C4 count further
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin (150s limit)
    
    n = 31
    q = 5
    
    def count_c4s(adj):
        """Count number of C4 subgraphs using vectorized operations."""
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = np.dot(adj[u], adj[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    def delta_c4_swap(A, u1, v1, u2, v2):
        """
        Compute the change in C4 count when removing edge (u1,v1) and adding edge (u2,v2).
        Uses the fact that C4s involving the changed edges can be computed incrementally.
        """
        # A C4 containing edge (a,b) has form a-x-b-y-a or a-x-b-y-z... 
        # Actually, a C4 containing edge (u,v) must have two common neighbors x,y of u and v
        # such that x-y is an edge. So #C4s containing (u,v) = #pairs of common neighbors of u,v
        # that are themselves adjacent.
        
        # When we remove (u1,v1): we lose all C4s that use this edge
        # When we add (u2,v2): we gain all C4s that use this edge
        
        # C4s containing edge (a,b): for each pair (x,y) where x,y are common neighbors of a,b
        # and x~y, we have a C4 a-x-b-y-a (or a-y-b-x-a, same C4).
        # So count = number of edges in the common-neighborhood of (a,b).
        
        def c4s_containing_edge(a, b):
            """Count C4s that contain edge (a,b)."""
            common = np.where((A[a] == 1) & (A[b] == 1))[0]
            common = common[common != a]
            common = common[common != b]
            if len(common) < 2:
                return 0
            # Count edges among common neighbors
            count = 0
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if A[common[i], common[j]] == 1:
                        count += 1
            return count
        
        # But we need to be careful: when removing (u1,v1), the C4s lost are those containing (u1,v1)
        # AND also potentially C4s that don't contain (u1,v1) but are affected... no, only C4s
        # containing (u1,v1) are affected by its removal.
        # Similarly, only C4s containing (u2,v2) are gained by its addition.
        
        # However, there's a subtlety: if {u1,v1} and {u2,v2} share vertices, the interactions
        # get complex. Let's handle the simple case where they're disjoint, and for overlapping
        # cases, just do a full recount (which is expensive but rare).
        
        if len({u1, v1, u2, v2}) < 4:
            # Overlapping edges - do full recount
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            new_c4s = count_c4s(A)
            A[u1, v1] = 1
            A[v1, u1] = 1
            A[u2, v2] = 0
            A[v2, u2] = 0
            return new_c4s - count_c4s(A)
        
        # Disjoint edges: delta = -c4s_containing(u1,v1) + c4s_containing(u2,v2)
        # But c4s_containing uses current A, which still has (u1,v1) present and (u2,v2) absent.
        # For removal: C4s containing (u1,v1) in current graph = c4s_containing_edge(u1, v1)
        # For addition: C4s containing (u2,v2) after adding = c4s_containing_edge(u2, v2) 
        #   but computed with (u2,v2) present. Since (u2,v2) is not present now, we need to
        #   compute what it would be. The common neighbors of u2,v2 don't change when we add (u2,v2),
        #   so c4s_containing_edge(u2,v2) with (u2,v2) added equals the same computation but
        #   excluding u2,v2 from common neighbors (which we already do).
        
        lost = c4s_containing_edge(u1, v1)
        gained = c4s_containing_edge(u2, v2)
        
        return gained - lost
    
    def local_search(A, deadline, max_iterations=5000):
        """Perform local search with edge swaps to minimize C4 count."""
        current_c4s = count_c4s(A)
        best_c4s = current_c4s
        best_A = A.copy()
        
        random.seed(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Get current edges and non-edges
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
            non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
            
            if not edge_list or not non_edge_list:
                break
            
            # Shuffle for randomness
            random.shuffle(edge_list)
            random.shuffle(non_edge_list)
            
            improved = False
            # Try a larger subset of swaps with incremental delta computation
            max_e_out = min(60, len(edge_list))
            max_e_in = min(60, len(non_edge_list))
            
            for e_out in edge_list[:max_e_out]:
                if time.time() > deadline - 5:
                    break
                for e_in in non_edge_list[:max_e_in]:
                    if e_out == e_in:
                        continue
                    
                    u1, v1 = e_out
                    u2, v2 = e_in
                    
                    # Compute delta using incremental method
                    delta = delta_c4_swap(A, u1, v1, u2, v2)
                    new_c4s = current_c4s + delta
                    
                    if new_c4s < best_c4s:
                        # Accept the swap
                        A[u1, v1] = 0
                        A[v1, u1] = 0
                        A[u2, v2] = 1
                        A[v2, u2] = 1
                        current_c4s = new_c4s
                        best_c4s = new_c4s
                        best_A = A.copy()
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try more random swaps
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    e_out = random.choice(edge_list)
                    e_in = random.choice(non_edge_list)
                    if e_out == e_in:
                        continue
                    
                    u1, v1 = e_out
                    u2, v2 = e_in
                    
                    delta = delta_c4_swap(A, u1, v1, u2, v2)
                    new_c4s = current_c4s + delta
                    
                    if new_c4s < best_c4s:
                        A[u1, v1] = 0
                        A[v1, u1] = 0
                        A[u2, v2] = 1
                        A[v2, u2] = 1
                        current_c4s = new_c4s
                        best_c4s = new_c4s
                        best_A = A.copy()
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_A, best_c4s
    
    # Define points in PG(2, 5)
    points = []
    for t in range(5):
        points.append((1, t, 0))
    points.append((0, 1, 0))
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    
    assert len(points) == 31
    
    # Try multiple bilinear forms to find a 90-edge C4-free graph
    # Standard polarity: B(u,v) = u0*v0 + u1*v1 + u2*v2
    # We can also try other symmetric bilinear forms
    
    best_A = None
    best_c4s = float('inf')
    
    # Try the standard polarity
    bilinear_forms = [
        lambda u, v: (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q,  # standard
        lambda u, v: (u[0]*v[0] + 2*u[1]*v[1] + 3*u[2]*v[2]) % q,  # scaled
        lambda u, v: (u[0]*v[0] + u[1]*v[2] + u[2]*v[1]) % q,  # symmetric off-diagonal
    ]
    
    for bilinear in bilinear_forms:
        if time.time() > deadline - 30:
            break
        
        A = np.zeros((n, n), dtype=np.int8)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ui, vi, wi = points[i]
                uj, vj, wj = points[j]
                val = bilinear((ui, vi, wi), (uj, vj, wj))
                if val == 0:
                    A[i, j] = 1
        
        # Make symmetric and remove self-loops
        A = (A + A.T) // 2
        np.fill_diagonal(A, 0)
        
        edge_count = A.sum() // 2
        
        if edge_count == 90:
            # Find the non-edge that creates the fewest new C4s
            A3 = A @ A @ A
            
            min_new_c4s = float('inf')
            best_edge = None
            
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        c4s = A3[u, v]
                        if c4s < min_new_c4s:
                            min_new_c4s = c4s
                            best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                A[u, v] = 1
                A[v, u] = 1
                
                # Now do local search to minimize C4s
                A_optimized, c4s = local_search(A, deadline)
                
                if c4s < best_c4s:
                    best_c4s = c4s
                    best_A = A_optimized
        
        elif edge_count < 90:
            # Add edges until we have 90, then add one more
            # This is a fallback; try to add edges that minimize C4s
            A3 = A @ A @ A
            added = 0
            while edge_count < 90 and added < 10:
                min_new_c4s = float('inf')
                best_edge = None
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            c4s = A3[u, v]
                            if c4s < min_new_c4s:
                                min_new_c4s = c4s
                                best_edge = (u, v)
                if best_edge is None:
                    break
                u, v = best_edge
                A[u, v] = 1
                A[v, u] = 1
                edge_count += 1
                added += 1
                A3 = A @ A @ A
            
            if edge_count == 90:
                # Add one more edge
                A3 = A @ A @ A
                min_new_c4s = float('inf')
                best_edge = None
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            c4s = A3[u, v]
                            if c4s < min_new_c4s:
                                min_new_c4s = c4s
                                best_edge = (u, v)
                
                if best_edge is not None:
                    u, v = best_edge
                    A[u, v] = 1
                    A[v, u] = 1
                    
                    A_optimized, c4s = local_search(A, deadline)
                    
                    if c4s < best_c4s:
                        best_c4s = c4s
                        best_A = A_optimized
    
    # If no valid graph found, use fallback
    if best_A is None:
        # Fallback: random graph + local search
        A = np.zeros((n, n), dtype=np.int8)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges[:91]:
            A[u, v] = 1
            A[v, u] = 1
        
        best_A, best_c4s = local_search(A, deadline)
    
    # Build edge list from best_A
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    for e in edges:
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30
        assert e[0] != e[1]
    
    edge_set = set()
    for e in edges:
        key = tuple(sorted(e))
        assert key not in edge_set, f"Duplicate edge: {e}"
        edge_set.add(key)
    
    return edges

# EVOLVE-BLOCK-END
