import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    n = 31
    # Generate points in PG(2, 5)
    points = []
    
    # Points with z=0
    points.append((1, 0, 0))
    points.append((1, 1, 0))
    points.append((1, 2, 0))
    points.append((1, 3, 0))
    points.append((1, 4, 0))
    points.append((0, 1, 0))
    
    # Points with z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
            
    # Build adjacency matrix A
    A = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
                
    if len(edges) != 90:
        # Fallback if construction fails
        while len(edges) < 91:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j and A[i, j] == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
        return edges

    # Fast C4 count using matrix multiplication
    def count_c4(A):
        A2 = A @ A
        # C4 = sum_{u<v} C(common(u,v), 2)
        # = (sum_{u,v} C(A2[u,v],2) - sum_u C(A2[u,u],2)) / 2
        # But A2[u,u] = degree(u), and for u!=v, A2[u,v] = common neighbors
        # Actually simpler: for each pair u<v, count common neighbors
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = A2[u, v]
                if c >= 2:
                    count += c * (c - 1) // 2
        return count

    # Find best single edge to add (minimizes new C4s)
    A3 = A @ A @ A  # length-3 paths
    
    min_cost = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline - 30:
            break
        for j in range(i + 1, n):
            if A[i, j] == 0:
                cost = A3[i, j]
                if cost < min_cost:
                    min_cost = cost
                    best_edge = (i, j)
    
    if best_edge is None:
        best_edge = (0, 1)
    
    # Add the best edge
    A[best_edge[0], best_edge[1]] = 1
    A[best_edge[1], best_edge[0]] = 1
    edges.append(list(best_edge))
    
    current_c4 = count_c4(A)
    
    # Local search: try edge swaps to reduce C4 count
    # Swap: remove edge (a,b), add edge (c,d) where (c,d) is a non-edge
    # Net change in C4: -C4s_removed_by_removing_ab + C4s_added_by_adding_cd
    # C4s involving edge (a,b) = sum over common neighbors of a and b... 
    # Actually, removing edge (a,b) removes all C4s that use that edge.
    # A C4 uses edge (a,b) if there exist x,y such that a-x-y-b-a is a cycle
    # i.e., x is neighbor of a (not b), y is neighbor of b (not a), and x-y is an edge
    # Number of C4s containing edge (a,b) = |{x in N(a)\{b} : x has a neighbor y in N(b)\{a} and x-y in E}|
    # More simply: for edge (a,b), C4s containing it = sum_{x in N(a)\{b}} |N(x) ∩ N(b)\{a}|
    
    def c4s_containing_edge(A, a, b):
        """Count C4s that use edge (a,b)"""
        count = 0
        for x in range(n):
            if x != a and x != b and A[a, x] == 1:
                # x is neighbor of a, not b (since graph is simple, A[b,x] may be 0 or 1)
                # For each common neighbor y of x and b (y != a, y != b, y != x)
                # where A[x,y] == 1 and A[b,y] == 1
                for y in range(n):
                    if y != a and y != b and y != x and A[x, y] == 1 and A[b, y] == 1:
                        count += 1
        return count

    def c4s_created_by_adding_edge(A, c, d):
        """Count C4s that would be created by adding edge (c,d) to current graph"""
        # This equals the number of length-3 paths between c and d in current graph
        # = A3[c,d] where A3 = A^3
        A3 = A @ A @ A
        return A3[c, d]

    # Precompute degree-2 common neighbor counts for efficiency
    # Local search with edge swaps
    improved = True
    max_iterations = 200
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        best_swap = None
        best_delta = 0  # We want to minimize, so delta = new_c4 - old_c4 < 0
        
        # Try all pairs: remove edge (a,b), add edge (c,d)
        # This is O(m^2 * n^2) which is too slow. Let's be smarter.
        # For each edge to remove, find the best edge to add.
        
        edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i,j] == 1]
        
        for a, b in edge_list:
            if time.time() > deadline - 5:
                break
            
            # C4s removed by removing edge (a,b)
            c4s_removed = c4s_containing_edge(A, a, b)
            
            # Temporarily remove edge (a,b)
            A[a, b] = 0
            A[b, a] = 0
            
            # Find best edge to add
            A3_temp = A @ A @ A
            min_add_cost = float('inf')
            best_add = None
            
            for c in range(n):
                if time.time() > deadline - 5:
                    break
                for d in range(c + 1, n):
                    if A[c, d] == 0:
                        cost = A3_temp[c, d]
                        if cost < min_add_cost:
                            min_add_cost = cost
                            best_add = (c, d)
            
            # Restore edge (a,b)
            A[a, b] = 1
            A[b, a] = 1
            
            if best_add is not None:
                delta = min_add_cost - c4s_removed
                if delta < best_delta:
                    best_delta = delta
                    best_swap = ((a, b), best_add)
        
        if best_swap is not None and best_delta < 0:
            (a, b), (c, d) = best_swap
            # Perform the swap
            A[a, b] = 0
            A[b, a] = 0
            A[c, d] = 1
            A[d, c] = 1
            # Update edges list
            edges.remove([a, b])
            edges.remove([b, a]) if [b, a] in edges else None
            edges.append([c, d])
            current_c4 += best_delta
        else:
            break
    
    return edges
