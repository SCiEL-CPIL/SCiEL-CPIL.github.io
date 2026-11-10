import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])

    A = np.zeros((n, n), dtype=np.int32)
    for e in edges:
        A[e[0], e[1]] = 1
        A[e[1], e[0]] = 1

    def count_4cycles(A_mat):
        """Count 4-cycles: (1/2) * sum_{i<j} C((A^2)[i,j], 2)"""
        A2 = A_mat @ A_mat
        iu = np.triu_indices(n, 1)
        vals = A2[iu].astype(np.int64)
        c4_vals = np.where(vals >= 2, vals * (vals - 1) // 2, 0)
        return int(np.sum(c4_vals)) // 2

    adj_sets = [set() for _ in range(n)]
    for e in edges:
        adj_sets[e[0]].add(e[1])
        adj_sets[e[1]].add(e[0])
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)

    base_count = count_4cycles(A)

    # Add the best 91st edge (minimize new C4s)
    best_edge = None
    best_cost = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            cost = 0
            for a in adj_sets[j]:
                if a == i:
                    continue
                for b in adj_sets[i]:
                    if b == j or b == a:
                        continue
                    if a in adj_sets[b]:
                        cost += 1
            if cost < best_cost:
                best_cost = cost
                best_edge = (i, j)

    if best_edge is not None:
        u, v = best_edge
        edges.append([u, v])
        A[u, v] = 1
        A[v, u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edge_set.add((u, v))

    current_count = count_4cycles(A)

    # Fast local search: swap one edge for another
    # Precompute neighbor lists as sorted lists for speed
    def delta_c4_swap(A_mat, u1, v1, u2, v2):
        """Compute change in C4 count when removing (u1,v1) and adding (u2,v2)."""
        # Only pairs involving u1,v1,u2,v2 have their A^2 entries changed
        # A^2[i,j] = sum_k A[i,k]*A[k,j]
        # Removing edge (u1,v1): A^2 changes for pairs where u1 or v1 is involved in the path
        # Adding edge (u2,v2): similarly
        # 
        # For efficiency, just recompute A^2 for affected rows/cols
        # Actually with n=31, full recompute is fast enough if we avoid Python loops
        
        A_new = A_mat.copy()
        A_new[u1, v1] = 0
        A_new[v1, u1] = 0
        A_new[u2, v2] = 1
        A_new[v2, u2] = 1
        A2_new = A_new @ A_new
        iu = np.triu_indices(n, 1)
        vals_new = A2_new[iu].astype(np.int64)
        c4_new = np.where(vals_new >= 2, vals_new * (vals_new - 1) // 2, 0)
        new_count = int(np.sum(c4_new)) // 2
        return new_count - count_4cycles(A_mat)

    # Better: compute delta analytically
    def delta_c4_analytic(A_mat, A2_mat, u1, v1, u2, v2):
        """
        Compute change in C4 count when removing edge (u1,v1) and adding edge (u2,v2).
        
        C4 count = (1/2) * sum_{i<j} C(A2[i,j], 2)
        
        When we change A, A2 changes. Let's compute which A2 entries change.
        A2[i,j] = sum_k A[i,k] * A[k,j]
        
        Removing (u1,v1): affects A2[i,j] for i in {u1} ∪ N(u1) and j in {v1} ∪ N(v1) etc.
        Actually: A2[i,j] changes if the path i-k-j uses edge (u1,v1) or (v1,u1).
        Path i-u1-v1-j: i~u1, v1~j → A2[i,j] decreases by 1 if i~u1 and v1~j
        Path i-v1-u1-j: i~v1, u1~j → A2[i,j] decreases by 1 if i~v1 and u1~j
        
        Similarly for adding (u2,v2): A2[i,j] increases by 1 if i~u2 and v2~j, or i~v2 and u2~j.
        
        So the affected pairs are those where at least one of the path endpoints is connected
        to u1/v1 (for removal) or u2/v2 (for addition).
        
        Let's just compute the new A2 for affected entries.
        """
        # Get neighbors
        N_u1 = set(np.where(A_mat[u1] > 0)[0])
        N_v1 = set(np.where(A_mat[v1] > 0)[0])
        N_u2 = set(np.where(A_mat[u2] > 0)[0])
        N_v2 = set(np.where(A_mat[v2] > 0)[0])
        
        # Pairs whose A2 value changes:
        # Removal of (u1,v1): A2[i,j] changes if (i in N_u1 and j in N_v1) or (i in N_v1 and j in N_u1)
        #   (excluding i=j cases which are diagonal)
        # Addition of (u2,v2): A2[i,j] changes if (i in N_u2 and j in N_v2) or (i in N_v2 and j in N_u2)
        
        # Build set of affected (i,j) pairs with i<j
        affected = set()
        for i in N_u1:
            for j in N_v1:
                if i != j:
                    affected.add((min(i,j), max(i,j)))
        for i in N_v1:
            for j in N_u1:
                if i != j:
                    affected.add((min(i,j), max(i,j)))
        for i in N_u2:
            for j in N_v2:
                if i != j:
                    affected.add((min(i,j), max(i,j)))
        for i in N_v2:
            for j in N_u2:
                if i != j:
                    affected.add((min(i,j), max(i,j)))
        
        # Also the pairs (u1,v1) and (u2,v2) themselves might be in affected set
        # But their A2 values also change (they were 0 or nonzero before/after)
        
        # Compute old and new contribution from affected pairs
        old_contrib = 0
        new_contrib = 0
        for (i, j) in affected:
            old_val = A2_mat[i, j]
            # Compute new A2[i,j]
            new_val = old_val
            # Removal effects
            if (A_mat[i, u1] > 0 and A_mat[j, v1] > 0) or (A_mat[i, v1] > 0 and A_mat[j, u1] > 0):
                new_val -= 1
            # Addition effects
            if (A_mat[i, u2] > 0 and A_mat[j, v2] > 0) or (A_mat[i, v2] > 0 and A_mat[j, u2] > 0):
                new_val += 1
            
            if old_val >= 2:
                old_contrib += old_val * (old_val - 1) // 2
            if new_val >= 2:
                new_contrib += new_val * (new_val - 1) // 2
        
        return (new_contrib - old_contrib) // 2

    # Local search
    improved = True
    iteration = 0
    max_iterations = 200
    no_improve = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 20:
        improved = False
        iteration += 1
        
        current_count = count_4cycles(A)
        A2 = A @ A
        
        edge_list = list(edge_set)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        best_swap = None
        best_delta = 0  # want most negative
        
        for e1 in edge_list:
            if time.time() > deadline - 20:
                break
            u1, v1 = e1
            for e2 in non_edges:
                u2, v2 = e2
                delta = delta_c4_analytic(A, A2, u1, v1, u2, v2)
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            adj_sets[u1].discard(v1)
            adj_sets[v1].discard(u1)
            adj_sets[u2].add(v2)
            adj_sets[v2].add(u2)
            
            edge_set.discard(e1)
            edge_set.add(e2)
            
            edges = [list(e) for e in edge_set]
            improved = True
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 2:
                break

    # Final validation
    final_edges = []
    seen = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            final_edges.append([e[0], e[1]])
            seen.add(key)

    if len(final_edges) > 91:
        final_edges = final_edges[:91]
    elif len(final_edges) < 91:
        for i in range(n):
            for j in range(i+1, n):
                if len(final_edges) >= 91:
                    break
                if (i, j) not in seen:
                    final_edges.append([i, j])
                    seen.add((i, j))
            if len(final_edges) >= 91:
                break

    return final_edges[:91]
