import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 285.0  # 300s limit with buffer
    
    def time_left():
        return deadline - time.time()
    
    # Build Paley graph on 37 vertices (seed)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n0 = q
    adj = np.zeros((n0, n0), dtype=np.int8)
    for i in range(n0):
        for j in range(i + 1, n0):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Convert to adjacency sets for fast membership testing
    adj_sets = [set() for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            if adj[i, j]:
                adj_sets[i].add(j)
                adj_sets[j].add(i)
    
    def find_k_clique(vertices, k, adj_s):
        """Find a clique of size k in the induced subgraph on vertices."""
        if len(vertices) < k:
            return None
        vertices = list(vertices)
        def backtrack(clique, candidates):
            if len(clique) == k:
                return clique
            for idx, v in enumerate(candidates):
                if all(v in adj_s[u] for u in clique):
                    res = backtrack(clique + [v], candidates[idx+1:])
                    if res:
                        return res
            return None
        return backtrack([], vertices)
    
    def find_k_independent(vertices, k, adj_s):
        """Find an independent set of size k in the induced subgraph on vertices."""
        if len(vertices) < k:
            return None
        vertices = list(vertices)
        def backtrack(indset, candidates):
            if len(indset) == k:
                return indset
            for idx, v in enumerate(candidates):
                if all(v not in adj_s[u] for u in indset):
                    res = backtrack(indset + [v], candidates[idx+1:])
                    if res:
                        return res
            return None
        return backtrack([], vertices)
    
    def check_extension_valid(neighbor_set, n, adj_s):
        """Check if adding a new vertex with neighbor_set is valid.
        Returns (valid, violation_type, violation_vertices)
        """
        # Check: neighbor_set must not contain K4
        if len(neighbor_set) >= 4:
            clique = find_k_clique(neighbor_set, 4, adj_s)
            if clique:
                return False, "clique", clique
        
        # Check: non-neighbors must not contain independent set of size 4
        non_neighbors = [v for v in range(n) if v not in neighbor_set]
        if len(non_neighbors) >= 4:
            ind = find_k_independent(non_neighbors, 4, adj_s)
            if ind:
                return False, "independent", ind
        
        return True, None, None
    
    def try_extend(n, adj_s, adj_mat, time_budget):
        """Try to add one vertex to the graph. Returns (success, new_adj_s, new_adj_mat, new_edges)."""
        end_time = time.time() + time_budget
        
        for restart in range(200):
            if time.time() > end_time:
                break
            
            # Random initial neighbor set
            A = set(random.sample(range(n), random.randint(0, n)))
            
            for step in range(500):
                if time.time() > end_time:
                    break
                
                valid, vtype, vverts = check_extension_valid(A, n, adj_s)
                if valid:
                    # Success! Add the vertex
                    new_n = n + 1
                    new_adj_s = adj_s + [set(A)]
                    for u in A:
                        new_adj_s[u].add(new_n - 1)
                    
                    # Build new adjacency matrix
                    new_adj_mat = np.zeros((new_n, new_n), dtype=np.int8)
                    for i in range(new_n):
                        for j in range(i + 1, new_n):
                            if j in new_adj_s[i]:
                                new_adj_mat[i, j] = 1
                                new_adj_mat[j, i] = 1
                    
                    new_edges = []
                    for u in A:
                        new_edges.append((min(u, new_n - 1), max(u, new_n - 1)))
                    
                    return True, new_adj_s, new_adj_mat, new_edges
                
                # Fix violation
                if vtype == "clique":
                    # Remove a vertex from the clique from A
                    v = random.choice(vverts)
                    A.discard(v)
                elif vtype == "independent":
                    # Add a vertex from the independent set to A
                    v = random.choice(vverts)
                    A.add(v)
                
                # Random perturbation to escape local minima
                if random.random() < 0.3:
                    v = random.randint(0, n - 1)
                    if v in A:
                        A.discard(v)
                    else:
                        A.add(v)
        
        return False, adj_s, adj_mat, []
    
    def fast_validate(n, adj_mat):
        """Validate that the graph has no K5 and no independent 5-set."""
        # Count K5s by checking all 5-tuples is too slow
        # Instead, use the weighted edge count method
        total_k5 = 0
        total_ind5 = 0
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v] == 1:
                    # Count triangles in common neighbors
                    common = [k for k in range(n) if k != u and k != v and adj_mat[u, k] and adj_mat[v, k]]
                    if len(common) >= 3:
                        # Count triangles in common
                        for i in range(len(common)):
                            a = common[i]
                            for j in range(i + 1, len(common)):
                                b = common[j]
                                if adj_mat[a, b]:
                                    for k in range(j + 1, len(common)):
                                        c = common[k]
                                        if adj_mat[a, c] and adj_mat[b, c]:
                                            total_k5 += 1
                else:
                    # Count independent triangles in common non-neighbors
                    common_non = [k for k in range(n) if k != u and k != v and not adj_mat[u, k] and not adj_mat[v, k]]
                    if len(common_non) >= 3:
                        for i in range(len(common_non)):
                            a = common_non[i]
                            for j in range(i + 1, len(common_non)):
                                b = common_non[j]
                                if not adj_mat[a, b]:
                                    for k in range(j + 1, len(common_non)):
                                        c = common_non[k]
                                        if not adj_mat[a, c] and not adj_mat[b, c]:
                                            total_ind5 += 1
        
        k5_count = total_k5 // 10
        ind5_count = total_ind5 // 10
        return k5_count + ind5_count == 0
    
    # Phase 1: Try to extend from 37 up to 42
    current_n = n0
    current_adj_s = adj_sets
    current_adj_mat = adj
    
    # Try extending up to 42
    max_target = 42
    while current_n < max_target:
        if time_left() < 30:
            break
        
        # Allocate time for this extension attempt
        alloc = min(15.0, time_left() / 3)
        if alloc < 5:
            break
        
        success, new_adj_s, new_adj_mat, new_edges = try_extend(
            current_n, current_adj_s, current_adj_mat, alloc
        )
        
        if success:
            current_n = new_adj_s and len(new_adj_s)
            current_adj_s = new_adj_s
            current_adj_mat = new_adj_mat
            print(f"Extended to n={current_n}")
        else:
            # Try a few more times with different random seeds
            for attempt in range(3):
                if time_left() < 30:
                    break
                success, new_adj_s, new_adj_mat, new_edges = try_extend(
                    current_n, current_adj_s, current_adj_mat, min(10.0, time_left() / 4)
                )
                if success:
                    current_n = len(new_adj_s)
                    current_adj_s = new_adj_s
                    current_adj_mat = new_adj_mat
                    print(f"Extended to n={current_n} on retry")
                    break
            if not success:
                break
    
    # Phase 2: If we got to at least 38, try to polish with SA
    if current_n >= 38 and time_left() > 30:
        # Do a quick SA polish to see if we can fix any remaining issues
        # or confirm validity
        pass  # The extension approach should produce valid graphs
    
    # Final validation
    if current_n > n0:
        if fast_validate(current_n, current_adj_mat):
            edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if current_adj_mat[i, j]:
                        edges.append((i, j))
            return current_n, edges
        else:
            # Invalid, fall back to seed
            pass
    
    # Return seed graph
    edges = []
    for i in range(n0):
        for j in range(i + 1, n0):
            if adj[i, j]:
                edges.append((i, j))
    return n0, edges
