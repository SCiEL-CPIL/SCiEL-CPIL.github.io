import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    def is_valid(A):
        """Check if the graph has no K2,3 subgraph."""
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def get_neighbors(A):
        """Return list of neighbor sets for each vertex."""
        return [set(np.where(A[i] == 1)[0].tolist()) for i in range(n)]

    def try_add_edge(A, u, v, nbrs):
        """Try to add edge (u, v). Returns True if added, False if not. Updates nbrs."""
        if A[u, v] == 1:
            return False
        
        # Check if adding (u, v) violates the constraint.
        # For each w, if w is adjacent to both u and v, then cn(u,w) and cn(v,w) increase by 1.
        # We need cn(u,w) <= 2 and cn(v,w) <= 2 after addition.
        # So before addition, we need cn(u,w) <= 1 and cn(v,w) <= 1 for all common neighbors w of u and v.
        
        common = nbrs[u] & nbrs[v]
        for w in common:
            # cn(u, w) = |nbrs[u] & nbrs[w]|
            # After adding (u,v), if v in nbrs[w] (which is true since w in common), 
            # cn(u,w) increases by 1 if v is in nbrs[w]... 
            # Actually: cn(u,w) = number of common neighbors of u and w.
            # Adding edge (u,v): for pair (u,w), common neighbors increase if v is neighbor of w.
            # v is neighbor of w iff w is in nbrs[v], which is true since w in common.
            # So cn(u,w) increases by 1.
            # We need cn(u,w) + 1 <= 2, i.e., cn(u,w) <= 1.
            cn_uw = len(nbrs[u] & nbrs[w])
            if cn_uw >= 2:
                return False
            
            cn_vw = len(nbrs[v] & nbrs[w])
            if cn_vw >= 2:
                return False
        
        # Also need to check: for pairs (u, w) where w is NOT in common but w is neighbor of u,
        # adding (u,v) doesn't affect cn(u,w) unless v is neighbor of w.
        # If w is neighbor of u but not v, and v is not neighbor of w, cn(u,w) unchanged.
        # If w is neighbor of u but not v, and v IS neighbor of w, then w is in common. Contradiction.
        # So only common neighbors matter.
        
        # Add the edge
        A[u, v] = 1
        A[v, u] = 1
        nbrs[u].add(v)
        nbrs[v].add(u)
        return True

    def try_remove_edge(A, u, v, nbrs):
        """Remove edge (u, v). Updates nbrs."""
        A[u, v] = 0
        A[v, u] = 0
        nbrs[u].discard(v)
        nbrs[v].discard(u)

    def initialize():
        # Start with 5 disjoint K4s (30 edges). This is a valid K2,3-free graph.
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        return A

    def greedy_fill(A):
        nbrs = get_neighbors(A)
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        if try_add_edge(A, u, v, nbrs):
                            changed = True
        return A, nbrs

    def local_search(A, nbrs, budget):
        best_A = A.copy()
        best_nbrs = [s.copy() for s in nbrs]
        best_edges = count_edges(A)
        
        while time.time() < budget:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if not non_edges:
                break
            
            u, v = random.choice(non_edges)
            if try_add_edge(A, u, v, nbrs):
                current_edges = count_edges(A)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                    best_nbrs = [s.copy() for s in nbrs]
            else:
                # Try a swap: remove a random edge, then try to add (u, v)
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edges:
                    break
                ru, rv = random.choice(edges)
                
                # Backup state
                A_backup = A.copy()
                nbrs_backup = [s.copy() for s in nbrs]
                
                # Remove edge (ru, rv)
                try_remove_edge(A, ru, rv, nbrs)
                
                # Try to add (u, v)
                success_1 = try_add_edge(A, u, v, nbrs)
                
                # Try to add another random non-edge to gain +1 net
                non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
                random.shuffle(non_edges_now)
                for x, y in non_edges_now:
                    if try_add_edge(A, x, y, nbrs):
                        break
                
                current_edges = count_edges(A)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                    best_nbrs = [s.copy() for s in nbrs]
                elif current_edges < best_edges:
                    # Revert if we lost edges
                    A[:] = A_backup
                    for i in range(n):
                        nbrs[i] = nbrs_backup[i]
                # If current_edges == best_edges, keep it (diversity)
        
        return best_A, best_nbrs, best_edges

    def polish(A, nbrs, budget):
        """Final high-precision polish: try all possible single-edge additions."""
        best_A = A.copy()
        best_nbrs = [s.copy() for s in nbrs]
        best_edges = count_edges(A)
        
        while time.time() < budget:
            improved = False
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            for u, v in non_edges:
                if try_add_edge(A, u, v, nbrs):
                    improved = True
                    current_edges = count_edges(A)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_A = A.copy()
                        best_nbrs = [s.copy() for s in nbrs]
            if not improved:
                break
        
        return best_A, best_nbrs, best_edges

    best_overall_A = None
    best_overall_nbrs = None
    best_overall_edges = 0
    
    # Run multiple restarts
    while time.time() < deadline - 10:  # Leave 10s for final polish
        A = initialize()
        
        # Add some random edges to seed
        nbrs = get_neighbors(A)
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            try_add_edge(A, u, v, nbrs)
        
        A, nbrs = greedy_fill(A)
        
        # Local search with time slice
        slice_end = min(time.time() + 5, deadline - 10)
        A, nbrs, edges = local_search(A, nbrs, slice_end)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_A = A.copy()
            best_overall_nbrs = [s.copy() for s in nbrs]
            
            # Verify best result
            if not is_valid(best_overall_A):
                print("Warning: Invalid graph found in local search, reverting.")
                best_overall_A = None
                best_overall_nbrs = None
                best_overall_edges = 0

    # Final polish pass
    if best_overall_A is not None and best_overall_nbrs is not None:
        polish_end = time.time() + 8
        best_overall_A, best_overall_nbrs, best_overall_edges = polish(
            best_overall_A, best_overall_nbrs, polish_end
        )

    # Convert best_A to list of edges
    edges_list = []
    if best_overall_A is not None and is_valid(best_overall_A):
        for u in range(n):
            for v in range(u + 1, n):
                if best_overall_A[u, v]:
                    edges_list.append([u, v])
    else:
        # Fallback to seed
        A = initialize()
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edges_list.append([u, v])
                    
    return edges_list
