import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    deadline = time.time() + 100  # Safety margin for MILP + polish
    
    def is_valid(A):
        """Check if the graph has no K2,3 subgraph."""
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def try_add_edge(A, u, v):
        """Try to add edge (u, v). Returns True if added, False if not."""
        if A[u, v] == 1:
            return False
        
        # Check if adding (u, v) violates the constraint.
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            cn_uw = int(np.dot(A[u, :], A[:, w]))
            if A[v, w] == 1:
                if cn_uw >= 2:
                    return False
            
            # Check pair (v, w)
            cn_vw = int(np.dot(A[v, :], A[:, w]))
            if A[u, w] == 1:
                if cn_vw >= 2:
                    return False
        
        # If we passed all checks, add the edge
        A[u, v] = 1
        A[v, u] = 1
        return True

    def solve_milp():
        """Solve the MILP to find densest K2,3-free graph."""
        # Variables: x_{uv} for each unordered pair (u,v) with u < v
        # Total variables: n*(n-1)/2 = 190
        
        num_vars = n * (n - 1) // 2
        
        # Objective: maximize sum of x_{uv}
        c = np.ones(num_vars)  # Minimize -c^T x, so c = -1 for maximization
        c = -c  # We minimize, so negate for maximization
        
        # Constraints: for each pair (u, v), sum of common neighbors <= 2
        # For each pair (u, v) and each w != u, v:
        # x_{uw} + x_{vw} <= 2 (this is not quite right)
        
        # Actually, the constraint is: for each pair (u, v), the number of w such that 
        # x_{uw}=1 and x_{vw}=1 is at most 2.
        # This is a quadratic constraint, but we can linearize it.
        
        # Alternative approach: use the fact that for each triple (u, v, w), 
        # we need at most 2 of the three edges among them to form a path of length 2.
        # Wait, that's not right either.
        
        # Let me think more carefully. K2,3 means two vertices u,v with three common neighbors.
        # So for each pair (u,v), we need: sum_w [x_{uw} * x_{vw}] <= 2
        # This is quadratic. We can linearize by introducing auxiliary variables.
        
        # For each triple (u, v, w) with u < v and w not in {u,v}:
        # Let y_{uvw} = x_{uw} * x_{vw}
        # Constraints: y_{uvw} <= x_{uw}, y_{uvw} <= x_{vw}, y_{uvw} >= x_{uw} + x_{vw} - 1
        # And: sum_w y_{uvw} <= 2 for each pair (u,v)
        
        # Total variables: 190 (edges) + C(20,2)*18 = 190*18 = 3420 (auxiliary) = 3610
        # This might be too large for quick MILP solving.
        
        # Let's try a different approach: use the fact that the constraint is on 
        # the adjacency matrix. We can use a simpler formulation.
        
        # Actually, let's just use a heuristic approach that's faster.
        
        # Initialize with a good starting point
        A = np.zeros((n, n), dtype=int)
        
        # Start with 5 disjoint K4s
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        
        # Greedy fill
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        if try_add_edge(A, u, v):
                            changed = True
        
        return A

    def local_search(A, budget):
        best_A = A.copy()
        best_edges = count_edges(A)
        
        while time.time() < budget:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if not non_edges:
                break
            
            u, v = random.choice(non_edges)
            if try_add_edge(A, u, v):
                current_edges = count_edges(A)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
            else:
                # Try a swap: remove a random edge, then try to add (u, v)
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edges:
                    break
                ru, rv = random.choice(edges)
                
                # Backup state
                A_backup = A.copy()
                
                # Remove edge (ru, rv)
                A[ru, rv] = 0
                A[rv, ru] = 0
                
                # Try to add (u, v)
                try_add_edge(A, u, v)
                
                # Try to add another random non-edge to gain +1 net
                non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
                for _ in range(3):
                    if not non_edges_now:
                        break
                    x, y = random.choice(non_edges_now)
                    if try_add_edge(A, x, y):
                        non_edges_now.remove((x, y))
                
                current_edges = count_edges(A)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                elif current_edges < best_edges:
                    # Revert if we lost edges
                    A[:] = A_backup
        
        return best_A, best_edges

    best_overall_A = None
    best_overall_edges = 0
    
    # Phase 1: Multiple restarts with greedy + local search
    while time.time() < deadline - 20:  # Reserve 20s for final polish
        A = solve_milp()  # Actually just uses greedy initialization
        
        # Add some random edges to seed
        for _ in range(15):
            u, v = random.sample(range(n), 2)
            try_add_edge(A, u, v)
        
        # Local search with time slice
        slice_end = min(time.time() + 3, deadline - 20)
        A, edges = local_search(A, slice_end)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_A = A.copy()
    
    # Phase 2: Final polish on best solution
    if best_overall_A is not None:
        polish_end = min(time.time() + 15, deadline - 5)
        best_overall_A, best_overall_edges = local_search(best_overall_A.copy(), polish_end)
    
    # Convert best_A to list of edges
    edges_list = []
    if best_overall_A is not None and is_valid(best_overall_A):
        for u in range(n):
            for v in range(u + 1, n):
                if best_overall_A[u, v]:
                    edges_list.append([u, v])
    else:
        # Fallback to seed
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edges_list.append([u, v])
                    
    return edges_list
