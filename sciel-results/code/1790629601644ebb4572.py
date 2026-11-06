import numpy as np
import time
import random

def construct():
    n = 20
    start_time = time.time()
    timeout = 110
    
    best_edges = 0
    best_A = None
    
    # Helper to get list of edges from adjacency matrix
    def get_edges(A):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    edges.append([u, v])
        return edges

    # Helper to check validity and count violations
    # Returns (is_valid, list_of_violating_pairs)
    def check_validity(A):
        C = A @ A
        # C[u, v] is number of common neighbors
        # We need C[u, v] <= 2 for all u != v
        violations = []
        for u in range(n):
            for v in range(u + 1, n):
                if C[u, v] > 2:
                    violations.append((u, v, C[u, v]))
        return (len(violations) == 0), violations

    # Helper to repair graph by removing edges
    def repair(A):
        # Copy A
        A = A.copy()
        C = A @ A
        # We remove edges until no pair has > 2 common neighbors
        # Max iterations to prevent infinite loops
        max_iter = 1000
        for _ in range(max_iter):
            # Find a violating pair
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if C[u, v] > 2:
                        # This pair (u, v) has > 2 common neighbors
                        # Find common neighbors
                        common = np.where((A[u, :] == 1) & (A[v, :] == 1))[0]
                        # Pick one common neighbor w
                        w = random.choice(common.tolist())
                        # Remove edge (u, w) or (v, w) randomly
                        if random.random() < 0.5:
                            if A[u, w] == 1:
                                # Remove (u, w)
                                A[u, w] = 0
                                A[w, u] = 0
                                # Update C
                                # C[x, w] decreases by 1 for all x in N(u)
                                # C[w, x] decreases by 1 for all x in N(u)
                                # Note: C[u, v] decreases by 1
                                for x in range(n):
                                    if A[u, x] == 1:
                                        C[x, w] -= 1
                                        C[w, x] -= 1
                                # Also C[u, v] specifically:
                                # u is no longer connected to w, so w is no longer a common neighbor of u and v?
                                # Wait, w was a common neighbor of u and v.
                                # Removing (u, w) means w is no longer in N(u).
                                # So w is no longer a common neighbor of u and any other node?
                                # C[u, y] counts z in N(u) \cap N(y).
                                # If we remove (u, w), then for any y, if w was in N(u) \cap N(y), C[u, y] decreases.
                                # So C[u, y] -= 1 for all y in N(w) (excluding u? no, if y is in N(w) and y != u, w was in N(u) \cap N(y) only if u-w and y-w. Yes.)
                                # Actually, simpler:
                                # C = A @ A.
                                # Change in C due to removing (u, w):
                                # C[x, y] changes if x or y is u or w.
                                # C[u, y] = sum_k A[u, k] A[k, y].
                                # A[u, w] became 0.
                                # So C[u, y] decreases by A[w, y] for all y.
                                # C[x, w] decreases by A[x, u] for all x.
                                # C[u, w] decreases by A[u, u] (0) + ... wait.
                                # C[u, w] = sum_k A[u, k] A[k, w].
                                # Term k=w: A[u, w] A[w, w] = 0.
                                # Term k=u: A[u, u] A[u, w] = 0.
                                # So C[u, w] doesn't change directly from the diagonal terms?
                                # Let's just recalculate C for affected rows/cols or just recalculate whole C if n is small.
                                # n=20, C is 20x20. Recalculating A @ A is 20*20*20 = 8000 ops. Very fast.
                                # Given the complexity of incremental updates and risk of bugs, and n is small,
                                # let's just recalculate C after each removal.
                                A = A.copy() # Ensure local view
                                C = A @ A
                                found = True
                                break
                        else:
                            if A[v, w] == 1:
                                A[v, w] = 0
                                A[w, v] = 0
                                A = A.copy()
                                C = A @ A
                                found = True
                                break
                if found:
                    break
            if not found:
                break
        
        # Final check
        C = A @ A
        for u in range(n):
            for v in range(u + 1, n):
                if C[u, v] > 2:
                    # Should not happen if repair worked, but just in case
                    # Force remove
                    pass
        return A

    # Helper to greedily add edges
    def improve(A):
        A = A.copy()
        C = A @ A
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        # Check if adding (u, v) is valid
                        # New C[u, v] will be current C[u, v] (unchanged by adding edge between them? No.)
                        # Adding (u, v) increases C[x, y] for x in N(u), y in N(v).
                        # Specifically, C[x, v] increases by 1 for x in N(u).
                        # C[u, y] increases by 1 for y in N(v).
                        # We need to ensure no C[x, y] becomes > 2.
                        
                        # Check x in N(u):
                        valid = True
                        for x in range(n):
                            if A[u, x] == 1 and x != v:
                                if C[x, v] >= 2:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        # Check y in N(v):
                        for y in range(n):
                            if A[v, y] == 1 and y != u:
                                if C[u, y] >= 2:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        # Add edge
                        A[u, v] = 1
                        A[v, u] = 1
                        # Update C
                        for x in range(n):
                            if A[u, x] == 1 and x != v:
                                C[x, v] += 1
                                C[v, x] += 1
                        for y in range(n):
                            if A[v, y] == 1 and y != u:
                                C[u, y] += 1
                                C[y, u] += 1
                        changed = True
        return A

    # Main Loop
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        
        # 1. Generate Random Initial Graph
        # Start with p=0.5
        A = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    A[u, v] = 1
                    A[v, u] = 1
        
        # 2. Repair
        A = repair(A)
        
        # 3. Improve
        A = improve(A)
        
        # 4. Update Best
        edge_count = int(np.sum(A)) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_A = A.copy()
            
        # 5. Perturb (if we have a best solution)
        if best_A is not None and iterations > 10:
            # Create a perturbed version from best
            A_pert = best_A.copy()
            num_remove = random.randint(2, 5)
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if A_pert[u, v] == 1]
            if len(edges_list) >= num_remove:
                random.shuffle(edges_list)
                for u, v in edges_list[:num_remove]:
                    A_pert[u, v] = 0
                    A_pert[v, u] = 0
            
            # Repair and Improve the perturbed version
            A_pert = repair(A_pert)
            A_pert = improve(A_pert)
            
            edge_count_pert = int(np.sum(A_pert)) // 2
            if edge_count_pert > best_edges:
                best_edges = edge_count_pert
                best_A = A_pert.copy()
        
        # Safety break if no progress for a long time? 
        # The random restarts in step 1 help escape local optima.

    if best_A is None:
        # Fallback: return empty or a known valid small graph
        return []
        
    return get_edges(best_A)
