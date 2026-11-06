import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    def initialize():
        # Start with 5 disjoint K4s (30 edges)
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        return A

    def calc_K(A):
        return A @ A

    def can_add(A, K, u, v):
        if A[u, v]:
            return False
        # Check constraints for all w
        # For pair (u, w) where w is not v and u-w is non-edge:
        # New K[u, w] = K[u, w] + A[v, w]
        # Must be <= 2
        # For pair (v, w) where w is not u and v-w is non-edge:
        # New K[v, w] = K[v, w] + A[u, w]
        # Must be <= 2
        
        # Optimization: iterate w
        for w in range(n):
            if w == u or w == v:
                continue
            if A[u, w] == 0:
                # u and w are non-adjacent
                if K[u, w] + A[v, w] > 2:
                    return False
            if A[v, w] == 0:
                # v and w are non-adjacent
                if K[v, w] + A[u, w] > 2:
                    return False
        return True

    def do_add(A, K, u, v):
        A[u, v] = 1
        A[v, u] = 1
        # Update K
        for w in range(n):
            if w == u or w == v:
                continue
            if A[v, w]:
                K[u, w] += 1
                K[w, u] += 1
            if A[u, w]:
                K[v, w] += 1
                K[w, v] += 1
        # K[u, v] is not constrained for adjacent, but let's keep it consistent
        # K[u, v] = sum A[u, k] A[k, v]. Since A[u, v]=1, this doesn't add to common neighbors directly in the loop above?
        # The loop above updates K[x, y] for x,y != u,v.
        # What about K[u, v]?
        # K[u, v] = count of common neighbors.
        # Adding edge (u, v) doesn't change the set of common neighbors of u and v?
        # Wait. Common neighbors of u and v are nodes w such that A[u, w]=1 and A[v, w]=1.
        # This set is unchanged by adding (u, v) itself.
        # So K[u, v] remains the same value.
        # However, our update logic for K[x, y] usually assumes x,y are the endpoints of the changed edge?
        # No, K is a matrix of common neighbors.
        # When we add (u, v):
        # 1. u and v become neighbors.
        # 2. For any w, if v is neighbor of w, then u gains w as a common neighbor with... wait.
        # Let's re-verify K update.
        # K[x, y] = sum_k A[x, k] A[k, y].
        # Change A[u, v] from 0 to 1.
        # Terms in K[x, y] that involve A[u, v] or A[v, u]:
        # If x=u, y=w: K[u, w] = sum_k A[u, k] A[k, w].
        #   k=v term: A[u, v] A[v, w]. Before: 0. After: A[v, w]. So K[u, w] += A[v, w].
        # If x=v, y=w: K[v, w] = sum_k A[v, k] A[k, w].
        #   k=u term: A[v, u] A[u, w]. Before: 0. After: A[u, w]. So K[v, w] += A[u, w].
        # If x=w, y=u: K[w, u] = sum_k A[w, k] A[k, u].
        #   k=v term: A[w, v] A[v, u]. Before: 0. After: A[w, v]. So K[w, u] += A[w, v].
        # If x=w, y=v: K[w, v] = sum_k A[w, k] A[k, v].
        #   k=u term: A[w, u] A[u, v]. Before: 0. After: A[w, u]. So K[w, v] += A[w, u].
        # If x=u, y=v: K[u, v] = sum_k A[u, k] A[k, v].
        #   No term involves A[u, v] directly (k is index of sum, A[u, k] A[k, v]).
        #   So K[u, v] is unchanged.
        #
        # So the update logic in do_add is correct for x,y != u,v.
        # We don't need to update K[u, v] or K[v, u] because they don't change.
        # But wait, in can_add, we check K[u, w] for non-adjacent u, w.
        # If u, w become adjacent? No, we are adding u, v.
        # The pairs that change status are (u, v).
        # The pairs that change K values are (u, w) and (v, w).
        
    def do_remove(A, K, u, v):
        A[u, v] = 0
        A[v, u] = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if A[v, w]:
                K[u, w] -= 1
                K[w, u] -= 1
            if A[u, w]:
                K[v, w] -= 1
                K[w, v] -= 1

    def greedy_fill(A, K):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0 and can_add(A, K, u, v):
                        do_add(A, K, u, v)
                        changed = True

    def local_search(A, K, budget):
        best_edges = int(np.sum(A)) // 2
        best_A = A.copy()
        
        while time.time() < budget:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if not non_edges:
                break
            
            u, v = random.choice(non_edges)
            if can_add(A, K, u, v):
                do_add(A, K, u, v)
                if int(np.sum(A)) // 2 > best_edges:
                    best_edges = int(np.sum(A)) // 2
                    best_A = A.copy()
            else:
                # Try a swap: remove a random edge, then try to add (u, v) and another
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edges:
                    break
                ru, rv = random.choice(edges)
                
                # Backup state
                A_backup = A.copy()
                K_backup = K.copy()
                
                do_remove(A, K, ru, rv)
                
                # Try to add (u, v)
                success_1 = False
                if can_add(A, K, u, v):
                    do_add(A, K, u, v)
                    success_1 = True
                
                # Try to add another random non-edge to gain +1 net
                # If success_1, we are at same edge count. We need +1 more.
                # If not success_1, we are at -1 edge count. We need +2 more? Or just accept -1 if it leads to better?
                # Usually we want to maximize edges. So we should only accept swaps if we can add back >= 1 edge, preferably 2.
                
                # Let's try to add one more edge
                non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
                success_2 = False
                if non_edges_now:
                    # Try a few random ones
                    for _ in range(5):
                        if not non_edges_now: break
                        x, y = random.choice(non_edges_now)
                        if can_add(A, K, x, y):
                            do_add(A, K, x, y)
                            success_2 = True
                            # Remove from list to avoid re-trying? Or just break
                            break
                
                current_edges = int(np.sum(A)) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                elif current_edges < best_edges:
                    # Revert if we lost edges and didn't improve
                    A[:] = A_backup
                    K[:] = K_backup
                # If current_edges == best_edges, keep it (diversity)
        
        return best_A, best_edges

    best_overall_A = None
    best_overall_edges = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        A = initialize()
        # Randomize slightly? Or just use seed.
        # Let's add some random edges to seed
        for _ in range(10):
            u, v = random.sample(range(n), 2)
            if A[u, v] == 0:
                if can_add(A, calc_K(A), u, v):
                    do_add(A, calc_K(A), u, v)
        
        K = calc_K(A)
        greedy_fill(A, K)
        
        # Local search with time slice
        slice_end = min(time.time() + 5, deadline)
        A, edges = local_search(A, K, slice_end)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_A = A
    
    # Convert best_A to list of edges
    edges_list = []
    if best_overall_A is not None:
        for u in range(n):
            for v in range(u + 1, n):
                if best_overall_A[u, v]:
                    edges_list.append([u, v])
    else:
        # Fallback to seed
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    edges_list.append([4 * k + a, 4 * k + b])
                    
    return edges_list

# EVOLVE-BLOCK-END
