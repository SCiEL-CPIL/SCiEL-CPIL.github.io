import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 20
    best_edges = []
    best_count = 0
    
    def get_edge_list(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j]:
                    edges.append([i, j])
        return edges

    def is_valid(A):
        C = A @ A
        # Check off-diagonal elements
        # np.fill_diagonal(C, 0)
        # return np.max(C) <= 2
        # More efficient:
        return np.all(C[~np.eye(n, dtype=bool)] <= 2)

    def repair(A):
        # Remove edges until valid
        while True:
            C = A @ A
            # Find violating pairs
            viol = np.argwhere((C > 2) & (~np.eye(n, dtype=bool)))
            if len(viol) == 0:
                break
            # Pick random violation
            u, v = viol[random.randint(0, len(viol)-1)]
            # Pick a neighbor of u to remove
            neighbors = np.where(A[u] == 1)[0]
            if len(neighbors) == 0:
                break # Should not happen if u,v have common neighbors
            k = neighbors[random.randint(0, len(neighbors)-1)]
            A[u, k] = 0
            A[k, u] = 0

    def greedy_add(A):
        # Add edges greedily until no more can be added
        changed = True
        while changed:
            changed = False
            C = A @ A
            # Find non-edges
            non_edges = np.argwhere((A == 0) & (~np.eye(n, dtype=bool)))
            if len(non_edges) == 0:
                break
            # Shuffle to avoid bias
            random.shuffle(non_edges) # This is slow, better to pick random index
            # Actually, iterating all is fine for N=20 (190 edges)
            for i, j in non_edges:
                if A[i, j]: continue # Might have been added
                # Check if adding i-j is valid
                # New common neighbors for (i, j) is C[i,j] (which is 0 if not edge? No, C[i,j] is common neighbors. If not edge, C[i,j] is just count of common neighbors. Wait.
                # If i,j are not connected, adding them doesn't change common neighbors of (i,j) directly?
                # No. Common neighbors of (i,j) is count of k such that A[i,k]=1 and A[j,k]=1.
                # This value is C[i,j].
                # If we add edge (i,j), does C[i,j] change? No.
                # But adding (i,j) changes common neighbors of pairs (i, k) and (j, k).
                # For pair (i, k), new common neighbor is j (if A[j,k]=1).
                # So we need to check if adding (i,j) causes any pair to exceed 2.
                
                # Efficient check:
                # Pairs affected: (i, x) for all x != i, and (j, x) for all x != j.
                # For (i, x): new common count = C[i,x] + A[j,x] (since j is now neighbor of i).
                # For (j, x): new common count = C[j,x] + A[i,x] (since i is now neighbor of j).
                # Also pair (i, j) itself: common count remains C[i,j].
                
                valid = True
                # Check (i, x)
                # C[i, :] + A[j, :] must be <= 2 for all x != i, j
                # Note: C[i,i] is degree of i. We don't care about diagonal.
                # A[j, i] is 0 (since i,j non-edge).
                
                # Vectorized check:
                # new_common_i = C[i, :] + A[j, :]
                # new_common_j = C[j, :] + A[i, :]
                
                # We need new_common_i[x] <= 2 for all x != i, j
                # and new_common_j[x] <= 2 for all x != i, j
                
                # Exclude i and j from check?
                # For x=i: new_common_i[i] = C[i,i] + A[j,i] = deg(i) + 0. This is degree, not common neighbors of pair (i,i). We don't check diagonal.
                # For x=j: new_common_i[j] = C[i,j] + A[j,j] = C[i,j] + 0 = C[i,j]. This is the common neighbors of (i,j). Must be <= 2.
                
                # So we check all x.
                # But we must ensure we don't exceed 2.
                
                # Let's just do the full matrix multiplication check for simplicity first, N=20 is tiny.
                A_temp = A.copy()
                A_temp[i, j] = 1
                A_temp[j, i] = 1
                if is_valid(A_temp):
                    A = A_temp
                    changed = True
                    break # Restart loop to recalculate C and non_edges? Or just continue?
                    # Breaking restarts the inner loop, which is safe.
        
        return A

    for _ in range(100000): # Large number of restarts, controlled by time
        if time.time() > deadline:
            break
            
        # Random init
        # Use p=0.5 or 0.6
        A = (np.random.rand(n, n) < 0.6).astype(int)
        A = np.triu(A, 1)
        A = A + A.T
        np.fill_diagonal(A, 0)
        
        repair(A)
        A = greedy_add(A)
        
        # Perturbation phase
        # Try to improve by removing and re-adding
        for _ in range(10): # Small number of perturbation steps
            if time.time() > deadline:
                break
            # Remove 1-3 random edges
            edges = np.argwhere(A == 1)
            edges = [tuple(e) for e in edges if e[0] < e[1]]
            if not edges: break
            k = random.randint(1, 3)
            for _ in range(k):
                if not edges: break
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                A[u, v] = 0
                A[v, u] = 0
                edges.pop(idx)
            
            A = greedy_add(A)
            
        count = int(np.sum(A) / 2)
        if count > best_count:
            best_count = count
            best_edges = get_edge_list(A)
            
    return best_edges
