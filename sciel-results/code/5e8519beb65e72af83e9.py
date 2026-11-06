import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    def check_time():
        return time.time() < deadline

    def get_cost(A):
        """
        Calculate the number of K5s and independent 5-sets.
        A is a numpy array of shape (N, N) with 0/1 entries.
        """
        N = A.shape[0]
        # A^2 gives common neighbors
        A2 = A @ A
        cost = 0
        
        # Count K5s
        # Sum over all edges (u,v) of C(common_neighbors(u,v) - 2, 3)
        # Note: (A2)_{uv} includes u and v if there is a self-loop? No, A has 0 diagonal.
        # (A2)_{uv} = sum_k A_{uk} A_{kv}. This is the number of common neighbors.
        # For a K5, u and v are connected. Common neighbors are the other 3.
        # So we need C((A2)_{uv}, 3) if A_{uv}=1?
        # Wait. If u,v are connected, do they count as common neighbors?
        # No, common neighbors are w such that (u,w) and (v,w) are edges.
        # u is not a neighbor of u. v is a neighbor of u.
        # So (A2)_{uv} counts w != u, v.
        # So if (u,v) is an edge, and there are 3 common neighbors, we have a K5 containing u,v and those 3.
        # Each K5 is counted 10 times (once for each edge).
        
        # Iterate over edges
        edges = np.where(A > 0)
        for u, v in zip(edges[0], edges[1]):
            if u < v: # Only count each edge once
                common = A2[u, v]
                if common >= 3:
                    # C(common, 3)
                    cost += common * (common - 1) * (common - 2) // 6
        
        # Count Independent 5-sets
        # Independent 5-set in G is K5 in G_bar.
        # G_bar has edges where A has 0.
        # Common non-neighbors in G = Common neighbors in G_bar.
        # D_comp(u, v) = N - 2 - (deg(u) + deg(v) - 2*A[u,v] - D(u,v)) ?
        # Let's use the formula derived: D_comp(u,v) = N - d(u) - d(v) + D(u,v)
        # Where D(u,v) = (A2)_{uv}.
        
        deg = A.sum(axis=1)
        
        # Iterate over non-edges
        # To avoid O(N^2) python loop, we can vectorize?
        # But N is small (43). O(N^2) is ~1800 iterations. Fast enough.
        
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v] == 0:
                    d_comp = N - deg[u] - deg[v] + A2[u, v]
                    if d_comp >= 3:
                        cost += d_comp * (d_comp - 1) * (d_comp - 2) // 6
        
        return cost // 10 # Divide by 10 because each K5/Ind5 is counted 10 times

    def solve_N(N, time_limit):
        if not check_time():
            return None
        
        # Initialize random graph
        A = np.random.randint(0, 2, (N, N))
        np.fill_diagonal(A, 0)
        A = np.triu(A, 1)
        A = A + A.T
        
        current_cost = get_cost(A)
        if current_cost == 0:
            return A
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while check_time() and time.time() < start_time + time_limit:
            # Pick random edge
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            
            # Flip
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]
            
            new_cost = get_cost(A)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost == 0:
                    return A
            else:
                # Revert
                A[u, v] = 1 - A[u, v]
                A[v, u] = A[u, v]
            
            T *= cooling
            if T < min_T:
                T = min_T
                # Restart with higher T if stuck?
                # Or just continue
                # Let's reset T occasionally
                if random.random() < 0.1:
                    T = 1.0
                    
        return None

    # Try N from 43 down to 38
    # Allocate time
    # 43: 100s
    # 42: 80s
    # 41: 50s
    # 40: 30s
    # 39: 15s
    # 38: 5s
    
    time_alloc = {
        43: 100,
        42: 80,
        41: 50,
        40: 30,
        39: 15,
        38: 5
    }
    
    for N in [43, 42, 41, 40, 39, 38]:
        if not check_time():
            break
        
        limit = time_alloc.get(N, 5)
        # Ensure we don't exceed global deadline
        remaining = deadline - time.time()
        if limit > remaining:
            limit = remaining
            
        A = solve_N(N, limit)
        if A is not None:
            # Convert to edges
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if A[i, j] == 1:
                        edges.append((i, j))
            return N, edges

    # Fallback to Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
