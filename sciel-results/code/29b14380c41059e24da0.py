import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit

    def check_time():
        return time.time() < deadline

    def get_cost(A):
        """
        Calculate the number of K5s and independent 5-sets.
        A is a numpy array of shape (N, N) with 0/1 entries.
        Returns the total count of violations (K5 + Ind5).
        """
        N = A.shape[0]
        A2 = A @ A
        cost = 0
        
        # Count K5s
        # For each edge (u, v), if there are >= 3 common neighbors, 
        # it forms C(common, 3) K5s containing that edge.
        # Each K5 has 10 edges, so it is counted 10 times.
        edges = np.where(A > 0)
        # Only consider i < j
        u_list = edges[0]
        v_list = edges[1]
        mask = u_list < v_list
        u_list = u_list[mask]
        v_list = v_list[mask]
        
        for u, v in zip(u_list, v_list):
            common = A2[u, v]
            if common >= 3:
                # C(common, 3)
                cost += common * (common - 1) * (common - 2) // 6
        
        # Count Independent 5-sets
        # Independent 5-set in G is K5 in G_bar.
        # Common non-neighbors in G = Common neighbors in G_bar.
        # d_comp(u, v) = N - 2 - (deg(u) + deg(v) - 2*A[u,v] - common(u,v))
        # Wait, deg(u) + deg(v) = common(u,v) + (neighbors of u not v) + (neighbors of v not u) + 2*A[u,v]
        # Non-neighbors of u = N - 1 - deg(u)
        # Common non-neighbors of u and v:
        # w is not neighbor of u AND w is not neighbor of v.
        # Total vertices = N.
        # w != u, w != v.
        # deg(u) + deg(v) - common(u,v) - 2*A[u,v] is the size of union of neighbors (excluding u,v themselves? No).
        # Let's use the formula:
        # d_comp(u, v) = N - 2 - (deg(u) + deg(v) - 2*A[u,v] - A2[u,v])
        # Let's verify.
        # If u, v are non-adjacent (A[u,v]=0).
        # deg(u) + deg(v) - A2[u,v] is the number of vertices w (w!=u,v) that are neighbor of u OR neighbor of v.
        # So N - 2 - (deg(u) + deg(v) - A2[u,v]) is the number of vertices w (w!=u,v) that are NOT neighbor of u AND NOT neighbor of v.
        # This is d_comp(u, v).
        
        deg = A.sum(axis=1)
        
        # Iterate over non-edges
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v] == 0:
                    d_comp = N - 2 - (deg[u] + deg[v] - A2[u, v])
                    if d_comp >= 3:
                        cost += d_comp * (d_comp - 1) * (d_comp - 2) // 6
        
        return cost // 10

    def generate_paley_seed(N):
        """
        Generate a seed graph by extending the Paley graph on 37 vertices.
        """
        A = np.zeros((N, N), dtype=int)
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        for i in range(q):
            for j in range(q):
                if i != j and (j - i) % q in residues:
                    A[i, j] = 1
        
        # Randomly connect new vertices (37 to N-1)
        for i in range(q, N):
            # Connect to existing
            for j in range(q):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
            # Connect to other new vertices
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        return A

    def generate_random_seed(N):
        A = np.random.randint(0, 2, (N, N))
        np.fill_diagonal(A, 0)
        A = np.triu(A, 1)
        A = A + A.T
        return A

    def solve_N(N, time_limit):
        if not check_time():
            return None
        
        # Try multiple seeds
        seeds = []
        if N >= 37:
            seeds.append(generate_paley_seed(N))
        # Add random seeds
        for _ in range(4):
            seeds.append(generate_random_seed(N))
        
        time_per_seed = time_limit / len(seeds)
        
        for A in seeds:
            if not check_time() or time.time() > start_time + time_limit:
                continue
                
            current_cost = get_cost(A)
            if current_cost == 0:
                return A
            
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            seed_start = time.time()
            
            while check_time() and time.time() - seed_start < time_per_seed:
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
                    # Restart T occasionally
                    if random.random() < 0.05:
                        T = 1.0
                        
        return None

    # Try N from 42 down to 38
    time_alloc = {
        42: 200,
        41: 50,
        40: 20,
        39: 10,
        38: 5
    }
    
    for N in [42, 41, 40, 39, 38]:
        if not check_time():
            break
        
        limit = time_alloc.get(N, 5)
        remaining = deadline - time.time()
        if limit > remaining:
            limit = remaining
            
        A = solve_N(N, limit)
        if A is not None:
            # Verify
            if get_cost(A) == 0:
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
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                edges.append((i, j))
    
    return n, edges
