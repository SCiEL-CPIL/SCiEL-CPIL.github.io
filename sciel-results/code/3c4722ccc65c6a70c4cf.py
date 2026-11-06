import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin
    
    def now():
        return time.time()
    
    def time_left():
        return deadline - now()
    
    def has_k4(adj, n):
        """Check if adj (list of bitmasks) has a K4. Return True if found."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                if not (ai >> j) & 1:
                    continue
                # Common neighbors of i and j (excluding i, j)
                common = adj[i] & adj[j]
                common &= ~((1 << (i+1)) - 1)  # Remove bits 0..i
                common &= ~(1 << j)
                # Need two vertices k, l in common that are connected
                temp = common
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check if k has a neighbor in common
                    if adj[k] & common:
                        return True
        return False
    
    def has_independent_6(adj, n):
        """Check if adj has an independent set of size 6 (equivalent to K6 in complement)."""
        # Build complement
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
        
        # Find K6 in comp
        # Iterate over all 6-subsets is too slow. Use a smarter approach.
        # Check if there exists a 6-clique in comp.
        # We can use a recursive search with pruning, but let's try a bitset approach.
        
        # For each vertex, compute its neighbors in comp.
        # A K6 exists if there's a set of 6 mutually connected vertices.
        
        # Try all pairs (i, j) in comp, then find K4 in their common neighbors.
        for i in range(n):
            ci = comp[i]
            for j in range(i+1, n):
                if not (ci >> j) & 1:
                    continue
                common = comp[i] & comp[j]
                common &= ~((1 << (j+1)) - 1)
                common &= ~(1 << i)
                # Need K4 in common (within comp)
                temp = common
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    temp ^= lsb
                    common2 = comp[k] & common
                    if common2:
                        temp2 = common2
                        while temp2:
                            lsb2 = temp2 & -temp2
                            l = lsb2.bit_length() - 1
                            temp2 ^= lsb2
                            if comp[l] & common2:
                                # Found K4 in common => K6 with i, j
                                return True
        return False
    
    def is_valid(adj, n):
        return not has_k4(adj, n) and not has_independent_6(adj, n)
    
    def make_adj_from_edges(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def spectral_construction(n, iterations=2000):
        """
        Start with random ±1 matrix, iteratively flip signs to reduce spectral radius,
        then threshold to create a graph.
        """
        import numpy as np
        import scipy.linalg as la
        
        # Initialize with random ±1
        A = np.random.choice([-1, 1], size=(n, n))
        # Make symmetric
        A = (A + A.T) / 2
        # Zero diagonal
        np.fill_diagonal(A, 0)
        
        best_A = A.copy()
        best_lambda = float('inf')
        
        for it in range(iterations):
            if now() > deadline - 20:
                break
            
            # Compute eigenvalues
            try:
                eigvals = la.eigvals(A)
                max_abs = max(abs(eigvals))
            except:
                break
            
            if max_abs < best_lambda:
                best_lambda = max_abs
                best_A = A.copy()
            
            # Flip a random off-diagonal entry
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            A[i, j] *= -1
            A[j, i] *= -1
        
        # Threshold: edge if A[i,j] > 0
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if best_A[i, j] > 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj
    
    def local_search(adj, n, max_steps=5000):
        """
        Local search: flip edges to eliminate K4s and independent 6-sets.
        """
        adj = adj.copy()
        steps = 0
        while steps < max_steps:
            if now() > deadline - 15:
                break
            steps += 1
            
            # Check for K4
            if has_k4(adj, n):
                # Find a K4 and remove a random edge from it
                # Simple approach: try removing random edges until K4 goes away
                for _ in range(20):
                    i = random.randint(0, n-1)
                    j = random.randint(i+1, n-1)
                    if (adj[i] >> j) & 1:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        if not has_k4(adj, n):
                            break
            elif has_independent_6(adj, n):
                # Add a random non-edge
                for _ in range(20):
                    i = random.randint(0, n-1)
                    j = random.randint(i+1, n-1)
                    if not ((adj[i] >> j) & 1):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        if not has_independent_6(adj, n):
                            break
            else:
                # Valid!
                break
        
        return adj
    
    def greedy_construct(n):
        """
        Greedy: build vertex by vertex, connecting to existing vertices
        in a way that avoids creating K4s or independent 6-sets.
        """
        adj = [0] * n
        for v in range(1, n):
            if now() > deadline - 20:
                break
            # Try random subsets of connections to existing vertices
            best_conn = None
            for attempt in range(100):
                # Random subset of {0, ..., v-1}
                conn = 0
                for u in range(v):
                    if random.random() < 0.5:
                        conn |= (1 << u)
                # Check if adding v with these connections creates K4 or indep 6
                valid = True
                # Check K4 involving v
                # v plus 3 mutual neighbors
                # Check if any 3 neighbors of v form a triangle
                neighbors = []
                for u in range(v):
                    if (conn >> u) & 1:
                        neighbors.append(u)
                if len(neighbors) >= 3:
                    for a, b, c in combinations(neighbors, 3):
                        if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                            valid = False
                            break
                if not valid:
                    continue
                # Check independent 6 involving v
                # v plus 5 mutual non-neighbors
                non_neighbors = [u for u in range(v) if not ((conn >> u) & 1)]
                if len(non_neighbors) >= 5:
                    # Check if any 5 non-neighbors form an independent set
                    # This is expensive, so skip for now or do a quick check
                    # For simplicity, just check if they are pairwise non-adjacent
                    found_indep = False
                    for subset in combinations(non_neighbors, 5):
                        is_indep = True
                        for x, y in combinations(subset, 2):
                            if (adj[x] >> y) & 1:
                                is_indep = False
                                break
                        if is_indep:
                            found_indep = True
                            break
                    if found_indep:
                        valid = False
                        continue
                if valid:
                    best_conn = conn
                    break
            
            if best_conn is not None:
                adj[v] = best_conn
                for u in range(v):
                    if (best_conn >> u) & 1:
                        adj[u] |= (1 << v)
        
        return adj
    
    # Try different strategies for increasing n
    best_result = None
    best_n = 0
    
    for n in range(35, 15, -1):
        if now() > deadline - 30:
            break
        
        found = False
        strategies = [
            ('spectral', lambda: spectral_construction(n, iterations=1000)),
            ('greedy', lambda: greedy_construct(n)),
        ]
        
        for strategy_name, strategy_fn in strategies:
            if now() > deadline - 30:
                break
            for attempt in range(3):
                if now() > deadline - 30:
                    break
                adj = strategy_fn()
                adj = local_search(adj, n, max_steps=3000)
                if is_valid(adj, n):
                    best_n = n
                    best_result = edges_from_adj(adj, n)
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Try to find a smaller valid graph
            for n2 in range(n-1, 15, -1):
                if now() > deadline - 30:
                    break
                adj = spectral_construction(n2, iterations=500)
                adj = local_search(adj, n2, max_steps=2000)
                if is_valid(adj, n2):
                    best_n = n2
                    best_result = edges_from_adj(adj, n2)
                    break
    
    if best_result is None:
        # Fallback: return a small valid graph (Paley graph of order 17)
        squares = {x * x % 17 for x in range(1, 17)}
        n = 17
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_result}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}")
