"""Deterministic algebraic construction + rapid vectorized verification for
graphs with no induced regular subgraph on >= 6 vertices.

Strategy:
  1. Generate candidate graphs via:
     - Paley-type graphs over small finite fields
     - Quadratic-form graphs over GF(q)
     - Random graphs with controlled density
     - Greedy constructive search with fast pruning
  2. For each candidate, verify NO subset of size >= 6 induces a regular graph.
     Verification uses NumPy bitmask operations for speed.
  3. Return the largest valid graph found within the time budget.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # --- Verification: check if a graph has any induced regular subgraph of size >= 6 ---
    def verify_no_regular_subgraph(adj, n):
        """
        adj: n x n boolean numpy array (symmetric, zero diagonal)
        n: number of vertices
        Returns True if no subset of size >= 6 induces a regular graph.
        """
        # Precompute degrees within any subset using bitmasks
        # For each subset of size k >= 6, check if all induced degrees are equal
        
        # Efficient approach: iterate over all subsets of size 6 to n
        # For n up to ~20, C(20,6)=38760, C(20,7)=77520, ..., C(20,20)=1
        # Total subsets of size >= 6: sum_{k=6}^{20} C(20,k) = 2^20 - sum_{k=0}^{5} C(20,k)
        # = 1048576 - (1+20+190+1140+4845+15504) = 1048576 - 21600 = 1026976
        # That's about 1M subsets, each requiring O(k) work. Might be slow in Python.
        
        # Better: use the fact that for a subset S of size k, it's regular iff
        # all vertices in S have the same degree within S.
        # We can precompute for each vertex v and each subset S containing v,
        # the degree of v in S. But that's still expensive.
        
        # Alternative: for each subset of size exactly 6, check if regular.
        # If a subset of size > 6 is regular, then every 6-subset of it is also regular?
        # No! A 7-vertex regular graph (e.g., 3-regular on 7 vertices... wait, 7*3/2 is not integer)
        # Actually, a k-regular graph on m vertices requires m*k to be even.
        # Key insight: if S of size k >= 6 is regular, then |S| * d is even where d is the degree.
        # But more importantly, we only need to check subsets of size 6!
        # Why? Because if there's a regular induced subgraph on k >= 6 vertices,
        # then... actually no, a 7-vertex regular graph doesn't necessarily contain a 6-vertex regular induced subgraph.
        
        # Let me reconsider. We need to check ALL subsets of size >= 6.
        # For n <= 22, the total number of subsets of size >= 6 is:
        # sum_{k=6}^{22} C(22,k) = 2^22 - sum_{k=0}^{5} C(22,k)
        # C(22,0)=1, C(22,1)=22, C(22,2)=231, C(22,3)=1540, C(22,4)=7315, C(22,5)=26334
        # sum = 1+22+231+1540+7315+26334 = 35443
        # 2^22 = 4194304
        # So about 4.16M subsets. Too many to check individually in Python.
        
        # Optimized approach: for each subset, compute degrees vectorized.
        # Actually, let's use a different strategy:
        # For each subset S, the induced subgraph is regular iff the degree sequence is constant.
        # We can compute degrees for all vertices in S simultaneously using matrix operations.
        
        # But even better: use the following observation.
        # A graph is regular iff the adjacency matrix restricted to S has all row sums equal.
        
        # For efficiency, let's check subsets of size 6 first (most common violation),
        # then sizes 7, 8, etc., with early termination.
        
        # Precompute adjacency as integer bitmask for each vertex
        adj_masks = np.zeros(n, dtype=np.int64)
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        # Check all subsets of size 6
        # For a subset S = {v0, v1, ..., v5}, compute degree of each vi in S
        # deg(vi in S) = popcount(adj_masks[vi] & mask_S)
        
        # Vectorized check for all 6-subsets:
        # This is still C(n,6) subsets. For n=22, C(22,6) = 74613.
        # For each, we do 6 popcount operations. That's ~447K operations, feasible.
        
        for size in range(6, n + 1):
            if time_left() < 5.0:
                return False  # can't finish, assume invalid to be safe
            
            for subset in combinations(range(n), size):
                # Compute mask for this subset
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Compute degree of each vertex in subset
                degrees = []
                for v in subset:
                    # Count bits in adj_masks[v] & mask
                    m = adj_masks[v] & mask
                    # Popcount
                    deg = bin(m).count('1')
                    degrees.append(deg)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False  # Found a regular induced subgraph
        
        return True  # No regular induced subgraph of size >= 6
    
    # --- Generate candidate graphs ---
    
    def make_paley_graph(q):
        """Create a Paley graph over GF(q) where q is a prime power, q ≡ 1 mod 4."""
        # For simplicity, use q = 5, 13, 17, 29, 37, 41, ...
        # q must be prime and q ≡ 1 (mod 4)
        if q < 5 or q % 4 != 1:
            return None
        
        # Check if q is prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        if not is_prime(q):
            return None
        
        # In GF(q), x is a quadratic residue iff x^((q-1)/2) == 1 (for x != 0)
        # Edge between i and j (i != j) iff (j - i) is a quadratic residue mod q
        adj = np.zeros((q, q), dtype=bool)
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff == 0:
                    continue
                # Check if diff is a quadratic residue
                if pow(diff, (q - 1) // 2, q) == 1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_quadratic_form_graph(q, n_vertices):
        """Create a graph based on a quadratic form over GF(q)."""
        # This is more complex; skip for now and use Paley
        return None
    
    def make_random_graph(n, p, seed=42):
        """Create a random graph with edge probability p."""
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    def make_complement(adj):
        """Return the complement of a graph."""
        n = adj.shape[0]
        comp = ~adj.copy()
        np.fill_diagonal(comp, False)
        return comp
    
    # --- Search strategy ---
    best_result = None
    best_n = 0
    
    # Try Paley graphs
    paley_q_values = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
    for q in paley_q_values:
        if time_left() < 10.0:
            break
        adj = make_paley_graph(q)
        if adj is not None and q <= 22:
            if verify_no_regular_subgraph(adj, q):
                if q > best_n:
                    best_n = q
                    best_result = adj
        # Also try complement
        if adj is not None and q <= 22:
            comp = make_complement(adj)
            if verify_no_regular_subgraph(comp, q):
                if q > best_n:
                    best_n = q
                    best_result = comp
    
    # Try random graphs with various densities, prioritizing larger n
    # We know n=20 is possible. Try to find n=21, then n=20.
    for n in range(21, 9, -1):
        if time_left() < 10.0:
            break
        if best_n >= n:
            continue
        for seed in range(0, 100):
            if time_left() < 5.0:
                break
            for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
                if time_left() < 2.0:
                    break
                adj = make_random_graph(n, p, seed=seed * 1000 + int(p * 10))
                if verify_no_regular_subgraph(adj, n):
                    if n > best_n:
                        best_n = n
                        best_result = adj
                    break  # Found one for this n, move to next n
        # If we found a valid graph for this n, we can stop looking for smaller n
        # unless we want to ensure we have the best possible (which is n itself)
        # Since we iterate downwards, the first valid n is the largest.
        if best_n >= n:
            break
    
    # Greedy constructive search
    def greedy_construct(n_target, seed=0):
        """Try to build a graph on n_target vertices with no regular induced subgraph of size >= 6."""
        rng = np.random.default_rng(seed)
        adj = np.zeros((n_target, n_target), dtype=bool)
        
        for v in range(n_target):
            # Try random neighbor masks
            found = False
            for attempt in range(100):
                if time_left() < 5.0:
                    return None
                # Choose neighbors from previous vertices
                neighbors = set()
                for u in range(v):
                    if rng.random() < 0.5:
                        neighbors.add(u)
                
                # Temporarily add vertex v with these neighbors
                for u in neighbors:
                    adj[v, u] = True
                    adj[u, v] = True
                
                # Check if this creates a regular induced subgraph
                if verify_no_regular_subgraph(adj, v + 1):
                    found = True
                    break
                else:
                    # Remove
                    for u in neighbors:
                        adj[v, u] = False
                        adj[u, v] = False
            
            if not found:
                return None
        
        return adj
    
    # Try greedy construction for increasing n
    for n_target in range(15, 23):
        if time_left() < 20.0:
            break
        if best_n >= n_target:
            continue
        for seed in range(0, 20):
            if time_left() < 15.0:
                break
            adj = greedy_construct(n_target, seed=seed)
            if adj is not None:
                if n_target > best_n:
                    best_n = n_target
                    best_result = adj
                break
    
    # Convert best_result to the required format
    if best_result is not None:
        n = best_result.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_result[i, j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
