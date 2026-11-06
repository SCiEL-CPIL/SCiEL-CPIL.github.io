import itertools
import random
import numpy as np
import time

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Maximizes n.
    """
    # Target n values to try, in descending order
    # 21 is the target for improvement, 20 is known to exist
    target_ns = [21, 20]
    
    # Total time budget for the function
    total_time_limit = 115  # seconds
    
    start_time = time.time()
    
    def get_edges_from_matrix(A):
        n = A.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i][j]:
                    edges.append([i, j])
        return edges

    def is_regular_induced(A, S):
        """
        Checks if the induced subgraph on vertices S is regular.
        S is a tuple/list of vertex indices.
        Optimized for small k.
        """
        k = len(S)
        if k < 2:
            return True
        
        # Manual extraction for speed if k is small
        # Build sub-adjacency list
        # Map original indices to 0..k-1
        # Actually, just summing edges between pairs in S is enough.
        # Degree of u in G[S] is count of v in S, v!=u, such that (u,v) in E.
        
        # For small k, double loop is fine
        degrees = [0] * k
        for i in range(k):
            u = S[i]
            deg = 0
            for j in range(k):
                if i == j: continue
                v = S[j]
                if A[u][v]:
                    deg += 1
            degrees[i] = deg
            
        # Check if all degrees are equal
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True

    def find_bad_subset(A, time_limit):
        """
        Finds a subset S of size >= 6 that induces a regular subgraph.
        Returns the subset S or None if no such subset exists.
        Returns 'TIMEOUT' if time limit exceeded.
        """
        n = A.shape[0]
        start_t = time.time()
        
        # Check sizes from 6 to n
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                if time.time() - start_t > time_limit:
                    return 'TIMEOUT'
                if is_regular_induced(A, S):
                    return S
        return None

    def local_search(n, time_limit):
        """
        Performs local search to find a valid graph on n vertices.
        """
        # Initialize random graph
        A = (np.random.rand(n, n) < 0.5).astype(int)
        np.fill_diagonal(A, 0)
        A = (A + A.T) > 0
        A = A.astype(int)
        
        # Check initial
        bad_S = find_bad_subset(A, time_limit / 10)
        if bad_S == 'TIMEOUT':
            return None
        if bad_S is None:
            return A
        
        max_steps = 50
        step_time_limit = time_limit / max_steps
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            bad_S = find_bad_subset(A, step_time_limit)
            if bad_S == 'TIMEOUT':
                break
            if bad_S is None:
                return A
                
            S_list = list(bad_S)
            
            if len(S_list) < 2:
                continue
                
            # Pick random edge in S to flip
            u, v = random.sample(S_list, 2)
            
            # Flip edge
            A[u][v] = 1 - A[u][v]
            A[v][u] = A[u][v]
            
        # Return None to indicate failure
        return None

    best_A = None
    best_n = 0
    
    # Try n=21 first with a small time budget
    # Then n=20 with the rest
    time_alloc_21 = 10.0
    time_alloc_20 = 100.0
    
    # Attempt n=21
    if time.time() - start_time < total_time_limit - time_alloc_20:
        A = local_search(21, time_alloc_21)
        if A is not None:
            # Verify
            if find_bad_subset(A, 1.0) is None:
                best_A = A
                best_n = 21
    
    # Attempt n=20 if 21 failed or to ensure we have a fallback
    if best_A is None:
        remaining = total_time_limit - (time.time() - start_time)
        if remaining > 5:
            A = local_search(20, remaining)
            if A is not None:
                # Verify
                if find_bad_subset(A, 1.0) is None:
                    best_A = A
                    best_n = 20

    if best_A is not None:
        return {'n': best_n, 'edges': get_edges_from_matrix(best_A)}
    
    # Fallback: n=6
    # Any non-regular graph on 6 vertices is valid (since only subset of size >=6 is the whole graph)
    while time.time() - start_time < total_time_limit:
        n = 6
        A = (np.random.rand(n, n) < 0.5).astype(int)
        np.fill_diagonal(A, 0)
        A = (A + A.T) > 0
        A = A.astype(int)
        degrees = A.sum(axis=1)
        if not all(d == degrees[0] for d in degrees):
            return {'n': n, 'edges': get_edges_from_matrix(A)}
            
    # Absolute fallback
    return {'n': 5, 'edges': []}
