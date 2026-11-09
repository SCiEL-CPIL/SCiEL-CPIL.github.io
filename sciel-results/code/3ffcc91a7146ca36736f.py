import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Seed graph: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17

    # Try to find larger graphs
    # Start from a reasonable lower bound and go up
    # If we fail at n, we might still find it at n+1 with different random seed?
    # Usually harder, but possible.
    # We will try a few restarts for each n.
    
    for n in range(20, 36):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple restarts
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            # Density 0.5 is a good start
            A = np.random.rand(n, n) < 0.5
            A = A | A.T
            np.fill_diagonal(A, False)
            
            # Local search
            if local_search(A, n, deadline):
                # Convert to edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j]:
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                found = True
                break  # Found a valid graph for this n, try next n
        
        # If we found a graph for n, we continue to n+1 to see if we can do better
        # If we didn't, we continue anyway, but maybe reduce restarts for higher n?
        # For now, keep constant restarts.

    return {'n': best_n, 'edges': best_edges}

def local_search(A, n, deadline):
    # A is modified in place
    # Returns True if valid, False if timeout
    while time.time() < deadline:
        # 1. Check for K4
        k4 = find_k4(A, n)
        if k4 is not None:
            # Fix K4 by removing a random edge from it
            u, v, w, x = k4
            # Edges in K4: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x)
            # Note: find_k4 returns i, j, v1, v2 where (i,j) is an edge, and v1,v2 are common neighbors connected.
            # So the K4 is {i, j, v1, v2}.
            # The edges are all pairs from this set.
            edges_in_k4 = [
                (u, v), (u, w), (u, x),
                (v, w), (v, x),
                (w, x)
            ]
            # Randomly pick one to remove
            e = random.choice(edges_in_k4)
            A[e[0], e[1]] = False
            A[e[1], e[0]] = False
            continue
        
        # 2. Check for Independent Set of size 6
        ind6 = find_independent_6(A, n)
        if ind6 is not None:
            # Fix by adding a random edge between two vertices in the independent set
            # They are non-adjacent, so adding an edge is safe (doesn't create self-loop)
            v1, v2 = random.sample(ind6, 2)
            A[v1, v2] = True
            A[v2, v1] = True
            continue
        
        # 3. If no violations, we are done
        return True
    
    return False

def find_k4(A, n):
    # Find one K4. Returns (u, v, w, x) or None.
    # Iterate over edges
    # Optimization: If we just removed an edge, we might not need to check all edges.
    # But simple full scan is robust.
    
    # To speed up, we can check a random subset of edges first?
    # No, let's just scan.
    
    # Get all edges
    # Using np.argwhere is O(n^2) to create list.
    # Iterating i, j is also O(n^2).
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                # Common neighbors of i and j
                # A[i] and A[j] are boolean arrays of length n
                common = A[i] & A[j]
                # Get indices of common neighbors
                C = np.nonzero(common)[0]
                if len(C) < 2:
                    continue
                
                # Check if the subgraph induced by C has any edge
                # If yes, we have a K4 consisting of i, j, and two vertices from C
                # We need to find two vertices in C that are connected.
                
                # Efficient check:
                # If (A[np.ix_(C, C)]).any():
                #    # Find the specific edge
                #    sub = A[np.ix_(C, C)]
                #    idx = np.argwhere(sub)
                #    # Filter out diagonal (though A has no self loops, so sub diagonal is False)
                #    # Actually, if sub has any True, it's an edge between two distinct vertices in C
                #    c1_idx, c2_idx = idx[0]
                #    v1 = C[c1_idx]
                #    v2 = C[c2_idx]
                #    return (i, j, v1, v2)
                
                # To avoid creating the submatrix if possible?
                # We can iterate over C and check A[C[k], C]
                # But C is small.
                
                if (A[np.ix_(C, C)]).any():
                    sub = A[np.ix_(C, C)]
                    idx = np.argwhere(sub)
                    # Ensure we pick a valid edge (i != j in the submatrix)
                    # Since A is simple, diagonal is 0.
                    # So any True is a valid edge.
                    c1_idx, c2_idx = idx[0]
                    v1 = C[c1_idx]
                    v2 = C[c2_idx]
                    return (i, j, v1, v2)
    
    return None

def find_independent_6(A, n):
    # Greedy algorithm to find an independent set of size 6
    # Returns the set if found, else None
    
    candidates = list(range(n))
    is_set = []
    
    while candidates:
        # Pick vertex with minimum degree in the induced subgraph
        min_deg = float('inf')
        v = -1
        
        # To speed up, we can use numpy
        # degs = A[:, candidates].sum(axis=1)
        # v = candidates[np.argmin(degs[candidates])]
        
        # But Python loop is fine for small n
        for u in candidates:
            deg = A[u, candidates].sum()
            if deg < min_deg:
                min_deg = deg
                v = u
        
        if v == -1:
            break
            
        is_set.append(v)
        
        # Remove v and its neighbors from candidates
        # Neighbors of v in candidates
        # A[v, candidates] is a boolean array
        # We keep u if not A[v, u]
        
        # Vectorized removal
        # mask = ~A[v, candidates]
        # candidates = [candidates[i] for i in range(len(candidates)) if mask[i]]
        
        # Or:
        new_candidates = []
        for u in candidates:
            if u != v and not A[v, u]:
                new_candidates.append(u)
        candidates = new_candidates
        
        if len(is_set) >= 6:
            return is_set
            
    return None
