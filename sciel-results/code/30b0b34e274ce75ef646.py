import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Target n
    target_n = 21  # Try for 21, fallback to 20
    
    # We will try to construct a graph of size target_n
    # If we fail, we try target_n - 1, etc.
    
    def get_potential_subsets(A, k):
        """
        Returns a list of (mask, required_M_mask) for subsets S' of {0..k-1}
        such that if M matches required_M_mask on S', S' becomes regular.
        """
        potentials = []
        # Iterate subsets of size >= 5
        # Optimization: Only check subsets that are "close" to regular in A?
        # No, the condition is on D_u.
        
        # For k=20, 2^20 = 1M subsets.
        # Checking each takes ~20 ops. 20M ops.
        # This might be ~1-2 seconds in Python.
        # Acceptable if we don't do it too many times.
        
        vertices = range(k)
        # Use bitmask iteration
        for mask in range(1, 1 << k):
            # Count bits
            # Python 3.10+ has int.bit_count()
            if mask.bit_count() < 5:
                continue
            
            # Extract vertices
            verts = [i for i in vertices if (mask >> i) & 1]
            size = len(verts)
            
            # Calculate D_u for u in verts
            # D_u = sum of A[u, v] for v in verts, v != u
            # We can use A[verts, :][:, verts].sum(axis=1) - 1 (self loop)
            # But A is 0 on diagonal.
            sub_A = A[np.ix_(verts, verts)]
            D = sub_A.sum(axis=1)
            
            sum_D = D.sum()
            denom = size - 1
            
            if sum_D % denom != 0:
                continue
            
            K = sum_D // denom
            
            # Check if M_u = K - D_u is in {0, 1}
            req_M = []
            valid = True
            for u, d_u in zip(verts, D):
                m_val = K - d_u
                if m_val < 0 or m_val > 1:
                    valid = False
                    break
                req_M.append((u, m_val))
            
            if valid:
                # Create a mask for required M
                req_mask = 0
                for u, m_val in req_M:
                    if m_val == 1:
                        req_mask |= (1 << u)
                # We also need to ensure that for bits NOT in S', M can be anything.
                # The condition only restricts M on S'.
                # So a candidate M is bad if (M & mask) == req_mask.
                potentials.append((mask, req_mask))
        
        return potentials

    def try_construct(n_target):
        # Backtracking search
        # State: A (k x k), k
        # We'll use recursion or iterative deepening?
        # Simple recursion with backtracking.
        
        def backtrack(A, k):
            if time.time() > deadline - 5:
                return None
            
            if k == n_target:
                # Success
                edges = []
                for i in range(k):
                    for j in range(i+1, k):
                        if A[i, j]:
                            edges.append([i, j])
                return edges
            
            # Precompute potentials for current k
            # This is expensive. 
            # Optimization: Cache potentials? 
            # Potentials depend on A. A changes.
            
            if k == 0:
                # No previous vertices
                # M is empty. No subsets of size >= 5.
                # Any M (empty) is valid.
                new_A = np.zeros((1, 1), dtype=int)
                res = backtrack(new_A, 1)
                return res
            
            potentials = get_potential_subsets(A, k)
            
            # Generate all 2^k candidates for M
            # Shuffle them to randomize
            candidates = list(range(1 << k))
            random.shuffle(candidates)
            
            for m_val in candidates:
                # Check if m_val causes any violation
                bad = False
                for p_mask, p_req in potentials:
                    if (m_val & p_mask) == p_req:
                        bad = True
                        break
                
                if not bad:
                    # Valid extension
                    new_A = np.zeros((k+1, k+1), dtype=int)
                    new_A[:k, :k] = A
                    # Set new edges
                    for i in range(k):
                        if (m_val >> i) & 1:
                            new_A[k, i] = 1
                            new_A[i, k] = 1
                    
                    res = backtrack(new_A, k+1)
                    if res is not None:
                        return res
            
            return None
        
        # Start with empty graph
        A_init = np.zeros((0, 0), dtype=int)
        return backtrack(A_init, 0)

    # Try to construct for target_n
    # Since backtracking might be slow, we limit the depth or restarts?
    # The problem size is small (n=21).
    # But 2^20 candidates at each step is 1M.
    # If we have to check 1M candidates * 1M potentials (worst case), it's 10^12. Too slow.
    # However, potentials list is usually small.
    # And we stop at the first valid candidate.
    # If valid candidates are rare, we scan many.
    
    # Let's add a timeout inside the loop.
    
    edges = try_construct(target_n)
    
    if edges is None:
        # Try smaller n
        for n_try in range(target_n - 1, 4, -1):
            if time.time() > deadline - 10:
                break
            edges = try_construct(n_try)
            if edges is not None:
                target_n = n_try
                break
    
    if edges is None:
        # Fallback: Random graph on 5 vertices (no 6-subsets)
        return {'n': 5, 'edges': []}
        
    return {'n': target_n, 'edges': edges}
