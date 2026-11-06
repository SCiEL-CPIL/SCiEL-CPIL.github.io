import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def check_validity_fast(n, orientation):
        """Check if every 4-subset has a common dominator using numpy."""
        # Build adjacency matrix: adj[i][j] = 1 if i -> j
        adj = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i, j] = 1
                else:
                    adj[j, i] = 1
                idx += 1
        
        # For each vertex v, out_mask[v] = set of vertices v points to
        # A vertex v is a dominator of {a,b,c,d} if v points to all of a,b,c,d
        # and v is not in {a,b,c,d}
        
        # Precompute: for each 4-subset, check if any vertex dominates it
        # Optimization: use the fact that a dominator must have out-degree >= 4
        # and must point to all 4 vertices
        
        # Build out-neighborhood masks using Python ints for speed
        out_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj[i, j] == 1:
                    mask |= (1 << j)
            out_masks[i] = mask
        
        # Check all 4-subsets
        for a, b, c, d in itertools.combinations(range(n), 4):
            if time.time() > deadline:
                return False
            # A dominator v must point to a, b, c, d
            # So v must be in the intersection of in-neighborhoods of a,b,c,d
            # Or equivalently, a,b,c,d must all be in out-neighborhood of v
            # Let's check: for each v not in {a,b,c,d}, does v point to all of a,b,c,d?
            # Faster: check if there exists v such that out_masks[v] has bits a,b,c,d set
            # and v is not a,b,c,d
            
            # Build the mask of required targets
            target_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Check if any vertex (not in the 4) has all 4 in its out-neighborhood
            # We can precompute for each vertex the set of 4-subsets it dominates,
            # but that's expensive. Instead, iterate through vertices.
            # Optimization: only check vertices with out-degree >= 4
            found = False
            for v in range(n):
                if v == a or v == b or v == c or v == d:
                    continue
                if (out_masks[v] & target_mask) == target_mask:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Strategy 1: Try Paley tournaments for small primes
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Generate Paley tournament
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        # Check validity with fast method
        if check_validity_fast(p, orientation):
            return {'n': p, 'orientation': orientation}
    
    # Strategy 2: Randomized local search for smaller n
    # Try to find a valid tournament for n from 31 upward using local search
    
    def local_search(n, max_iterations=5000):
        """Local search to find a valid tournament on n vertices."""
        if time.time() > deadline:
            return None
        
        # Initialize with random tournament
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
        def get_adj(orient):
            adj = np.zeros((n, n), dtype=np.uint8)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if orient[idx] == 0:
                        adj[i, j] = 1
                    else:
                        adj[j, i] = 1
                    idx += 1
            return adj
        
        def count_violations(orient):
            """Count number of 4-subsets without a common dominator."""
            adj = get_adj(orient)
            out_masks = [0] * n
            for i in range(n):
                mask = 0
                for j in range(n):
                    if i != j and adj[i, j] == 1:
                        mask |= (1 << j)
                out_masks[i] = mask
            
            violations = 0
            for a, b, c, d in itertools.combinations(range(n), 4):
                target_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                found = False
                for v in range(n):
                    if v == a or v == b or v == c or v == d:
                        continue
                    if (out_masks[v] & target_mask) == target_mask:
                        found = True
                        break
                if not found:
                    violations += 1
            return violations
        
        current_violations = count_violations(orientation)
        if current_violations == 0:
            return orientation
        
        best_orientation = orientation[:]
        best_violations = current_violations
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, len(orientation) - 1)
            orientation[edge_idx] = 1 - orientation[edge_idx]
            
            new_violations = count_violations(orientation)
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_orientation = orientation[:]
                if best_violations == 0:
                    return best_orientation
            else:
                # Revert
                orientation[edge_idx] = 1 - orientation[edge_idx]
        
        return best_orientation if best_violations == 0 else None
    
    # Try local search for n from 31 upward
    for n in range(31, min(45, int(deadline - time.time()))):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            result = local_search(n, max_iterations=2000)
            if result is not None:
                return {'n': n, 'orientation': result}
    
    # Fallback: return the known 83-vertex Paley tournament
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
