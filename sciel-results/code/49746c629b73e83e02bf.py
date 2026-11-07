import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and edge sets for quick lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute edge counts for all subsets to quickly identify cliques and independent sets
    # edge_count[mask] = number of edges induced by the set of vertices in mask
    size = 1 << n
    edge_count = [0] * size
    
    # DP to compute edge counts
    # For each mask, pick the lowest set bit v.
    # edge_count[mask] = edge_count[mask without v] + (number of neighbors of v in mask without v)
    # We can precompute neighbor masks for each vertex
    neigh_mask = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if adj[u][v]:
                m |= (1 << v)
        neigh_mask[u] = m

    for mask in range(1, size):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        # Count edges between v and prev_mask
        # This is the popcount of (neigh_mask[v] & prev_mask)
        common = neigh_mask[v] & prev_mask
        edge_count[mask] = edge_count[prev_mask] + common.bit_count()

    # Identify all valid modules (cliques or independent sets)
    # A set is a clique if edge_count[mask] == k*(k-1)/2 where k is popcount
    # A set is independent if edge_count[mask] == 0
    # We exclude the empty set
    valid_masks = []
    for mask in range(1, size):
        k = mask.bit_count()
        e = edge_count[mask]
        if e == 0 or e == k * (k - 1) // 2:
            valid_masks.append(mask)
            
    if not valid_masks:
        # Fallback: single vertices are always valid (k=1, e=0, independent)
        # This should never happen for n>=1
        valid_masks = [1 << i for i in range(n)]

    # Formulate ILP
    # Variables: x_j for each valid mask j
    m = len(valid_masks)
    
    # Objective: minimize sum(x_j)
    c = np.ones(m)
    
    # Constraints: sum_{j: v in S_j} x_j = 1 for each vertex v
    # A is n x m
    A = np.zeros((n, m))
    for j, mask in enumerate(valid_masks):
        for v in range(n):
            if (mask >> v) & 1:
                A[v, j] = 1
                
    b = np.ones(n)
    
    # Bounds: x_j in [0, 1]
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    # Integrality: all binary
    integrality = np.ones(m)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb=b, ub=b), bounds=bounds, integrality=integrality)
        if res.status == 0:
            # Extract solution
            parts = []
            for j in range(m):
                if res.x[j] > 0.5:
                    mask = valid_masks[j]
                    part = [v for v in range(n) if (mask >> v) & 1]
                    parts.append(part)
            
            # Verify and format
            if len(parts) > 0:
                # Check if all vertices are covered exactly once
                seen = [0] * n
                for part in parts:
                    for v in part:
                        seen[v] += 1
                if all(count == 1 for count in seen):
                    # Sort parts and vertices for consistent output
                    for part in parts:
                        part.sort()
                    parts.sort(key=lambda x: x[0])
                    return [[int(v) for v in part] for part in parts]
        # If solver fails or returns invalid, fall back
    except Exception:
        pass

    # Fallback: Use the previous DP approach if ILP fails or times out
    # We implement a simplified version of the seed's logic as a robust fallback
    return _fallback_dp()

def _fallback_dp():
    """
    Fallback using the original DP approach if SciPy ILP fails.
    """
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Generate maximal cliques and independent sets
    # For simplicity in fallback, we just use single vertices and try to merge
    # Or better, reuse the candidate generation from the seed but simplified
    
    # Actually, let's just do a greedy fallback if ILP fails, as the ILP should work.
    # But to be safe, let's implement a quick greedy.
    parts = []
    used = [False] * n
    for v in range(n):
        if used[v]:
            continue
        # Try to find a large set containing v that is clique or indep
        # Simple greedy: start with {v}, add vertices that keep it clique or indep
        # This is O(N^3) roughly, fine for N=14
        
        # Let's just use single vertices if everything else fails? No, that's N parts.
        # Let's try a bit better greedy.
        
        # Actually, the ILP is very robust. If it fails, something is wrong with SciPy.
        # Let's implement the exact DP from the seed as the ultimate fallback.
        
        return _exact_dp_fallback()

def _exact_dp_fallback():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute valid masks (cliques and independent sets)
    size = 1 << n
    valid_masks = [False] * size
    valid_masks[0] = True # Empty set is valid but we don't use it as a part
    
    # Use bit operations to check validity
    # A mask is valid if it's a clique or independent set.
    # We can check this by iterating through vertices.
    # For speed, we can precompute edge_count as before, but here we just do it directly.
    
    # Let's reuse the edge_count logic
    neigh_mask = [0] * n
    for u in range(n):
        m = 0
        for v in range(n):
            if adj[u][v]:
                m |= (1 << v)
        neigh_mask[u] = m

    edge_count = [0] * size
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        common = neigh_mask[v] & prev_mask
        edge_count[mask] = edge_count[prev_mask] + common.bit_count()
        
    valid_indices = []
    for mask in range(1, size):
        k = mask.bit_count()
        e = edge_count[mask]
        if e == 0 or e == k * (k - 1) // 2:
            valid_indices.append(mask)
            
    # DP over masks
    # dp[mask] = min parts to cover mask
    dp = [100] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Try all valid subsets of mask
        sub = mask
        while sub:
            if valid_indices and sub in valid_indices: # This check is O(M), slow.
                # Better: check if sub is valid directly
                k = sub.bit_count()
                e = edge_count[sub]
                if e == 0 or e == k * (k - 1) // 2:
                    rest = mask ^ sub
                    if dp[rest] + 1 < dp[mask]:
                        dp[mask] = dp[rest] + 1
            sub = (sub - 1) & mask
        # Optimization: We can break if we find a solution equal to lower bound
        # Lower bound is 1 if mask itself is valid
        k = mask.bit_count()
        e = edge_count[mask]
        if e == 0 or e == k * (k - 1) // 2:
            dp[mask] = 1
            continue # No need to check submasks if mask itself is a part

    # Reconstruct
    parts = []
    mask = (1 << n) - 1
    while mask:
        sub = mask
        found = False
        while sub:
            k = sub.bit_count()
            e = edge_count[sub]
            if e == 0 or e == k * (k - 1) // 2:
                rest = mask ^ sub
                if dp[rest] + 1 == dp[mask]:
                    part = [v for v in range(n) if (sub >> v) & 1]
                    part.sort()
                    parts.append(part)
                    mask = rest
                    found = True
                    break
            sub = (sub - 1) & mask
        if not found:
            # Should not happen
            break
            
    parts.sort(key=lambda x: x[0])
    return [[int(v) for v in part] for part in parts]

if __name__ == '__main__':
    res = construct()
    print(res)
