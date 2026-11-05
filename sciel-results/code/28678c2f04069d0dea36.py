import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _make_paley(p):
    """Build a Paley tournament on p vertices (p ≡ 3 mod 4 prime)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = [0] * (p * (p - 1) // 2)
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation[idx] = 0
            else:
                orientation[idx] = 1
            idx += 1
    return p, orientation

def _is_valid_fast(n, orientation, timeout=10):
    """
    Check if tournament satisfies: every 4-vertex set has a common dominator.
    Use bitset representation for speed. Returns (valid, partial_check_count).
    """
    # Build adjacency: adj[i] is a Python int bitmask of out-neighbors of i
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                adj[i] |= (1 << j)
            else:
                # j -> i
                adj[j] |= (1 << i)
            idx += 1
    
    # For each 4-set, check if there exists a vertex v with all 4 as out-neighbors
    # Equivalently: for every 4-set S, the intersection of out-neighborhoods of S is non-empty
    # Actually we need: exists v such that v has arcs to ALL four vertices in S.
    # That means v is NOT in S, and v -> all elements of S.
    # So we need: for every 4-set S, there exists v ∉ S such that adj[v] contains all bits in S.
    
    # Alternative: for every 4-set S, check if any v ∉ S has adj[v] & mask_S == mask_S
    # where mask_S is the bitmask of S.
    
    start_time = time.time()
    checked = 0
    all_vertices = (1 << n) - 1
    
    for combo in combinations(range(n), 4):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        # We need a vertex v NOT in combo such that adj[v] & mask == mask
        # Iterate over all vertices not in combo
        # For speed: we can precompute for each 4-set whether it's dominated
        # But let's just check: is there any vertex outside S that dominates all of S?
        
        # A vertex v dominates S if v ∉ S and (adj[v] & mask) == mask
        # We can check: does any v in all_vertices \\ mask have adj[v] containing mask?
        
        # Efficient: for each vertex v, we can precompute the 4-sets it dominates.
        # But that's expensive. Instead, for each 4-set, check if any candidate works.
        
        # Candidates: vertices not in S. For n=67, that's 63 candidates.
        # But we can be smarter: only vertices with out-degree >= 4 can possibly dominate a 4-set.
        # Even better: for a given mask, we need adj[v] & mask == mask, i.e., mask ⊆ adj[v].
        
        # Let's just iterate through all vertices not in S
        found = False
        for v in range(n):
            if mask & (1 << v):
                continue  # v is in S, skip
            if (adj[v] & mask) == mask:
                found = True
                break
        
        checked += 1
        if not found:
            return (False, checked)
        
        if time.time() - start_time > timeout:
            return (None, checked)  # timeout
    
    return (True, checked)

def _validate_full(n, orientation):
    """Full validation without timeout."""
    return _is_valid_fast(n, orientation, timeout=120)[0]

def _local_search(n, orientation, time_limit=30):
    """
    Local search: try to flip edges to fix violations.
    Strategy: find a violating 4-set, then try flipping one edge in it to fix.
    """
    start_time = time.time()
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            orientation[idx] = 0  # reset
            idx += 1
    
    # Build adj
    def build_adj(orientation):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    adj[i] |= (1 << j)
                else:
                    adj[j] |= (1 << i)
                idx += 1
        return adj
    
    def find_violation(adj):
        for combo in combinations(range(n), 4):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            for v in range(n):
                if mask & (1 << v):
                    continue
                if (adj[v] & mask) == mask:
                    break
            else:
                return combo
        return None
    
    def get_edge_index(i, j):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)
    
    def set_edge(orientation, adj, i, j, val):
        if i > j:
            i, j = j, i
        idx = get_edge_index(i, j)
        old = orientation[idx]
        if old != val:
            orientation[idx] = val
            if val == 0:
                # i -> j
                adj[i] |= (1 << j)
                adj[j] &= ~(1 << i)
            else:
                # j -> i
                adj[j] |= (1 << i)
                adj[i] &= ~(1 << j)
    
    adj = build_adj(orientation)
    
    while time.time() - start_time < time_limit:
        violation = find_violation(adj)
        if violation is None:
            return True  # valid
        
        # Try to fix by flipping one edge within the violation or between violation and a dominator
        a, b, c, d = violation
        vertices = [a, b, c, d]
        
        # For each pair in the violation, try flipping
        fixed = False
        for i_idx in range(4):
            for j_idx in range(i_idx + 1, 4):
                i, j = vertices[i_idx], vertices[j_idx]
                idx = get_edge_index(i, j)
                old_val = orientation[idx]
                new_val = 1 - old_val
                # Apply flip
                set_edge(orientation, adj, i, j, new_val)
                
                # Check if this fixes the violation
                # Rebuild check for this specific 4-set
                mask = 0
                for v in vertices:
                    mask |= (1 << v)
                dominated = False
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    if (adj[v] & mask) == mask:
                        dominated = True
                        break
                
                if dominated:
                    fixed = True
                    break
                else:
                    # Revert
                    set_edge(orientation, adj, i, j, old_val)
            
            if fixed:
                break
        
        if not fixed:
            # Try random flips
            for _ in range(100):
                if time.time() - start_time >= time_limit:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                idx = get_edge_index(i, j)
                old_val = orientation[idx]
                new_val = 1 - old_val
                set_edge(orientation, adj, i, j, new_val)
            
            # Re-check
            violation = find_violation(adj)
            if violation is None:
                return True
    
    # Final check
    return find_violation(adj) is None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with Paley tournament on 83 vertices
    p, orientation = _make_paley(83)
    
    # Try to reduce n by removing vertices
    # A vertex can be removed if every 4-set in the remaining graph still has a dominator
    # Strategy: try to remove vertices one by one, checking validity
    
    def try_remove_vertex(n, orientation, v_to_remove):
        """Try to remove vertex v_to_remove and check if the result is valid."""
        # Build new orientation without v_to_remove
        new_n = n - 1
        new_orientation = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i == v_to_remove or j == v_to_remove:
                    idx += 1
                    continue
                new_i = i if i < v_to_remove else i - 1
                new_j = j if j < v_to_remove else j - 1
                new_orientation.append(orientation[idx])
                idx += 1
        
        # Validate
        valid, _ = _is_valid_fast(new_n, new_orientation, timeout=5)
        if valid:
            return new_n, new_orientation, True
        return n, orientation, False
    
    # Greedy vertex removal
    current_n = p
    current_orientation = orientation
    
    # Order vertices by degree (out-degree + in-degree is always n-1, so use out-degree)
    # Vertices with extreme out-degrees might be more critical
    def compute_outdegrees(n, orientation):
        out_deg = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    out_deg[i] += 1
                else:
                    out_deg[j] += 1
                idx += 1
        return out_deg
    
    out_degrees = compute_outdegrees(current_n, current_orientation)
    # Try removing vertices with extreme out-degrees first (more likely to be "redundant" or "critical")
    # Actually, vertices that are dominators for many 4-sets are critical.
    # Let's just try all vertices in order of out-degree (ascending and descending)
    
    order = sorted(range(current_n), key=lambda v: (out_degrees[v], v))
    # Try both ascending and descending
    for order_list in [order, list(reversed(order))]:
        for v in order_list:
            if time.time() > deadline:
                break
            if v >= current_n:
                continue
            new_n, new_orient, success = try_remove_vertex(current_n, current_orientation, v)
            if success:
                current_n = new_n
                current_orientation = new_orient
                # Recompute out-degrees
                out_degrees = compute_outdegrees(current_n, current_orientation)
                order = sorted(range(current_n), key=lambda v: (out_degrees[v], v))
                break
        if time.time() > deadline:
            break
    
    # Local search to improve
    if time.time() < deadline - 20:
        local_search(current_n, current_orientation, time_limit=min(40, deadline - time.time() - 5))
    
    # Final validation
    valid, _ = _is_valid_fast(current_n, current_orientation, timeout=30)
    if valid:
        return {'n': current_n, 'orientation': current_orientation}
    
    # Fallback: return original Paley
    return {'n': p, 'orientation': list(orientation)}

# EVOLVE-BLOCK-END
