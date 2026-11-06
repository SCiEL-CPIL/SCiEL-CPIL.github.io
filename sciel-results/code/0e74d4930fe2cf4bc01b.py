# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb

def _paleyn_tournament(p):
    """Paley tournament on p vertices: i -> j iff (j-i) mod p is a nonzero quadratic residue."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return p, orientation

def _orient_to_adj(n, orientation):
    """Convert orientation list to adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
    return adj

def _adj_to_orient(n, adj):
    """Convert adjacency matrix back to orientation list."""
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _is_valid(n, adj):
    """Check if every 4-set has a common dominator. Returns True if valid."""
    # For each 4-set {a,b,c,d}, check if any vertex dominates all 4
    # A vertex v dominates set S if adj[v, s] == 1 for all s in S
    # Precompute: for each vertex v, the set of vertices it dominates
    # For each 4-set, check if intersection of dominators is non-empty
    
    # Fast check: for each 4-set, count how many vertices dominate it
    # Use bitsets for efficiency
    # dominator_mask[v] = bitmask of vertices v dominates
    # For 4-set (a,b,c,d): a vertex v dominates it iff (dom_mask[v] >> a & 1) & (dom_mask[v] >> b & 1) & ...
    
    # For n up to 67, we can use Python integers as bitsets
    dom_masks = []
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v, u]:
                mask |= (1 << u)
        dom_masks.append(mask)
    
    # Check all 4-sets
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    mask_abcd = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    # Check if any vertex has all 4 bits set
                    found = False
                    for v in range(n):
                        if (dom_masks[v] & mask_abcd) == mask_abcd:
                            found = True
                            break
                    if not found:
                        return False
    return True

def _is_valid_fast(n, adj):
    """Faster validity check using numpy vectorization."""
    # adj is n x n uint8
    # For each 4-set, we need to check if there exists v such that
    # sum(adj[v, S]) == 4 for S being the 4-set
    
    # Precompute column sums for each subset? Too expensive.
    # Alternative: for each vertex v, compute which 4-sets it covers.
    # A 4-set is covered if at least one vertex covers it.
    
    # Use the fact that for n=67, C(67,4) = 736281 which is manageable
    # but checking each against all 67 vertices is 67*736281 ≈ 49M ops
    
    # Better: for each vertex v, the 4-sets it covers are all 4-subsets of out-neighborhood
    # Mark all such 4-sets as covered.
    
    # Represent 4-sets by their index in a precomputed list
    # For n=67, that's 736281 4-sets. Use a boolean array.
    
    # Actually, let's use a different approach:
    # For each 4-set, the number of dominators = sum over v of [v dominates all 4]
    # = sum_v prod_{s in S} adj[v,s]
    # We need this to be > 0 for all S.
    
    # For efficiency, iterate over vertices and mark covered 4-sets.
    covered = np.zeros(n, dtype=bool)  # placeholder
    
    # Let's just do the straightforward check but with early termination
    # and numpy for the inner loop
    adj_np = adj.astype(np.uint8)
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if any v dominates {a,b,c,d}
                    # adj[v,a] & adj[v,b] & adj[v,c] & adj[v,d]
                    # Sum over v
                    s = np.sum(adj_np[:, a] & adj_np[:, b] & adj_np[:, c] & adj_np[:, d])
                    if s == 0:
                        return False
    return True

def _local_search_vectorized(n, max_time=5.0, seed=42):
    """
    Vectorized local search for finding a valid tournament on n vertices.
    Returns orientation list if found, else None.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed + n)
    
    # Initialize randomly
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i+1, n):
            if rng.integers(0, 2):
                adj[j, i] = 1
            else:
                adj[i, j] = 1
    
    # Precompute out-neighborhoods as sets
    out_nb = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                out_nb[i].add(j)
    
    # Count violations: number of 4-sets with no dominator
    def count_violations():
        # For each 4-set, check if any vertex dominates it
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Check if any v has a,b,c,d all in out_nb[v]
                        found = False
                        for v in range(n):
                            if a in out_nb[v] and b in out_nb[v] and c in out_nb[v] and d in out_nb[v]:
                                found = True
                                break
                        if not found:
                            violations += 1
        return violations
    
    current_violations = count_violations()
    if current_violations == 0:
        return _adj_to_orient(n, adj)
    
    best_violations = current_violations
    best_adj = adj.copy()
    
    # Local search with restarts
    max_restarts = 5
    per_restart_time = max_time / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > start_time + max_time:
            break
        
        # Fresh random init for each restart
        if restart > 0:
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i+1, n):
                    if rng.integers(0, 2):
                        adj[j, i] = 1
                    else:
                        adj[i, j] = 1
            out_nb = [set() for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if adj[i, j]:
                        out_nb[i].add(j)
            current_violations = count_violations()
            if current_violations == 0:
                return _adj_to_orient(n, adj)
        
        restart_start = time.time()
        
        while time.time() - restart_start < per_restart_time:
            # Pick random edge to flip
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            # Determine current direction
            if adj[i, j]:
                u, v = i, j  # i -> j, flip to j -> i
            else:
                u, v = j, i  # j -> i, flip to i -> j
            
            # u dominates v currently. After flip, v dominates u.
            # Update out_nb
            out_nb[u].discard(v)
            out_nb[v].add(u)
            adj[u, v] = 0
            adj[v, u] = 1
            
            new_violations = count_violations()
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_adj = adj.copy()
                if new_violations == 0:
                    return _adj_to_orient(n, adj)
            elif new_violations > current_violations:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 0
                out_nb[u].add(v)
                out_nb[v].discard(u)
            else:
                current_violations = new_violations
    
    # Return best found
    if best_violations == 0:
        return _adj_to_orient(n, best_adj)
    return None

def _prune_tournament(n, adj, max_time=30.0):
    """
    Try to prune vertices from a valid tournament.
    Returns (new_n, new_orientation) if a smaller valid tournament is found.
    """
    start_time = time.time()
    
    # First, identify which vertices are "essential" (unique dominators of some 4-set)
    # If a vertex is not a unique dominator of any 4-set, we might be able to remove it.
    # Actually, we can remove a vertex v if after removal, every 4-set in the remaining
    # graph still has a dominator.
    
    # Simpler approach: try removing each vertex one at a time and check validity.
    # If valid, keep the removal and continue.
    
    current_n = n
    current_adj = adj.copy()
    remaining = list(range(n))
    
    while time.time() - start_time < max_time:
        removed_any = False
        for v in list(remaining):
            if time.time() - start_time > max_time:
                break
            # Try removing v
            new_remaining = [u for u in remaining if u != v]
            if len(new_remaining) < 5:
                continue
            
            # Build new adjacency
            new_n = len(new_remaining)
            idx_map = {u: i for i, u in enumerate(new_remaining)}
            new_adj = np.zeros((new_n, new_n), dtype=np.uint8)
            for i, ui in enumerate(new_remaining):
                for j, uj in enumerate(new_remaining):
                    new_adj[i, j] = current_adj[ui, uj]
            
            # Check validity
            if _is_valid_fast(new_n, new_adj):
                # Remove v
                remaining = new_remaining
                current_n = new_n
                current_adj = new_adj
                removed_any = True
                break  # Restart the loop since indices changed
        
        if not removed_any:
            break
    
    if current_n < n:
        orientation = _adj_to_orient(current_n, current_adj)
        return current_n, orientation
    return n, _adj_to_orient(n, adj)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    # Strategy:
    # 1. Generate Paley 67 (known valid)
    # 2. Try to prune it
    # 3. Try local search for smaller n if time permits
    
    # Step 1: Paley 67
    n_paley, orient_paley = _paleyn_tournament(67)
    adj_paley = _orient_to_adj(n_paley, orient_paley)
    
    # Verify it's valid (should be, but just in case)
    if not _is_valid_fast(n_paley, adj_paley):
        # Fallback: try other primes
        for p in [83, 79, 71, 67, 63, 59, 53, 47, 43, 37, 31]:
            if p % 4 == 3:
                n_try, orient_try = _paleyn_tournament(p)
                adj_try = _orient_to_adj(n_try, orient_try)
                if _is_valid_fast(n_try, adj_try):
                    n_paley, orient_paley, adj_paley = n_try, orient_try, adj_try
                    break
    
    # Step 2: Try to prune
    time_left = deadline - time.time()
    if time_left > 10:
        prune_time = min(30.0, time_left * 0.5)
        n_pruned, orient_pruned = _prune_tournament(n_paley, adj_paley, max_time=prune_time)
        if n_pruned < n_paley:
            return {'n': n_pruned, 'orientation': orient_pruned}
    
    # Step 3: Try local search for smaller n
    time_left = deadline - time.time()
    if time_left > 5:
        # Try a few values
        for n_try in [40, 45, 50, 55, 60, 65]:
            if time.time() > deadline - 5:
                break
            lt = min(5.0, (deadline - time.time()) / 3)
            if lt < 1:
                break
            result = _local_search_vectorized(n_try, max_time=lt)
            if result is not None:
                return {'n': n_try, 'orientation': result}
    
    # Fallback: return Paley 67
    return {'n': n_paley, 'orientation': orient_paley}
# EVOLVE-BLOCK-END
