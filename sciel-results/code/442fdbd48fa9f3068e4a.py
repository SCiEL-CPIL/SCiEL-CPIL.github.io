"""Combined program: Paley construction + local search refinement for Schütte k=4."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_adj(n, orientation):
    """Build adjacency matrix from orientation list."""
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

def _count_violations_fast(n, adj):
    """
    Count how many 4-subsets lack a common dominator.
    For each 4-set {a,b,c,d}, check if any vertex v dominates all 4.
    Uses vectorized operations for speed.
    """
    # For each vertex v, compute which 4-sets it dominates
    # A vertex v dominates a 4-set S if v->s for all s in S
    # Precompute: for each v, the set of vertices it dominates
    # Then for each 4-set, check intersection
    
    # Faster approach: for each 4-set, check all n vertices
    # But that's O(n^5) which is too slow for large n
    # 
    # Alternative: for each vertex v, find all 4-sets it dominates
    # v dominates 4-set {a,b,c,d} iff a,b,c,d are all in out_neigh(v)
    # So the number of 4-sets dominated by v is C(out_degree(v), 4)
    # But we need to check if EVERY 4-set is dominated by AT LEAST ONE vertex
    
    # Let's use a different approach:
    # For each 4-set, compute the set of dominators (vertices that point to all 4)
    # A 4-set is "bad" if its dominator set is empty
    
    # Efficient method using bitsets:
    # Represent out-neighborhood of each vertex as a bitset
    # For a 4-set {a,b,c,d}, the dominators are: AND of out-neighborhoods of... 
    # No wait. v dominates {a,b,c,d} means v->a, v->b, v->c, v->d
    # So dominators of {a,b,c,d} = {v : v->a AND v->b AND v->c AND v->d}
    # This is the intersection of the IN-neighborhoods of a, b, c, d
    # (vertices that have arcs TO a, b, c, d)
    
    # in_neigh[x] = set of vertices v such that v -> x
    # dominators({a,b,c,d}) = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
    
    # Build in-neighborhood bitsets
    # in_neigh[x] is a bitset where bit v is set if v -> x
    in_neigh = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j, i] = 1  # i -> j, so i is in in_neigh[j]
    
    # For each 4-set, check if in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d] is non-empty
    # This is expensive. Let's sample instead for large n.
    
    return None  # Not used directly; we'll use sampling in the main check

def _is_valid_sampled(n, adj, num_samples=5000, seed=42):
    """Check validity by sampling random 4-sets."""
    rng = np.random.default_rng(seed)
    
    # Precompute in-neighborhoods as lists for quick checking
    in_neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j].add(i)
    
    for _ in range(num_samples):
        S = rng.choice(n, size=4, replace=False)
        a, b, c, d = S
        # Dominators = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        dominators = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        if len(dominators) == 0:
            return False
    return True

def _is_valid_exhaustive(n, adj):
    """Exhaustively check all 4-sets. Only feasible for small n."""
    if n > 35:
        return None  # Too expensive
    
    # Precompute in-neighborhoods as bitsets (using integers for speed)
    in_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_bits[j] |= (1 << i)
    
    from itertools import combinations
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        dominators = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
        if dominators == 0:
            return False
    return True

def _paleyn_tournament(p):
    """Paley tournament on p vertices."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return p, orientation

def _local_search(n, orientation, max_iterations=2000, deadline=None):
    """
    Local search: try flipping edges to reduce violations.
    Returns (n, orientation, valid) for the best found.
    """
    adj = _build_adj(n, orientation)
    
    # Precompute in-neighborhoods as sets
    in_neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j].add(i)
    
    def count_violations_sampled(num_samples=3000, seed=123):
        """Count how many sampled 4-sets are violated."""
        rng = np.random.default_rng(seed)
        violations = 0
        for _ in range(num_samples):
            S = rng.choice(n, size=4, replace=False)
            a, b, c, d = S
            dominators = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
            if len(dominators) == 0:
                violations += 1
        return violations
    
    # Initial violation count
    current_violations = count_violations_sampled()
    
    best_orientation = orientation[:]
    best_violations = current_violations
    
    if best_violations == 0:
        return n, best_orientation, True
    
    # Try flipping random edges
    num_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(456)
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random edge to flip
        pair_idx = rng.integers(0, num_pairs)
        # Convert linear index to (i, j)
        # Linear index for (i,j) with i<j: i*(2n-i-1)//2 + (j-i-1)
        # Inverse: find i such that i*(2n-i-1)//2 <= pair_idx < (i+1)*(2n-i-2)//2
        i = 0
        while i < n - 1:
            start = i * (2 * n - i - 1) // 2
            end = (i + 1) * (2 * n - i - 2) // 2
            if start <= pair_idx < end:
                break
            i += 1
        j = i + 1 + (pair_idx - i * (2 * n - i - 1) // 2)
        
        # Flip edge (i, j)
        old_dir = adj[i, j]
        # Reverse
        adj[i, j] = 1 - old_dir
        adj[j, i] = old_dir
        
        # Update in_neigh
        if old_dir == 0:  # was i -> j, now j -> i
            in_neigh[j].discard(i)
            in_neigh[i].add(j)
        else:  # was j -> i, now i -> j
            in_neigh[i].discard(j)
            in_neigh[j].add(i)
        
        new_violations = count_violations_sampled()
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_orientation = orientation[:]
            # Rebuild orientation from adj
            idx = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    best_orientation[idx] = 0 if adj[ii, jj] == 1 else 1
                    idx += 1
            current_violations = new_violations
        else:
            # Revert
            adj[i, j] = old_dir
            adj[j, i] = 1 - old_dir
            if old_dir == 0:
                in_neigh[j].add(i)
                in_neigh[i].discard(j)
            else:
                in_neigh[i].add(j)
                in_neigh[j].discard(i)
    
    # Final validity check
    valid = _is_valid_sampled(n, adj, num_samples=10000, seed=789)
    
    return n, best_orientation, valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer)
    
    # Strategy:
    # 1. Start with Paley tournament on smallest valid prime p ≡ 3 (mod 4) >= 31
    # 2. Try to improve via local search
    # 3. Also try random tournaments for small n as a fallback
    
    best_n = 91
    best_orientation = None
    
    # Try Paley tournaments
    primes_to_try = []
    for p in range(31, 91):
        if p >= 2:
            is_prime = True
            for x in range(2, int(p**0.5) + 1):
                if p % x == 0:
                    is_prime = False
                    break
            if is_prime:
                primes_to_try.append(p)
    
    valid_paleys = [p for p in primes_to_try if p % 4 == 3]
    
    # Try each Paley from smallest
    for p in valid_paleys:
        if time.time() > deadline - 20:
            break
        
        n, orientation = _paleyn_tournament(p)
        
        # Quick validity check
        adj = _build_adj(n, orientation)
        
        if n <= 35:
            valid = _is_valid_exhaustive(n, adj)
        else:
            valid = _is_valid_sampled(n, adj, num_samples=5000, seed=42)
        
        if valid:
            if n < best_n:
                best_n = n
                best_orientation = orientation
            break  # Found a valid Paley, no need to try larger ones
        
        # If not valid, try local search
        if time.time() > deadline - 15:
            continue
        
        n_ls, orient_ls, valid_ls = _local_search(
            n, orientation, 
            max_iterations=1000,
            deadline=deadline - 5
        )
        
        if valid_ls and n_ls < best_n:
            best_n = n_ls
            best_orientation = orient_ls
            break
    
    # If no valid found yet, fall back to p=83
    if best_orientation is None:
        n, orientation = _paleyn_tournament(83)
        best_n = n
        best_orientation = orientation
    
    # Final time check
    if time.time() > deadline:
        pass
    
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
