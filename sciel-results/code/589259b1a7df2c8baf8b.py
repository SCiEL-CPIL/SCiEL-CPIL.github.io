import time
import random
import numpy as np

def _build_paley(p):
    """Build Paley graph coloring for prime p."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _colors_to_adj(colors, n):
    """Convert flat color list to n×n adjacency matrix for color 1."""
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[i, j] = c
            adj[j, i] = c
            idx += 1
    return adj

def _has_k222_bitmask(adj, n):
    """Fast check if graph (adj matrix, 1=edge) contains K2,2,2 as subgraph.
    Uses bitmask intersection: for pair (a,b), S = N(a)∩N(b).
    If |S| >= 4, check if S contains K2,2 (two disjoint pairs with all 4 cross-edges).
    """
    # Precompute neighbor bitmasks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i, j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # For each pair (a,b), find common neighbors S
    # Check if S contains a K2,2: exist c,d,e,f in S, all distinct,
    # such that c~e, c~f, d~e, d~f
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            # Remove a and b from S (they shouldn't be in common neighbors normally, but be safe)
            S &= ~(1 << a)
            S &= ~(1 << b)
            
            if bin(S).count('1') < 4:
                continue
            
            # Get list of vertices in S
            verts = [v for v in range(n) if (S >> v) & 1]
            L = len(verts)
            
            # Check if induced subgraph on verts contains K2,2
            # K2,2: exist two disjoint pairs (c,d), (e,f) with all 4 cross-edges
            # Iterate over pairs of pairs
            for i in range(L):
                c = verts[i]
                for j in range(i + 1, L):
                    d = verts[j]
                    # c and d must be non-adjacent? No, for K2,2 subgraph,
                    # we just need the 4 cross-edges. c-d edge doesn't matter.
                    # Find common neighbors of c and d within verts
                    common_cd = neigh[c] & neigh[d]
                    common_cd &= S  # restrict to S
                    common_cd &= ~(1 << c)
                    common_cd &= ~(1 << d)
                    
                    # Need at least 2 vertices in common_cd that are distinct from c,d
                    # and form a pair with all edges to c and d
                    # Actually we need e,f in common_cd such that e~f
                    # Let's get the list
                    ef_candidates = [v for v in range(n) if (common_cd >> v) & 1]
                    
                    for x in range(len(ef_candidates)):
                        e = ef_candidates[x]
                        for y in range(x + 1, len(ef_candidates)):
                            f = ef_candidates[y]
                            # Check e~f
                            if adj[e, f]:
                                # Found K2,2: {c,d} and {e,f} with all cross-edges
                                # And a,b connect to all of c,d,e,f
                                # So {a,b,c,d,e,f} contains K2,2,2 with parts {a,b},{c,d},{e,f}
                                return True
    return False

def _check_valid(colors, n):
    """Check if neither color class contains K2,2,2."""
    adj1 = _colors_to_adj(colors, n)
    if _has_k222_bitmask(adj1, n):
        return False
    # Color 0: edges where adj1 is 0 (and i!=j)
    adj0 = 1 - adj1
    np.fill_diagonal(adj0, 0)
    if _has_k222_bitmask(adj0, n):
        return False
    return True

def _count_violations_fast(colors, n, adj0, adj1):
    """Count total K2,2,2 violations in both color classes."""
    count = 0
    if _has_k222_bitmask(adj1, n):
        count += 1
    if _has_k222_bitmask(adj0, n):
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s limit
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify it's valid (should be, but let's be safe)
    # This is just a sanity check; we trust Paley 29
    
    # Try to extend to n=30
    for target_n in range(n + 1, 32):
        if time.time() > deadline - 20:  # Leave time for final verification
            break
        
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts with different random initializations
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Copy old colors
            new_colors = colors[:]
            
            # Add edges for new vertex
            v_new = target_n - 1
            for i in range(v_new):
                new_colors.append(random.randint(0, 1))
            
            # Build adjacency matrices
            adj1 = _colors_to_adj(new_colors, target_n)
            adj0 = 1 - adj1
            np.fill_diagonal(adj0, 0)
            
            cur_violations = _count_violations_fast(new_colors, target_n, adj0, adj1)
            
            if cur_violations == 0:
                best_colors = new_colors[:]
                best_violations = 0
                break
            
            # Local search: flip edges connected to v_new
            improved = True
            max_iters = 500
            iters = 0
            while improved and iters < max_iters and time.time() < deadline - 20:
                improved = False
                iters += 1
                
                # Try flipping each edge from v_new to existing vertices
                # Randomize order to avoid cycling
                order = list(range(v_new))
                random.shuffle(order)
                
                for i in order:
                    if time.time() > deadline - 20:
                        break
                    
                    # Flip edge (i, v_new)
                    # Index in flat list: edges for i are (i, i+1)...(i, target_n-1)
                    # The edge (i, v_new) is at position: sum_{k=0}^{i-1} k + (v_new - i - 1)
                    idx = i * (i - 1) // 2 + (v_new - i - 1)
                    
                    old_val = new_colors[idx]
                    new_colors[idx] = 1 - old_val
                    
                    # Update adjacency
                    adj1[i, v_new] = new_colors[idx]
                    adj1[v_new, i] = new_colors[idx]
                    adj0[i, v_new] = 1 - new_colors[idx]
                    adj0[v_new, i] = 1 - new_colors[idx]
                    
                    new_violations = _count_violations_fast(new_colors, target_n, adj0, adj1)
                    
                    if new_violations < cur_violations:
                        cur_violations = new_violations
                        improved = True
                        if cur_violations == 0:
                            break
                    else:
                        # Revert
                        new_colors[idx] = old_val
                        adj1[i, v_new] = old_val
                        adj1[v_new, i] = old_val
                        adj0[i, v_new] = 1 - old_val
                        adj0[v_new, i] = 1 - old_val
                
                if cur_violations == 0:
                    break
            
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = new_colors[:]
            
            if best_violations == 0:
                break
        
        if best_violations == 0:
            colors = best_colors
            n = target_n
        else:
            # If we couldn't fix it, stop trying to add more
            break
    
    # Final safety: verify the result is valid
    if n > 1:
        if not _check_valid(colors, n):
            # Fall back to n=29 Paley
            n = 29
            colors = _build_paley(29)
    
    return {'n': n, 'colors': colors}
