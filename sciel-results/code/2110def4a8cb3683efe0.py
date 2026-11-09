import time
import random
import numpy as np

def _color_index(n, i, j):
    # Index in the flat list for edge (i, j) with i < j
    return i*(i-1)//2 + (j - i - 1)

def _build_paley(p):
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

def _build_adj_matrices(colors, n):
    """
    Build two adjacency matrices (numpy arrays) for color 0 and color 1.
    adj0[i,j] = 1 if edge (i,j) is color 0, else 0.
    adj1[i,j] = 1 if edge (i,j) is color 1, else 0.
    """
    adj0 = np.zeros((n, n), dtype=np.int8)
    adj1 = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i+1, n):
            idx = _color_index(n, i, j)
            c = colors[idx]
            if c == 0:
                adj0[i, j] = 1
                adj0[j, i] = 1
            else:
                adj1[i, j] = 1
                adj1[j, i] = 1
    return adj0, adj1

def _count_violations_fast(adj0, adj1, n, v):
    """
    Count monochromatic K2,2,2 subgraphs involving vertex v.
    Uses numpy for speed.
    """
    violations = 0
    
    # Neighbors in adj0 and adj1
    neigh0 = np.where(adj0[v] == 1)[0]
    neigh1 = np.where(adj1[v] == 1)[0]
    
    def check_color(neigh, adj):
        count = 0
        L = len(neigh)
        if L < 4:
            return 0
        
        # For each pair of neighbors (a, b) in neigh,
        # we need to find if there exist two other neighbors (c, d)
        # such that edges (a,c), (a,d), (b,c), (b,d) are all in the graph (color c).
        
        # Efficient way:
        # For each pair (a, b) in neigh:
        #   common = intersection of neighbors of a and b within neigh
        #   if |common| >= 2, then we have a K2,2,2.
        #   Specifically, if we pick any two from common, say c and d,
        #   then {a, b} and {c, d} form a K2,2,2.
        
        # Let's iterate over pairs (a, b) in neigh.
        # There are C(L, 2) pairs.
        # For each pair, compute common neighbors in neigh.
        
        # To make it faster, we can precompute the subgraph induced by neigh.
        # But L is small (<= 30).
        
        for i in range(L):
            a = neigh[i]
            for j in range(i+1, L):
                b = neigh[j]
                # Find common neighbors of a and b within neigh
                # adj[a] & adj[b] gives common neighbors in the whole graph.
                # We need to intersect with neigh.
                common = np.intersect1d(neigh, np.where((adj[a] & adj[b]) == 1)[0])
                if len(common) >= 2:
                    # Number of ways to choose 2 from common is C(len(common), 2)
                    # But we just need to count if there is at least one.
                    # Actually, the problem asks to count subgraphs.
                    # Each K2,2,2 is defined by 6 vertices.
                    # My previous logic counted partitions.
                    # Let's stick to a consistent counting method or just check validity (count > 0).
                    # For local search, minimizing count is good.
                    # C(len(common), 2) is the number of K2,2,2s with partition {a,b} | {c,d}.
                    count += len(common) * (len(common) - 1) // 2
        return count
    
    violations += check_color(neigh0, adj0)
    violations += check_color(neigh1, adj1)
    
    return violations

def _is_valid_fast(adj0, adj1, n):
    total = 0
    for v in range(n):
        total += _count_violations_fast(adj0, adj1, n, v)
    return total == 0

def _try_extend(colors, n, target_n, deadline):
    """
    Try to extend a valid coloring of K_n to K_target_n.
    Uses local search on the new edges.
    """
    if n >= target_n:
        return colors, n
    
    best_colors = None
    best_violations = float('inf')
    
    # Number of restarts based on time available
    # Let's do a fixed number of restarts, but break early if time is low.
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize new edges randomly
        cur_colors = colors[:]
        for v in range(n, target_n):
            for i in range(v):
                cur_colors.append(random.randint(0, 1))
        
        # Build adjacency matrices for the full graph
        adj0, adj1 = _build_adj_matrices(cur_colors, target_n)
        
        # Calculate initial violations
        # We only care about violations involving new vertices?
        # No, flipping edges between new vertices doesn't affect old vertices.
        # Flipping edges between old and new vertices affects old vertices too?
        # Yes. If I flip edge (old, new), it might create a K2,2,2 involving 'old'.
        # So I must check all vertices?
        # That is expensive.
        
        # Optimization:
        # When extending from n to n+1, only edges involving n+1 are new.
        # If I flip an edge (u, n+1), it only affects K2,2,2s that include edge (u, n+1).
        # Such K2,2,2s must include u and n+1.
        # So I only need to check vertices u and n+1?
        # No, a K2,2,2 is a set of 6 vertices.
        # If it contains edge (u, n+1), it contains vertices u and n+1.
        # It might contain other old vertices.
        # So checking violations for u and n+1 is sufficient to catch any NEW violations
        # introduced by flipping (u, n+1), provided the rest of the graph was valid.
        # Wait, if the rest of the graph was valid, then any K2,2,2 in the new graph
        # must contain at least one new edge.
        # If I only flip edges incident to the new vertex, then any new K2,2,2
        # must contain the new vertex.
        # So checking violations for the new vertex is sufficient!
        
        # Let's verify:
        # Start with valid K_n. Add v_new.
        # Edges (u, v_new) are the only new edges.
        # Any K2,2,2 in K_{n+1} must use at least one edge (u, v_new).
        # Thus it must contain v_new.
        # So _count_violations_fast(adj0, adj1, n+1, v_new) == 0 implies validity.
        
        # This holds for adding one vertex at a time.
        # If I add multiple vertices at once, it's more complex.
        # Let's add one by one to be safe and simple.
        
        cur_n = n
        cur_adj0, cur_adj1 = adj0, adj1
        cur_violations = _count_violations_fast(cur_adj0, cur_adj1, cur_n, cur_n - 1) # Check last added? No, we are initializing all at once.
        
        # Actually, let's just add one vertex at a time in the loop.
        # But the current code structure adds all at once.
        # Let's stick to adding one vertex at a time for the local search logic to be simple.
        
        # Reset for single vertex extension
        cur_colors = colors[:]
        cur_n = n
        
        success = True
        for v_new in range(n, target_n):
            if time.time() > deadline:
                success = False
                break
                
            # Add edges for v_new
            for i in range(v_new):
                cur_colors.append(random.randint(0, 1))
            
            cur_n = v_new + 1
            cur_adj0, cur_adj1 = _build_adj_matrices(cur_colors, cur_n)
            
            cur_violations = _count_violations_fast(cur_adj0, cur_adj1, cur_n, v_new)
            
            if cur_violations == 0:
                continue
                
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping each edge (i, v_new)
                for i in range(v_new):
                    if time.time() > deadline:
                        success = False
                        break
                    idx = _color_index(cur_n, i, v_new)
                    old_val = cur_colors[idx]
                    cur_colors[idx] = 1 - old_val
                    
                    # Update adjacency matrices
                    if old_val == 0:
                        cur_adj0[i, v_new] = 0
                        cur_adj0[v_new, i] = 0
                        cur_adj1[i, v_new] = 1
                        cur_adj1[v_new, i] = 1
                    else:
                        cur_adj1[i, v_new] = 0
                        cur_adj1[v_new, i] = 0
                        cur_adj0[i, v_new] = 1
                        cur_adj0[v_new, i] = 1
                        
                    new_violations = _count_violations_fast(cur_adj0, cur_adj1, cur_n, v_new)
                    
                    if new_violations < cur_violations:
                        cur_violations = new_violations
                        improved = True
                        if cur_violations == 0:
                            break
                    else:
                        # Revert
                        cur_colors[idx] = old_val
                        if old_val == 0:
                            cur_adj0[i, v_new] = 1
                            cur_adj0[v_new, i] = 1
                            cur_adj1[i, v_new] = 0
                            cur_adj1[v_new, i] = 0
                        else:
                            cur_adj1[i, v_new] = 1
                            cur_adj1[v_new, i] = 1
                            cur_adj0[i, v_new] = 0
                            cur_adj0[v_new, i] = 0
                            
                if cur_violations == 0:
                    break
                if not success:
                    break
                    
            if cur_violations > 0:
                success = False
                break
                
        if success:
            return cur_colors, cur_n
            
    return best_colors, n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Start with Paley 29
    p = 29
    n = p
    colors = _build_paley(p)
    
    # Verify Paley 29 is valid
    adj0, adj1 = _build_adj_matrices(colors, n)
    if not _is_valid_fast(adj0, adj1, n):
        # Should not happen, but just in case
        pass
        
    # Try to extend to 30, then 31
    for target_n in [30, 31]:
        if time.time() > deadline:
            break
            
        new_colors, new_n = _try_extend(colors, n, target_n, deadline)
        
        if new_n == target_n:
            colors = new_colors
            n = new_n
        else:
            break
            
    return {'n': n, 'colors': colors}
