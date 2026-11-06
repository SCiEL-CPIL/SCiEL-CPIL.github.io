import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def generate_projective_plane_graph(q=5):
    """
    Generate a graph from the projective plane PG(2,q) of order q.
    Vertices are the q^2+q+1 points. Two points are adjacent if the line
    through them has a specific property (e.g., based on slope parity or
    a quadratic residue condition on the slope).
    
    For q=5: 31 vertices. We use: two points are adjacent iff the slope
    of the line through them is a quadratic residue mod 5 (including
    special handling for vertical lines).
    """
    if q % 4 != 1:
        return None
    
    # Enumerate points of PG(2,q)
    # Points: (x, y) for x,y in GF(q) [q^2 points]
    #         (x, 1) for x in GF(q) [q points]  -- "at infinity" in y-direction
    #         (0, 1) [1 point]  -- "at infinity" in x-direction
    # Total: q^2 + q + 1
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y))  # affine points
    for x in range(q):
        points.append((x, q))  # points at infinity in y-direction, encoded as (x, q)
    points.append((-1, -1))  # point at infinity in x-direction
    
    n = q * q + q + 1
    
    # Quadratic residues mod q (for q=5: {1, 4})
    qr = set()
    for i in range(1, q):
        qr.add((i * i) % q)
    
    def slope_mod_q(p1, p2):
        """Compute slope of line through p1, p2 in GF(q), or special values."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Handle points at infinity
        if x1 == -1 and y1 == -1:
            # Point at infinity in x-direction: line through it and (x, y) has slope determined by y
            # Actually, the point (-1,-1) represents the direction of vertical lines
            # Line through (-1,-1) and (x,y): this is the vertical line x = x1 (if p2 is affine)
            # or the line at infinity
            if x2 == q:  # p2 is at infinity in y-direction
                # Both at infinity: the line is the line at infinity
                return -2  # special: not a QR
            else:
                # Vertical line, slope = infinity
                return -1  # special: treat as non-QR
        if x2 == -1 and y2 == -1:
            if x1 == q:
                return -2
            else:
                return -1
        if x1 == q and x2 == q:
            # Both at infinity in y-direction: line at infinity
            return -2
        if x1 == q:
            # p1 at infinity in y, p2 affine: slope = 0 (horizontal)
            return 0
        if x2 == q:
            # p2 at infinity in y, p1 affine: slope = 0
            return 0
        
        # Both affine
        dx = (x2 - x1) % q
        dy = (y2 - y1) % q
        if dx == 0:
            return -1  # vertical
        # slope = dy * dx^{-1} mod q
        # For q=5, inverse: 1->1, 2->3, 3->2, 4->4
        inv = {1: 1, 2: 3, 3: 2, 4: 4}
        m = (dy * inv[dx]) % q
        return m
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            m = slope_mod_q(points[i], points[j])
            if m >= 0 and m in qr:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4(adj, n):
    """Check if graph contains a K4."""
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    # u, v, w form a triangle
                    # Check for 4th vertex
                    common_vw = adj[v].intersection(adj[w])
                    for x in common_vw:
                        if x in adj[u] and x != v and x != w:
                            return True
    return False

def has_independent_set(adj, n, k):
    """Check if graph contains an independent set of size k using bitmask backtracking."""
    if n < k:
        return False
    masks = get_bitmasks(adj, n)
    
    def backtrack(start, current_mask, size):
        if size == k:
            return True
        if start >= n:
            return False
        # Pruning: even taking all remaining can't reach k
        if (n - start) < (k - size):
            return False
        for v in range(start, n):
            if current_mask & (1 << v):
                continue
            if backtrack(v + 1, current_mask | masks[v], size + 1):
                return True
        return False
    
    return backtrack(0, 0, 0)

def get_independent_sets_size_k(adj, n, k):
    """Return list of independent sets of size k using bitmasks for speed."""
    if n < k:
        return []
    
    masks = get_bitmasks(adj, n)
    sets = []
    
    def backtrack(start, current_set, neighbor_mask):
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        remaining = k - len(current_set)
        if n - start < remaining:
            return
        
        for v in range(start, n):
            if not (neighbor_mask & (1 << v)):
                current_set.append(v)
                backtrack(v + 1, current_set, neighbor_mask | masks[v])
                current_set.pop()
                
    backtrack(0, [], 0)
    return sets

def try_add_vertex(adj, n, deadline, masks):
    """
    Tries to find a valid set of neighbors N for a new vertex v_new.
    Returns the set N if found, else None.
    """
    # Step 1: Find all independent sets of size 5 in G
    ind_sets_5 = get_independent_sets_size_k(adj, n, 5)
    
    if not ind_sets_5:
        # No independent sets of size 5, so constraint 2 is always satisfied.
        # We just need a triangle-free set N.
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            # Check if adding v to N creates a triangle
            neigh_in_N = [u for u in N if u in adj[v]]
            triangle_formed = False
            for i in range(len(neigh_in_N)):
                for j in range(i+1, len(neigh_in_N)):
                    if neigh_in_N[j] in adj[neigh_in_N[i]]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N

    # Step 2: Backtracking search for N
    active_sets = [set(s) for s in ind_sets_5]
    
    def backtrack(N, active_sets_copy):
        if not active_sets_copy:
            return N.copy()
        
        if time.time() > deadline:
            return None
            
        # Heuristic: Pick the active set with the smallest number of candidates
        # that can be added to N without forming a triangle.
        
        best_set_idx = -1
        best_candidates = None
        min_candidates = float('inf')
        
        indices = list(range(len(active_sets_copy)))
        random.shuffle(indices)
        
        for idx in indices:
            target_set = active_sets_copy[idx]
            candidates = []
            for v in target_set:
                # Check if v can be added to N
                valid = True
                neigh_v_in_N = [u for u in N if u in adj[v]]
                for i in range(len(neigh_v_in_N)):
                    for j in range(i+1, len(neigh_v_in_N)):
                        if neigh_v_in_N[j] in adj[neigh_v_in_N[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(v)
            
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_set_idx = idx
                best_candidates = candidates
            
            if min_candidates == 0:
                break
        
        if min_candidates == 0:
            return None
            
        random.shuffle(best_candidates)
        
        for v in best_candidates:
            new_N = N | {v}
            
            # Update active sets: remove any set that contains v
            new_active_sets = []
            for s in active_sets_copy:
                if v not in s:
                    new_active_sets.append(s)
            
            result = backtrack(new_N, new_active_sets)
            if result is not None:
                return result
                
        return None

    # Try multiple times with different random shuffles
    for _ in range(50):
        res = backtrack(set(), [set(s) for s in ind_sets_5])
        if res is not None:
            return res
            
    return None

def local_search_polish(adj, n, masks, deadline):
    """
    Try edge swaps (add/remove edges) to enable further vertex additions.
    Returns True if the graph was modified.
    """
    improved = False
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Try random edge swaps
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        is_edge = v in adj[u]
        
        # Tentatively apply the swap
        if is_edge:
            adj[u].remove(v)
            adj[v].remove(u)
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        else:
            adj[u].add(v)
            adj[v].add(u)
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        
        # Check validity
        if has_k4(adj, n) or has_independent_set(adj, n, 6):
            # Revert
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
        else:
            # Valid! Try to add a vertex
            new_N = try_add_vertex(adj, n, deadline, masks)
            if new_N is not None:
                # Add new vertex
                adj.append(set(new_N))
                for u2 in new_N:
                    adj[u2].add(n)
                new_mask = 0
                for u2 in new_N:
                    new_mask |= (1 << u2)
                masks.append(new_mask)
                for u2 in new_N:
                    masks[u2] |= (1 << n)
                n += 1
                improved = True
            else:
                # Revert the edge swap
                if is_edge:
                    adj[u].add(v)
                    adj[v].add(u)
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
    
    return improved, n

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try projective plane construction first (31 vertices)
    g_pp = generate_projective_plane_graph(5)
    best_adj = None
    best_n = 0
    best_masks = None
    
    if g_pp is not None:
        pp_adj = build_adjacency(g_pp['n'], g_pp['edges'])
        pp_n = g_pp['n']
        pp_masks = get_bitmasks(pp_adj, pp_n)
        
        # Check if the projective plane graph is valid (no K4, no independent 6-set)
        if not has_k4(pp_adj, pp_n) and not has_independent_set(pp_adj, pp_n, 6):
            best_adj = pp_adj
            best_n = pp_n
            best_masks = pp_masks
    
    # Also try Paley 17 as fallback
    g17 = generate_paley(17)
    if g17:
        p17_adj = build_adjacency(17, g17['edges'])
        p17_masks = get_bitmasks(p17_adj, 17)
        if not has_k4(p17_adj, 17) and not has_independent_set(p17_adj, 17, 6):
            if 17 > best_n:
                best_adj = p17_adj
                best_n = 17
                best_masks = p17_masks
    
    if best_adj is None:
        return {'n': 1, 'edges': []}
    
    adj = best_adj
    n = best_n
    masks = best_masks
    
    # Main loop: try to add vertices
    while n < 36 and time.time() < deadline:
        added = False
        if time.time() > deadline - 5:
            break
            
        new_N = try_add_vertex(adj, n, deadline, masks)
        
        if new_N is not None:
            # Add new vertex n with neighbors new_N
            # Update adj
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            
            # Update masks
            # New mask for vertex n
            new_mask = 0
            for u in new_N:
                new_mask |= (1 << u)
            masks.append(new_mask)
            
            # Update existing masks to include n
            for u in new_N:
                masks[u] |= (1 << n)
                
            n += 1
            added = True
            
        if not added:
            # Stuck. Try local search polish to enable further additions
            if time.time() < deadline - 10:
                improved, n = local_search_polish(adj, n, masks, deadline)
                if improved:
                    continue
            
            # If still stuck, try more aggressive random restarts with edge perturbations
            success = False
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                    
                # Try a random edge flip to perturb the local structure
                if n >= 2:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        is_edge = v in adj[u]
                        if is_edge:
                            adj[u].remove(v)
                            adj[v].remove(u)
                            masks[u] &= ~(1 << v)
                            masks[v] &= ~(1 << u)
                        else:
                            adj[u].add(v)
                            adj[v].add(u)
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                        
                        # Check if perturbed graph is still valid
                        if has_k4(adj, n) or has_independent_set(adj, n, 6):
                            # Revert
                            if is_edge:
                                adj[u].add(v)
                                adj[v].add(u)
                                masks[u] |= (1 << v)
                                masks[v] |= (1 << u)
                            else:
                                adj[u].remove(v)
                                adj[v].remove(u)
                                masks[u] &= ~(1 << v)
                                masks[v] &= ~(1 << u)
                        else:
                            # Valid perturbation, try to add vertex
                            new_N = try_add_vertex(adj, n, deadline, masks)
                            if new_N is not None:
                                adj.append(set(new_N))
                                for u2 in new_N:
                                    adj[u2].add(n)
                                new_mask = 0
                                for u2 in new_N:
                                    new_mask |= (1 << u2)
                                masks.append(new_mask)
                                for u2 in new_N:
                                    masks[u2] |= (1 << n)
                                n += 1
                                success = True
                                break
                            else:
                                # Revert perturbation
                                if is_edge:
                                    adj[u].add(v)
                                    adj[v].add(u)
                                    masks[u] |= (1 << v)
                                    masks[v] |= (1 << u)
                                else:
                                    adj[u].remove(v)
                                    adj[v].remove(u)
                                    masks[u] &= ~(1 << v)
                                    masks[v] &= ~(1 << u)
            
            if not success:
                break

    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
