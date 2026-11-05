import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(120):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add(adj_sets, n, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    Returns True if C4 would be created.
    """
    neighbors_u = adj_sets[u]
    neighbors_v = adj_sets[v]
    
    # A C4 exists if there is a vertex x adjacent to u (x != v) such that
    # x shares a neighbor with v (other than u).
    # Equivalently: (N(u) \ {v}) intersects N(v) \ {u} in a way that creates a 4-cycle.
    # More precisely: exists x in N(u) with x != v such that (N(x) \ {u}) intersects N(v) \ {u}.
    # But since we're checking before adding (u,v), v is not in N(u) yet.
    # So: exists x in N(u) such that (N(x) \ {u}) intersects N(v).
    
    for x in neighbors_u:
        # x is a neighbor of u. Check if x has a common neighbor with v (other than u).
        # Since v is not yet connected to u, and x is connected to u,
        # we need: exists y in N(x) with y != u and y in N(v).
        common = adj_sets[x] & neighbors_v
        if common:
            # Check if the only common neighbor is u (which would just be a triangle, not C4)
            # Actually if u is in common, that means u in N(x) and u in N(v), but u is not in N(v) yet
            # (we're checking before adding). So u cannot be in neighbors_v.
            # Wait - neighbors_v is adj_sets[v], which doesn't include u yet.
            # So any intersection is a valid C4 witness.
            return True
    return False


def _check_c4_add_fast(adj_matrix, u, v):
    """
    Fast C4 check using numpy bit operations on adjacency matrix.
    adj_matrix: numpy bool array of shape (n, n)
    Returns True if adding edge (u,v) creates a C4.
    """
    n = adj_matrix.shape[0]
    # C4 exists if there is x such that adj[u,x] and adj[x,y] and adj[v,y] for some y
    # i.e., (A[u,:] & A[v,:]) has a vertex x where A[x,:] & A[v,:] has a non-u element
    # Simpler: count common neighbors via matrix multiply approach
    # Number of length-2 paths from u to v through intermediate = sum(A[u,:] * A[:,v])
    # But we need 4-cycles specifically: u-x-v-y-u where x!=y
    # A C4 containing edge (u,v) exists iff there exist x,y with x!=y,
    # x in N(u), y in N(v), x in N(y), i.e., A[x,y]=1 with x in N(u)\{v}, y in N(v)\{u}
    # Count = sum over x in N(u), y in N(v) of A[x,y] minus contributions where x==y or x==v or y==u
    # Since u,v not yet connected, x==v means v in N(u) which is false. y==u means u in N(v) which is false.
    # So count = (A[u,:] @ A[:,v]) - (number of x where A[u,x] and A[v,x])
    # i.e., common neighbors of u and v give triangles, not C4s.
    # C4 count = (A[u,:] @ A[:,v]) - |N(u) ∩ N(v)|
    
    # A[u,:] @ A[:,v] = sum_x A[u,x]*A[x,v] = sum_x A[u,x]*A[v,x] = |N(u) ∩ N(v)|
    # Wait that's just the number of common neighbors. That gives triangles.
    
    # For C4: u-x-w-v-u. x adjacent to u, w adjacent to v, x adjacent to w.
    # Count = sum_{x in N(u)} sum_{w in N(v)} A[x,w]
    # = sum_x A[u,x] * (A[:,v] @ A[:,x])... no
    # = sum_{x,w} A[u,x] * A[v,w] * A[x,w]
    # = (A[u,:] * (A @ A[:,v]))[...] no
    # Let me think differently.
    # Number of C4s through (u,v) = number of pairs (x,w) with x in N(u), w in N(v), x!=w, A[x,w]=1
    # = sum_{x in N(u), w in N(v), x!=w} A[x,w]
    # = sum_{x in N(u), w in N(v)} A[x,w] - sum_{x in N(u) ∩ N(v)} A[x,x] (but A[x,x]=0)
    # Actually if x==w, then x in N(u) and x in N(v), and A[x,x]=0, so those contribute 0.
    # So C4_count = sum_{x in N(u), w in N(v)} A[x,w]
    # = (A[u,:] * 1) ... let's use matrix form
    # = A[u,:] @ A[:,v] ... no that's sum_x A[u,x]*A[x,v] = common neighbors
    # 
    # Let me use: for each x in N(u), count neighbors of x that are in N(v).
    # = sum_{x} A[u,x] * (A[x,:] & N(v)).sum()
    # In numpy: (A[u,:] * (A @ A[:,v]))  -- no
    # (A @ A[:,v])[x] = sum_w A[x,w]*A[v,w] = number of common neighbors of x and v
    # So C4_count = sum_{x in N(u)} (A @ A[:,v])[x] = A[u,:] @ (A @ A[:,v])
    
    common_with_v = adj_matrix @ adj_matrix[:, v]  # for each x, # common neighbors of x and v
    c4_count = adj_matrix[u, :] @ common_with_v
    
    # But this counts x=v too if v were in N(u), which it isn't.
    # It also counts the case where the "C4" is actually a triangle u-x-v where x is common neighbor.
    # If x is a common neighbor of u and v, then A[u,x]=1, A[v,x]=1, and (A@A[:,v])[x] includes A[x,x]=0
    # plus other common neighbors. Hmm, this gets complicated.
    
    # Let me just use the set-based approach which is correct and fast enough for n<=47.
    return c4_count > 0


def _polarity_seed(n, rng):
    """
    Build a C4-free graph using a polarity-based construction.
    This gives a much better starting point than empty graph + random greedy.
    """
    adj_sets = [set() for _ in range(n)]
    
    # Partition vertices into two groups
    mid = n // 2
    group_a = list(range(mid))
    group_b = list(range(mid, n))
    rng.shuffle(group_a)
    rng.shuffle(group_b)
    
    # Strategy: create a bipartite graph between A and B with controlled density,
    # then add intra-group edges carefully.
    
    target_deg_b = max(1, (n - 39) // 2)
    
    for a in group_a:
        b_candidates = list(group_b)
        rng.shuffle(b_candidates)
        for b in b_candidates[:target_deg_b]:
            adj_sets[a].add(b)
            adj_sets[b].add(a)
    
    # Add intra-A edges
    a_remaining = list(group_a)
    rng.shuffle(a_remaining)
    for i in range(0, len(a_remaining) - 1, 2):
        u, v = a_remaining[i], a_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    # Add intra-B edges
    b_remaining = list(group_b)
    rng.shuffle(b_remaining)
    for i in range(0, len(b_remaining) - 1, 2):
        u, v = b_remaining[i], b_remaining[i+1]
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    
    return adj_sets


def _greedy_fill(adj_sets, n, deadline, rng, prioritize_low_degree=True):
    """
    Greedily add edges to boost degrees, prioritizing low-degree vertices.
    """
    # Build a priority queue of edges weighted by how much they help low-degree vertices
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Sort edges by potential benefit: edges connecting low-degree vertices first
    degrees = [len(s) for s in adj_sets]
    
    # Multiple passes: in each pass, try to add edges that help the most deficient vertices
    for pass_num in range(10):
        if time.time() > deadline:
            break
        
        degrees = [len(s) for s in adj_sets]
        
        # Collect candidate edges that could be added
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if u not in adj_sets[v]:
                    # Score: sum of deficiency of u and v
                    score = (n - 1 - degrees[u]) + (n - 1 - degrees[v])
                    candidates.append((score, u, v))
        
        # Sort by score descending, then randomize ties
        candidates.sort(key=lambda x: (-x[0], rng.random()))
        
        added_any = False
        for score, u, v in candidates:
            if time.time() > deadline:
                break
            if u in adj_sets[v]:
                continue
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added_any = True
                degrees[u] += 1
                degrees[v] += 1
        
        if not added_any:
            break


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Use polarity seed for better initialization
    adj_sets = _polarity_seed(n, rng)
    
    # Phase 1: Greedy fill prioritizing low-degree vertices
    _greedy_fill(adj_sets, n, deadline, rng)
    
    # Check if we're done
    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    # Phase 2: Local search to improve minimum degree
    max_iters = 5000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate degrees
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find a vertex with minimum degree (randomly among them)
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge from v to a non-neighbor, prioritizing those that help other low-degree vertices
        candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
        
        # Score candidates by how much they help
        scored_candidates = []
        for u in candidates:
            score = degrees[u]  # lower degree means more help
            scored_candidates.append((score, rng.random(), u))
        scored_candidates.sort()
        
        added = False
        for score, rand, u in scored_candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break
        
        if not added:
            # Cannot add edge to v directly.
            # Try a swap: Remove an edge that doesn't help much, then try to add edges to v.
            
            all_current_edges = []
            for u in range(n):
                for w in adj_sets[u]:
                    if u < w:
                        all_current_edges.append((u, w))
                        
            if not all_current_edges:
                break
            
            # Prioritize removing edges between high-degree vertices (less impact on min degree)
            # Score each edge by the sum of degrees of its endpoints (higher = safer to remove)
            scored_edges = []
            for a, b in all_current_edges:
                edge_score = degrees[a] + degrees[b]
                scored_edges.append((edge_score, rng.random(), a, b))
            
            # Try removing edges from highest score first (least harmful)
            scored_edges.sort(key=lambda x: (-x[0], x[1]))
            
            found_swap = False
            for edge_score, rand, a, b in scored_edges[:20]:  # Try up to 20 removals
                if time.time() > deadline:
                    break
                
                # Remove edge (a, b)
                adj_sets[a].remove(b)
                adj_sets[b].remove(a)
                degrees[a] -= 1
                degrees[b] -= 1
                
                # Try to add edges to v
                added_edges = []
                candidates_v = [u for u in range(n) if u != v and u not in adj_sets[v]]
                
                # Score by how much they help
                scored_cand_v = []
                for u in candidates_v:
                    scored_cand_v.append((degrees[u], rng.random(), u))
                scored_cand_v.sort()
                
                for score_v, rand_v, u in scored_cand_v:
                    if time.time() > deadline:
                        break
                    if not _check_c4_add(adj_sets, n, u, v):
                        adj_sets[u].add(v)
                        adj_sets[v].add(u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added_edges.append(u)
                        if len(added_edges) >= 3:
                            break
                
                new_min = min(degrees)
                
                if new_min >= min_deg:
                    # Accept
                    found_swap = True
                    break
                else:
                    # Revert
                    adj_sets[a].add(b)
                    adj_sets[b].add(a)
                    degrees[a] += 1
                    degrees[b] += 1
                    for u in added_edges:
                        adj_sets[u].remove(v)
                        adj_sets[v].remove(u)
                        degrees[u] -= 1
                        degrees[v] -= 1
            
            if not found_swap:
                # Try a different vertex next iteration
                continue
    
    # Final check
    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END
