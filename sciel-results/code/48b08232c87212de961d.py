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
        for attempt in range(80):
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


def _check_c4_add(adj_bits, u, v):
    """
    Check if adding edge (u,v) creates a C4 using numpy bitsets.
    adj_bits: numpy array of shape (n,) where each element is a bitset (int).
    Returns True if C4 would be created.
    """
    # A C4 exists if there's a vertex x adjacent to both u and v,
    # and x has another common neighbor with v (or u).
    # More precisely: adding (u,v) creates C4 iff |N(u) ∩ N(v)| >= 1
    # where N includes the new edge. But since we're checking BEFORE adding,
    # we need: exists x in N(u) such that x != v and |N(x) ∩ N(v)| >= 1
    # where N(x) doesn't include the new edge (u,v) yet.
    
    # Actually, adding edge (u,v) creates a C4 iff there exists x such that
    # x is adjacent to u AND x is adjacent to v AND x != u,v.
    # Wait, that's just a triangle, not C4.
    
    # C4: u-a-v-b-u. So adding (u,v) creates C4 if there exist a,b 
    # such that a is adjacent to u and v, and b is adjacent to u and v,
    # with a != b. OR: a adjacent to u, b adjacent to v, a adjacent to b,
    # and the path u-a-b-v forms a C4 with the new edge u-v.
    
    # Correct check: Adding (u,v) creates C4 iff there exist distinct a,b
    # such that a ∈ N(u)∩N(v) OR (a ∈ N(u)\{v}, b ∈ N(v)\{u}, a∈N(b), a≠b).
    # Actually simpler: C4 through new edge (u,v) means u-a-?-v where ? is 
    # another common neighbor. So: |N(u) ∩ N(v)| >= 1 before adding the edge.
    # No wait - if u and v already have a common neighbor, adding (u,v) 
    # creates a triangle, not necessarily C4.
    
    # Let me reconsider. A C4 is a cycle of length 4: u-x-y-v-u.
    # If we add edge (u,v), we create a C4 iff there exist x,y with:
    #   u-x edge exists, x-y edge exists, y-v edge exists, and x≠v, y≠u, x≠y.
    # Equivalently: there exists x ∈ N(u) (x≠v) and y ∈ N(v) (y≠u) 
    # such that x-y edge exists and x≠y.
    
    # Using bitsets:
    # N(u) without v: adj_bits[u] with bit v cleared
    # For each x in N(u)\{v}, check if N(x) ∩ (N(v)\{u}) is non-empty.
    
    nu = adj_bits[u] & ~(1 << v)  # N(u) excluding v
    nv = adj_bits[v] & ~(1 << u)  # N(v) excluding u
    
    # For any x in nu, if N(x) intersects nv, we have a C4
    # We can compute: union of N(x) for x in nu, then intersect with nv
    # But that's expensive. Instead:
    # C4 exists iff ∃x ∈ nu such that (adj_bits[x] & nv) != 0
    
    # Iterate over set bits in nu
    x = nu
    while x:
        # Get lowest set bit
        low = x & (-x)
        idx = low.bit_length() - 1
        if adj_bits[idx] & nv:
            return True
        x &= x - 1
    
    return False


def _init_structured(n, rng):
    """
    Build a C4-free graph using a structured approach based on 
    polarity graphs or similar constructions.
    Returns adj_bits: numpy array of shape (n,) with bitsets.
    """
    adj_bits = np.zeros(n, dtype=np.uint64)
    
    # Strategy: Use a "blow-up" of a small C4-free graph
    # For n up to 47, we can use a base graph and expand
    
    # Alternative: Random bipartite-like construction with C4 avoidance
    # Split into two parts, connect with controlled density
    
    mid = n // 2
    part_a = list(range(mid))
    part_b = list(range(mid, n))
    rng.shuffle(part_a)
    rng.shuffle(part_b)
    
    # Connect each vertex in A to a few vertices in B
    # To avoid C4 in bipartite part, ensure no two vertices in A share 2+ neighbors in B
    # This is equivalent to: the incidence matrix has no 2x2 all-ones submatrix
    
    # Use a greedy approach for bipartite connections
    b_neighbors = {b: set() for b in part_b}  # Track which A vertices connect to each B
    
    for a in part_a:
        # Find B vertices that don't create C4
        # A B vertex b is safe for a if |{a' in A : a' connected to b}| <= 0 
        # (i.e., no other A vertex is connected to b), OR we accept some risk
        # Actually for C4-free bipartite: no two A vertices can share a B neighbor
        # Wait, that's too restrictive. C4 in bipartite: a1-b1-a2-b2-a1
        # So a1 and a2 both connect to b1 AND both connect to b2.
        # So: for each pair of A vertices, they can share at most 1 B neighbor.
        
        # Greedy: try to connect a to random B vertices, checking C4
        b_candidates = list(part_b)
        rng.shuffle(b_candidates)
        
        for b in b_candidates:
            # Check if adding (a,b) creates C4
            # C4 would be: a - b - a' - b' - a where a' is another A vertex
            # connected to both b and some b' that a is also connected to.
            # Simpler: just use the general C4 check
            if not _check_c4_add(adj_bits, a, b):
                adj_bits[a] |= (1 << b)
                adj_bits[b] |= (1 << a)
                # Limit degree to keep things manageable
                if (adj_bits[a] & ((1 << n) - 1)).bit_count() >= max(2, (n - 39) // 3):
                    break
    
    # Add intra-part edges to boost degrees
    for part in [part_a, part_b]:
        shuffled = list(part)
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            for j in range(i + 1, len(shuffled)):
                u, v = shuffled[i], shuffled[j]
                if not _check_c4_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
    
    return adj_bits


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Use structured initialization
    adj_bits = _init_structured(n, rng)
    
    # Phase 1: Greedy fill - try to add remaining edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if not (adj_bits[u] >> v) & 1:
            if not _check_c4_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
    # Phase 2: Local search to improve minimum degree
    max_iters = 3000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate degrees
        degrees = [(adj_bits[i] & ((1 << n) - 1)).bit_count() for i in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find vertices with minimum degree
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge from v to a non-neighbor
        non_neighbors = [u for u in range(n) if u != v and not ((adj_bits[v] >> u) & 1)]
        rng.shuffle(non_neighbors)
        
        added = False
        for u in non_neighbors:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added = True
                break
        
        if not added:
            # Cannot add edge to v directly.
            # Try a swap: Remove a random edge (a, b), then try to add edges to v.
            
            all_current_edges = []
            for u in range(n):
                for w in range(u + 1, n):
                    if (adj_bits[u] >> w) & 1:
                        all_current_edges.append((u, w))
                        
            if not all_current_edges:
                break
                
            a, b = rng.choice(all_current_edges)
            
            # Remove edge (a, b)
            adj_bits[a] &= ~(1 << b)
            adj_bits[b] &= ~(1 << a)
            
            # Try to add edges to v
            added_edges = []
            non_neighbors = [u for u in range(n) if u != v and not ((adj_bits[v] >> u) & 1)]
            rng.shuffle(non_neighbors)
            
            for u in non_neighbors:
                if time.time() > deadline:
                    break
                if not _check_c4_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    added_edges.append(u)
                    if len(added_edges) >= 3:
                        break
                        
            new_degrees = [(adj_bits[i] & ((1 << n) - 1)).bit_count() for i in range(n)]
            new_min = min(new_degrees)
            
            if new_min >= min_deg:
                # Accept
                pass
            else:
                # Revert
                adj_bits[a] |= (1 << b)
                adj_bits[b] |= (1 << a)
                for u in added_edges:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                
                continue

    # Final check
    degrees = [(adj_bits[i] & ((1 << n) - 1)).bit_count() for i in range(n)]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END
