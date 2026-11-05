import time
import random
import math

N = 18
DEADLINE = 115.0  # Safety margin before 120s

def _build_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj):
    # K4-free iff for every vertex u, the subgraph induced on N(u) is triangle-free
    for u in range(N):
        nu = adj[u]
        # For each pair of neighbors i,j of u, check if i-j is an edge
        # More efficiently: for each neighbor i of u, check if adj[i] & nu has any edge
        v = nu
        while v:
            lb = v & (-v)
            i = lb.bit_length() - 1
            v -= lb
            ni_in_nu = adj[i] & nu
            # Check if ni_in_nu is independent (no edges within it)
            w = ni_in_nu
            while w:
                lb2 = w & (-w)
                j = lb2.bit_length() - 1
                w -= lb2
                if adj[j] & ni_in_nu:
                    return False
    return True

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps the graph K4-free."""
    if adj[u] & (1 << v):
        return True  # Edge already exists
    # Adding edge (u,v) creates a K4 if there exist x,y such that:
    # u-x, u-y, v-x, v-y, x-y are all edges
    # Equivalently, in the common neighbors of u and v, there is an edge.
    common = adj[u] & adj[v]
    # Check if common contains an edge
    w = common
    while w:
        lb = w & (-w)
        x = lb.bit_length() - 1
        w -= lb
        if adj[x] & common:
            return False
    return True

def _max_triangle_free(adj):
    """Exact branch and bound for largest triangle-free induced subgraph."""
    full = (1 << N) - 1
    best = [0]
    
    def is_compatible(v, mask):
        """Check if v can be added to mask without creating a triangle."""
        nv_in_mask = adj[v] & mask
        # Check if nv_in_mask is independent
        nn = nv_in_mask
        while nn:
            lb = nn & (-nn)
            u = lb.bit_length() - 1
            nn -= lb
            if adj[u] & nv_in_mask:
                return False
        return True

    def bb(current_mask, avail_mask):
        cur_size = current_mask.bit_count()
        rem_size = avail_mask.bit_count()
        
        # Upper bound
        if cur_size + rem_size <= best[0]:
            return
            
        if avail_mask == 0:
            if cur_size > best[0]:
                best[0] = cur_size
            return
            
        # Pick vertex with highest degree in avail for better pruning
        best_v = -1
        best_deg = -1
        a = avail_mask
        while a:
            lb = a & (-a)
            v = lb.bit_length() - 1
            a -= lb
            deg = (adj[v] & avail_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        lb = 1 << v
        new_avail = avail_mask - lb
        
        # Option 1: Include v (try first if compatible)
        if is_compatible(v, current_mask):
            bb(current_mask | lb, new_avail)
        
        # Option 2: Exclude v
        bb(current_mask, new_avail)

    bb(0, full)
    return best[0]

def _edges_from_adj(adj):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges

def _make_c9_baseline():
    """Create two disjoint C9 cycles on vertices 0-17.
    Cycle 1: 0-1-2-3-4-5-6-7-8-0
    Cycle 2: 9-10-11-12-13-14-15-16-17-9
    This is K4-free and has alpha = 8 (alpha(C9)=4, so alpha(2*C9)=8).
    """
    adj = [0] * N
    # First C9: vertices 0-8
    for i in range(9):
        u = i
        v = (i + 1) % 9
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Second C9: vertices 9-17
    for i in range(9):
        u = 9 + i
        v = 9 + (i + 1) % 9
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial candidate: two disjoint C9 cycles (score = 8)
    current_adj = _make_c9_baseline()
    current_score = _max_triangle_free(current_adj)
    
    best_adj = current_adj[:]
    best_score = current_score
    
    # Local search: try to add edges to reduce the triangle-free number
    # We want to find a K4-free graph where the largest triangle-free induced subgraph is as small as possible.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor: try to add a random edge
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
            
        if current_adj[u] & (1 << v):
            continue  # Edge already exists, skip
        
        new_adj = current_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        # Check if adding this edge keeps K4-free
        if not _can_add_edge(current_adj, u, v):
            continue
            
        new_score = _max_triangle_free(new_adj)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_adj = new_adj
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
    
    return _edges_from_adj(best_adj)
