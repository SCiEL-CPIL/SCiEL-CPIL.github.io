import time
import random
from itertools import combinations

_START_TIME = None

def _time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check K4-free using bitmasks. adj is list of int bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            common &= ~((1 << (b + 1)) - 1)  # only vertices > b
            common &= ~(1 << a)
            # For each c in common, check if adj[a]&adj[b]&adj[c] has any vertex > c
            c = common
            while c:
                lb = c & (-c)
                ci = lb.bit_length() - 1
                c ^= lb
                triple_common = adj[a] & adj[b] & adj[ci]
                triple_common &= ~((1 << (ci + 1)) - 1)  # only vertices > ci
                triple_common &= ~(1 << a)
                triple_common &= ~(1 << b)
                if triple_common:
                    return False
    return True

def _max_tf_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph via branch and bound with bitmasks."""
    best = [0]
    all_mask = (1 << n) - 1
    
    # Precompute: for each vertex, its neighbors as bitmask
    # A set S is triangle-free iff for every v in S, no two neighbors of v in S are adjacent to each other
    # Equivalently: for every triangle (a,b,c), S doesn't contain all three
    
    # Branch and bound: build set incrementally
    # State: (available_mask, current_mask, current_size)
    # available_mask: vertices that can still be added
    # When adding vertex v to current set C, v must not form a triangle with any pair in C
    # i.e., for all u, w in C: not (adj[v]&adj[u] has w) and (adj[v]&adj[w] has u)
    
    # More efficient: maintain "forbidden" set
    # When we have current set C, a vertex v can be added iff
    # for all u in C: (adj[v] & adj[u] & C) == 0  ... no wait
    # v forms triangle with u, w in C iff adj[v][u] and adj[v][w] and adj[u][w]
    # So v is addable iff: for all u in C with adj[v][u]=1, 
    #   adj[v] & adj[u] & C has no other vertex w with adj[u][w]=1
    # Simpler: v is addable iff (adj[v] & C) has no edge within C
    # i.e., for the set N_v = adj[v] & C, the induced subgraph on N_v has no edges
    # which means: for all u in N_v, adj[u] & N_v == 0
    
    def can_add(v, current_mask):
        """Check if v can be added to the set represented by current_mask."""
        neighbors_in_current = adj[v] & current_mask
        # Check if any two vertices in neighbors_in_current are adjacent
        nm = neighbors_in_current
        while nm:
            lb = nm & (-nm)
            u = lb.bit_length() - 1
            nm ^= lb
            if adj[u] & neighbors_in_current:
                return False
        return True
    
    def dfs(available, current_mask, size):
        if time.time() - start_time > deadline:
            return
        if size > best[0]:
            best[0] = size
        if not available:
            return
        # Upper bound
        if size + bin(available).count('1') <= best[0]:
            return
        
        # Try adding each available vertex
        avail = available
        while avail:
            lb = avail & (-avail)
            v = lb.bit_length() - 1
            avail ^= lb
            
            if can_add(v, current_mask):
                new_current = current_mask | lb
                # New available: remove v, and remove any vertex that would form triangle with v and some vertex in new_current
                new_avail = available & ~lb
                # For each u in new_avail, check if u can still be added
                # Actually, we need to filter: u can be added to new_current only if can_add(u, new_current)
                # But that's expensive. Instead, just pass new_avail = available & ~lb
                # and let can_add handle it in the next recursion
                dfs(new_avail, new_current, size + 1)
            
            # Also try NOT adding v (implicitly handled by continuing the loop)
    
    dfs(all_mask, 0, 0)
    return best[0]

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110  # Safety margin for 120s limit
    n = 18
    random.seed(42)
    
    best_edges = None
    best_score = 999
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def try_graph(adj):
        """Evaluate a K4-free graph, return its score."""
        if not _is_k4_free(adj, n):
            return 999
        return _max_tf_induced(adj, n, deadline, _START_TIME)
    
    # Start with K6,6,6 (Turan graph T_3(18))
    # This graph has the maximum number of edges among K4-free graphs.
    # Maximizing edges minimizes the largest triangle-free induced subgraph.
    adj = [0] * n
    for i in range(n):
        part_i = i // 6
        for j in range(n):
            if i != j and j // 6 != part_i:
                adj[i] |= (1 << j)
    
    score = try_graph(adj)
    best_score = score
    best_edges = adj_to_edges(adj)
    
    # Limited local search to verify no better solution exists nearby
    # Since K6,6,6 is the unique max-edge K4-free graph, we can only remove edges.
    # Removing edges increases the score, so this is unlikely to improve.
    # We do a very short search just in case.
    
    def flip_edge(adj, u, v):
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
    
    time_budget = _START_TIME + 5  # Only search for 5 seconds
    attempts = 0
    
    while time.time() < time_budget:
        attempts += 1
        if attempts % 10 == 0:
            # Check time frequently
            pass
            
        # Try removing a random edge
        u, v = random.sample(range(n), 2)
        if (adj[u] >> v) & 1:
            flip_edge(adj, u, v)
            new_score = try_graph(adj)
            if new_score < best_score:
                best_score = new_score
                best_edges = adj_to_edges(adj)
            else:
                flip_edge(adj, u, v)  # revert
    
    return best_edges
