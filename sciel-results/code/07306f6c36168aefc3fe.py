"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_adj(edges):
    """Build adjacency bitmasks from edge list."""
    adj = [0] * 18
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _check_k4_free(adj):
    """Check if graph is K4-free using bitmask adjacency."""
    n = 18
    # For each triple, check if they form a triangle, and if any vertex in the triangle
    # is also connected to a 4th vertex that forms K4
    # Actually, K4-free means no 4 vertices are mutually connected.
    # Check all 4-subsets: too expensive. Instead, for each triangle, check if there's a 4th vertex connected to all three.
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] >> v) & 1:
                continue
            for w in range(v+1, n):
                if not ((adj[u] >> w) & 1) and not ((adj[v] >> w) & 1):
                    continue
                # u, v, w form a triangle
                # Check if any other vertex x is connected to all three
                common = adj[u] & adj[v] & adj[w]
                # Remove u, v, w themselves
                common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                if common != 0:
                    return False
    return True

def _max_triangle_free(adj, deadline=None):
    """Find the largest triangle-free induced subgraph using branch and bound."""
    n = 18
    best = [0]
    
    # Precompute: for each vertex, its non-neighbors (within 0..n-1)
    # A set S is triangle-free if for every pair (u,v) in S, there's no w in S adjacent to both
    # Equivalently, S is an independent set in the "triangle graph" where edge (u,v) means there exists w adjacent to both u and v? No.
    # Actually, S is triangle-free iff for every pair u,v in S, the common neighbors of u and v within S is empty.
    
    # Branch and bound: build S incrementally.
    # State: current set S (as bitmask), candidate vertices that can be added (must not create a triangle with existing pairs in S).
    # A vertex v can be added to S if for every pair (u, w) in S, v is not adjacent to both u and w.
    # Equivalently, v must not be in the common neighborhood of any pair in S.
    
    # Let's use a recursive approach with bitmasks.
    # candidates: bitmask of vertices that can still be added
    # S: bitmask of current selected vertices
    
    def _can_add(v, S_mask, adj):
        """Check if vertex v can be added to S without creating a triangle."""
        # For every pair (u, w) in S, v must not be adjacent to both u and w
        # Equivalently, (adj[u] & adj[v]) & (adj[w] & adj[v]) & S_mask should have at most... 
        # Actually: if there exist u, w in S such that v is adjacent to both u and w, then adding v creates triangle (u, w, v).
        # So: for all u in S, (adj[u] & adj[v]) & S_mask should be 0? No, that's too strong.
        # We need: there is no pair u, w in S such that v is adjacent to both.
        # That is: for all u in S, the set of vertices in S adjacent to v and also adjacent to u must be empty.
        # Equivalently: let N_v_in_S = adj[v] & S_mask. Then for each u in N_v_in_S, adj[u] & S_mask & adj[v] should be 0? 
        # Wait: if u is in S and adjacent to v, and w is in S and adjacent to v and w is adjacent to u, then triangle.
        # So: for each u in (adj[v] & S_mask), we need (adj[u] & adj[v] & S_mask) == 0, except u itself.
        # Actually: (adj[u] & adj[v] & S_mask) includes u? No, u is not adjacent to itself.
        # So: for each u in (adj[v] & S_mask), if (adj[u] & adj[v] & S_mask) != 0, then there's a triangle.
        # But this is O(|S|) per check. Let's precompute.
        
        NvS = adj[v] & S_mask
        if NvS == 0:
            return True
        # For each u in NvS, check if adj[u] & adj[v] & S_mask has any bit set (other than possibly none)
        # Actually we just need: does there exist u, w in S such that u != w, v adj u, v adj w, u adj w?
        # That's: is the subgraph induced by NvS (within S) non-empty? i.e., is there an edge in S within NvS?
        # So: for each u in NvS, check if adj[u] & NvS != 0
        temp = NvS
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            if adj[u] & NvS:
                return False
            temp ^= lsb
        return True

    def _dfs(S_mask, candidates, depth):
        if deadline and time.time() > deadline:
            return
        if candidates == 0:
            if depth > best[0]:
                best[0] = depth
            return
        # Bound: even if we add all candidates, can we beat best?
        if depth + candidates.bit_count() <= best[0]:
            return
        # Try adding a vertex
        # Choose a vertex from candidates
        v = (candidates & (-candidates)).bit_length() - 1
        # Option 1: don't include v
        _dfs(S_mask, candidates & ~(1 << v), depth)
        # Option 2: include v (if possible)
        if _can_add(v, S_mask, adj):
            new_candidates = candidates & ~(1 << v)
            # Filter new_candidates: remove vertices that would create a triangle with v and any vertex already in S or with v
            # Actually, we need to filter new_candidates to only those that can be added after adding v.
            # A vertex w in new_candidates can be added if _can_add(w, S_mask | (1<<v), adj)
            # But that's expensive. Let's just recurse and let _can_add handle it.
            _dfs(S_mask | (1 << v), new_candidates, depth + 1)

    all_mask = (1 << 18) - 1
    _dfs(0, all_mask, 0)
    return best[0]

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with a known good graph: complete tripartite K6,6,6
    # This has max triangle-free induced subgraph of size 12 (two parts)
    # We want to minimize this.
    
    # Strategy: local search. Start with a random K4-free graph, then try edge flips to reduce max triangle-free size.
    
    # First, let's generate a random K4-free graph.
    # A simple approach: start with empty graph, add edges randomly, check K4-free.
    # Or: use a random 3-coloring and make it complete tripartite? That gives 12.
    
    # Better: start with the tripartite graph and try to modify it.
    # Or: use a different construction.
    
    # Let's try: random graph with controlled density, then repair to be K4-free.
    
    # Actually, let's use a heuristic: 
    # 1. Generate a random graph
    # 2. If not K4-free, remove edges to make it K4-free
    # 3. Evaluate max triangle-free size
    # 4. Local search: flip edges (add/remove) if it improves (reduces) the score while staying K4-free
    
    random.seed(42)
    
    def _random_k4_free():
        # Start with empty graph, add edges one by one
        adj = [0] * 18
        edges = []
        for u in range(18):
            for v in range(u+1, 18):
                if random.random() < 0.5:
                    # Check if adding (u,v) creates a K4
                    # Check all pairs (w,x) where w < x, w != u,v, x != u,v
                    # Actually, check if there exist w, x such that u,v,w,x form K4
                    # That means: w and x are both adjacent to u and v, and w is adjacent to x
                    common = adj[u] & adj[v]
                    common &= ~(1 << u) & ~(1 << v)
                    # Check if common contains an edge
                    temp = common
                    has_edge = False
                    while temp:
                        lsb = temp & (-temp)
                        w = lsb.bit_length() - 1
                        if adj[w] & common:
                            has_edge = True
                            break
                        temp ^= lsb
                    if not has_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
        return adj, edges
    
    def _evaluate(adj):
        return _max_triangle_free(adj)
    
    # Initial solution: tripartite
    adj = [0] * 18
    edges = []
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
    
    current_score = _evaluate(adj)
    best_adj = adj[:]
    best_edges = edges[:]
    best_score = current_score
    
    # Local search
    n = 18
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1
        improved = False
        
        # Try each edge flip
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if time.time() > deadline - 10:
                break
            has_edge = (adj[u] >> v) & 1
            
            # Try flipping
            if has_edge:
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add edge - check K4-free
                common = adj[u] & adj[v]
                common &= ~(1 << u) & ~(1 << v)
                temp = common
                has_edge_in_common = False
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    if adj[w] & common:
                        has_edge_in_common = True
                        break
                    temp ^= lsb
                if has_edge_in_common:
                    # Can't add this edge
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_score = _evaluate(adj)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                # Rebuild edges from adj
                best_edges = []
                for a in range(n):
                    for b in range(a+1, n):
                        if (adj[a] >> b) & 1:
                            best_edges.append([a, b])
                improved = True
            else:
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                break
        
        if not improved:
            # Try a random restart
            if time.time() < deadline - 20:
                adj, edges = _random_k4_free()
                current_score = _evaluate(adj)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_edges = edges[:]
    
    return best_edges

# EVOLVE-BLOCK-END
