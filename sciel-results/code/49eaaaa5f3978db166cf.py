import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses bitset-based C6 checking, greedy initialization, and simulated annealing.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    # Bitset-based adjacency: adj[i] is a 64-bit integer with bit j set if i~j
    adj = [0] * N
    edges = set()
    
    def add_edge(u, v):
        if v in adj[u]:  # Check via bit
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.add((min(u,v), max(u,v)))
        return True
    
    def remove_edge(u, v):
        if not (adj[u] >> v) & 1:
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.discard((min(u,v), max(u,v)))
        return True
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    def count_edges():
        return len(edges)
    
    def get_neighbors(u):
        """Return list of neighbors of u"""
        nb = []
        bits = adj[u]
        while bits:
            lsb = bits & (-bits)
            v = lsb.bit_length() - 1
            nb.append(v)
            bits ^= lsb
        return nb
    
    def is_c6_free(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 through (u,v) is a path u-a-b-c-d-v of length 5.
        We check: exists a in N(u), b in N(a)\{u}, c in N(b)\{a}, d in N(c)\{b} such that d in N(v).
        Equivalent: exists a in N(u) ∩ N(v) is NOT what we want (that's a triangle).
        We want: N(u) -> N(N(u)) -> N(N(N(u))) -> N(N(N(N(u)))) intersects N(v),
        with all intermediate nodes distinct and not u or v.
        
        More precisely: path u-a-b-c-d-v where a,b,c,d are distinct from u,v and each other.
        """
        # Get neighbors of u and v
        nbs_u = get_neighbors(u)
        nbs_v = get_neighbors(v)
        
        # For each a in N(u), for each b in N(a) \ {u}, for each c in N(b) \ {a},
        # check if any d in N(c) \ {b} is in N(v) and d != u, d != v, d != a, d != b
        # This is O(deg(u) * deg(a) * deg(b) * deg(c)) which can be large.
        
        # Better: precompute 2-step, 3-step, 4-step neighborhoods with path tracking
        # But for speed, let's do a smarter check.
        
        # A C6 through (u,v) exists iff there is a path of length 4 from u to v
        # that doesn't use u or v internally.
        
        # Use BFS-like approach with depth tracking
        # State: (node, depth, path_mask) where path_mask tracks visited nodes
        # But 2^40 is too large. Instead, track the path as a small set.
        
        # Since we only need paths of length 4 (u-a-b-c-d-v), we can do:
        # For each a in N(u):
        #   For each b in N(a) \ {u, v}:
        #     For each c in N(b) \ {a, u, v}:
        #       If N(c) ∩ N(v) \ {b, u, v} is non-empty, C6 exists
        
        for a in nbs_u:
            if a == v:
                continue
            nbs_a = get_neighbors(a)
            for b in nbs_a:
                if b == u or b == v or b == a:
                    continue
                nbs_b = get_neighbors(b)
                for c in nbs_b:
                    if c == u or c == v or c == a or c == b:
                        continue
                    nbs_c = get_neighbors(c)
                    for d in nbs_c:
                        if d == u or d == v or d == a or d == b or d == c:
                            continue
                        if has_edge(d, v):
                            return False  # C6 found
        
        return True  # No C6 found
    
    # Initial graph: Start with a denser structure
    # Use a random greedy initialization
    random.seed(42)
    
    # Start with empty graph and greedily add edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if has_edge(u, v):
            continue
        if is_c6_free(u, v):
            add_edge(u, v)
    
    best_edges = count_edges()
    best_adj = adj[:]
    
    # Simulated Annealing
    temp = 50.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to toggle
        if random.random() < 0.5:
            # Try to add a random non-edge
            u, v = random.choice(all_pairs)
            if has_edge(u, v):
                continue
            if is_c6_free(u, v):
                add_edge(u, v)
                if count_edges() > best_edges:
                    best_edges = count_edges()
                    best_adj = adj[:]
            else:
                # C6 would be formed; with some probability, remove a random edge from the blocking C6
                # For simplicity, just skip for now
                pass
        else:
            # Try to remove a random edge
            if not edges:
                continue
            u, v = random.choice(list(edges))
            remove_edge(u, v)
        
        temp *= cooling
        
        # If stuck for a while, do a 2-swap
        if temp < 1.0 and random.random() < 0.1:
            # 2-swap: remove edge (a,b) and add edge (c,d)
            if len(edges) >= 2:
                e1 = random.choice(list(edges))
                e2 = random.choice(all_pairs)
                a, b = e1
                c, d = e2
                if a != c and a != d and b != c and b != d:
                    # Check if removing (a,b) and adding (c,d) is valid
                    remove_edge(a, b)
                    if not has_edge(c, d) and is_c6_free(c, d):
                        add_edge(c, d)
                        if count_edges() > best_edges:
                            best_edges = count_edges()
                            best_adj = adj[:]
                    else:
                        # Revert
                        add_edge(a, b)
        
        if time.time() > deadline - 5:
            break
    
    # Final greedy pass to add any remaining safe edges
    for u, v in all_pairs:
        if has_edge(u, v):
            continue
        if is_c6_free(u, v):
            add_edge(u, v)
    
    if count_edges() > best_edges:
        best_adj = adj[:]
    
    # Return the best graph found
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                final_edges.append([u, v])
    
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
