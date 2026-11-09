"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random

N = 18
DEADLINE = 115.0  # safety margin before 120s/150s kill


def _is_k4_free(adj):
    """Check if graph (as bitmask adjacency) is K4-free."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] & (1 << b)):
                continue
            for c in range(b + 1, N):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                    continue
                # a,b,c form a triangle; check if any common neighbor d completes K4
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True


def _max_triangle_free_size(adj):
    """
    Find the maximum size of a triangle-free induced subgraph.
    Uses branch and bound.
    """
    # Precompute triangles for faster checking?
    # A subset S is triangle-free if it contains no triangle from the graph.
    # We can iterate through all subsets? 2^18 is ~262k, which is feasible if checking is fast.
    # But checking each subset takes O(N^2) or similar.
    # Better: Branch and bound on vertices.
    
    # Let's use a simpler approach first: check all subsets of size k from N down to 1.
    # To make it fast, we precompute the triangle mask for each pair? No.
    # We check if a mask contains a triangle.
    # Triangle exists in mask M if there is u,v,w in M such that adj[u]&adj[v]&adj[w]... no.
    # Triangle exists if there is an edge (u,v) in M such that (adj[u] & adj[v] & M) != 0.
    
    # To speed up, we can precompute for each pair (u,v) the set of common neighbors.
    # common[u][v] = adj[u] & adj[v]
    # Then for a mask M, it has a triangle if there exists u,v in M such that (common[u][v] & M) != 0.
    
    common = [[0]*N for _ in range(N)]
    for u in range(N):
        for v in range(u+1, N):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c
            
    def has_triangle(mask):
        # Check if induced subgraph on 'mask' has a triangle
        # Iterate through bits in mask
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            u = lsb.bit_length() - 1
            m -= lsb
            # Check edges from u to other vertices in mask
            # Neighbors of u in mask: adj[u] & mask
            neighbors = adj[u] & mask
            while neighbors:
                lsb2 = neighbors & -neighbors
                v = lsb2.bit_length() - 1
                neighbors -= lsb2
                # If u and v are connected (implied by v in adj[u]), check for common neighbor in mask
                # common[u][v] & mask
                if common[u][v] & mask:
                    return True
        return False

    # Search from N down to 1
    for k in range(N, 0, -1):
        # Generate all combinations of size k
        # Using itertools.combinations is slow for large N, but N=18 is small.
        # 18 choose 9 is 48620. 18 choose 10 is 85680.
        # Total checks ~ 2^18 / 18 * something?
        # Actually, we can just iterate all masks and check size?
        # No, we want the LARGEST.
        
        # Optimization: If we find one of size k, we can return k.
        # But we must check ALL of size k to be sure? No, if we find ANY of size k, the max is at least k.
        # Since we go down from N, the first one we find is the max.
        
        # Using itertools
        import itertools
        for subset in itertools.combinations(range(N), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if not has_triangle(mask):
                return k
    return 0


def _random_k4_free_graph():
    """Generate a random K4-free graph (complete tripartite)."""
    coloring = [random.randint(0, 2) for _ in range(N)]
    # Ensure all colors used
    while len(set(coloring)) < 3:
        coloring[random.randint(0, N - 1)] = random.randint(0, 2)
    
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if coloring[u] != coloring[v]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return adj


def _perturb_k4_free(adj):
    """
    Perturb a K4-free graph by adding/removing edges.
    Returns a new K4-free graph.
    """
    new_adj = adj[:]
    
    # Try to add edges
    for _ in range(5):
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v or (new_adj[u] & (1 << v)):
            continue
        
        # Check if adding (u,v) creates a K4
        # K4 is formed if there are x,y such that u-v, u-x, u-y, v-x, v-y, x-y are all edges.
        # x,y must be common neighbors of u and v.
        common = new_adj[u] & new_adj[v]
        
        creates_k4 = False
        # Check if any edge exists within 'common'
        c = common
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c -= lsb
            # Check neighbors of x in common
            neighbors_in_common = new_adj[x] & common
            if neighbors_in_common:
                creates_k4 = True
                break
        if not creates_k4:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
    # Try to remove edges
    for _ in range(3):
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u != v and (new_adj[u] & (1 << v)):
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
    return new_adj


def construct():
    start_time = time.time()
    random.seed(42)
    
    best_score = N + 1
    best_adj = None
    
    # Initial graph
    adj = _random_k4_free_graph()
    score = _max_triangle_free_size(adj)
    best_score = score
    best_adj = adj
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < DEADLINE:
        # Generate candidate
        if random.random() < 0.2:
            cand = _random_k4_free_graph()
        else:
            cand = _perturb_k4_free(best_adj)
            
        if not _is_k4_free(cand):
            continue
            
        # Evaluate
        # Quick check: if we can find a triangle-free subgraph of size best_score, it's not better.
        # But _max_triangle_free_size gives the exact max.
        # To save time, we can check if max_size >= best_score.
        # If max_size < best_score, it's better.
        
        # Since N=18 is small, _max_triangle_free_size is fast enough (~ms).
        # Let's just call it directly.
        score = _max_triangle_free_size(cand)
        
        if score < best_score:
            best_score = score
            best_adj = cand
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart
            best_adj = _random_k4_free_graph()
            best_score = _max_triangle_free_size(best_adj)
            no_improve = 0
            
    # Convert best_adj to edges
    edges = []
    if best_adj is None:
        # Fallback: complete tripartite 6-6-6
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
        return edges

    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges


# Quick self-test
if __name__ == "__main__":
    edges = construct()
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    print(f"K4-free: {_is_k4_free(adj)}")
    print(f"Score: {_max_triangle_free_size(adj)}")
    print(f"Number of edges: {len(edges)}")
