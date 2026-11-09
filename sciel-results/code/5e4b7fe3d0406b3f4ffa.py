"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph.

Strategy:
- Build a dense K4-free graph by starting from a random graph and removing edges
  that complete K4s.
- Use a fast, exact branch-and-bound to compute the size of the largest
  triangle-free induced subgraph.
- Perform local search (edge flips) to minimize that size while preserving
  K4-freeness.
"""
import time
import random
import sys

N = 18
DEADLINE = 145.0  # safety margin under 150s


def _is_k4_free(adj):
    """Check if the graph (adjacency bitmasks) is K4-free."""
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            # Check if common neighborhood has any edge
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if adj[k] & (common & ~(1 << k)):
                    return False
    return True


def _max_triangle_free(adj):
    """Exact branch-and-bound for the size of the largest triangle-free induced subgraph."""
    # Precompute all triangles as bitmasks
    triangles = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return N
    
    # Order vertices by degree (ascending) for better pruning
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    
    best = 0
    sys.setrecursionlimit(10000)
    
    def search(S, candidates):
        nonlocal best
        s_count = bin(S).count('1')
        if s_count > best:
            best = s_count
            if best == N:
                return
        c_count = bin(candidates).count('1')
        if s_count + c_count <= best:
            return
        if candidates == 0:
            return
        
        # Try adding each candidate vertex
        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Check if v can be added to S (S ∪ {v} is triangle-free)
            # v conflicts with S iff there exist a, b in S such that a~b and v~a and v~b
            # i.e., v and some a in S share a common neighbor in S
            can_add = True
            s = S
            while s:
                lsb2 = s & (-s)
                a = lsb2.bit_length() - 1
                s ^= lsb2
                if adj[v] & adj[a] & S:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            new_S = S | lsb
            # New candidates: vertices in c (that come after v in processing order)
            # that can also be added to new_S
            new_c = 0
            cc = c
            while cc:
                lsb3 = cc & (-cc)
                w = lsb3.bit_length() - 1
                cc ^= lsb3
                # Check if w can be added to new_S
                can_add_w = True
                s2 = new_S
                while s2:
                    lsb4 = s2 & (-s2)
                    a2 = lsb4.bit_length() - 1
                    s2 ^= lsb4
                    if adj[w] & adj[a2] & new_S:
                        can_add_w = False
                        break
                if can_add_w:
                    new_c |= lsb3
            
            search(new_S, new_c)
    
    search(0, (1 << N) - 1)
    return best


def _make_k4_free(adj, rng):
    """Remove edges from adj to make it K4-free. Returns modified adj."""
    adj = adj[:]
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    c = common
                    has_edge = False
                    while c:
                        lsb = c & (-c)
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[k] & (common & ~(1 << k)):
                            has_edge = True
                            break
                    if has_edge:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        changed = True
    return adj


def _random_k4_free(rng, density=0.6):
    """Generate a random K4-free graph with approximately the given edge density."""
    adj = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return _make_k4_free(adj, rng)


def _structured_complement(rng):
    """Generate a K4-free graph as complement of a sparse structured graph."""
    # Use a random 3-regular-like graph as the sparse part
    adj_sparse = [0] * N
    # Connect each vertex to a few others
    for v in range(N):
        # Pick 2-3 random neighbors
        others = [u for u in range(N) if u != v]
        rng.shuffle(others)
        for u in others[:2 + rng.randint(0, 1)]:
            adj_sparse[v] |= (1 << u)
            adj_sparse[u] |= (1 << v)
    
    # Complement
    comp = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if not (adj_sparse[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
    
    if _is_k4_free(comp):
        return comp
    return _make_k4_free(comp, rng)


def construct():
    start_time = time.time()
    rng = random.Random(2024)
    
    def elapsed():
        return time.time() - start_time
    
    # Generate initial candidates
    candidates = []
    
    # Several random K4-free graphs with different densities
    for density in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for _ in range(3):
            candidates.append(_random_k4_free(rng, density))
    
    # Structured complements
    for _ in range(5):
        candidates.append(_structured_complement(rng))
    
    # Score all candidates
    best_adj = None
    best_score = N + 1
    
    for adj in candidates:
        if elapsed() > DEADLINE - 80:
            break
        if not _is_k4_free(adj):
            continue
        s = _max_triangle_free(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
    
    if best_adj is None:
        # Fallback: complete graph minus a perfect matching is K4-free? No.
        # Use a 4-partite graph
        adj = [0] * N
        parts = [list(range(0, 5)), list(range(5, 9)), list(range(9, 13)), list(range(13, 18))]
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        best_adj = adj
        best_score = _max_triangle_free(adj)
    
    # Local search
    adj = best_adj[:]
    
    def get_edges(a):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if a[i] & (1 << j):
                    edges.append((i, j))
        return edges
    
    def get_non_edges(a):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if not (a[i] & (1 << j)):
                    edges.append((i, j))
        return edges
    
    improved = True
    while improved and elapsed() < DEADLINE - 10:
        improved = False
        
        # Try removing edges
        edges = get_edges(adj)
        rng.shuffle(edges)
        for (u, v) in edges:
            if elapsed() > DEADLINE - 10:
                break
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            s = _max_triangle_free(adj)
            if s < best_score:
                best_score = s
                best_adj = adj[:]
                improved = True
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Try adding edges (must maintain K4-free)
        non_edges = get_non_edges(adj)
        rng.shuffle(non_edges)
        for (u, v) in non_edges:
            if elapsed() > DEADLINE - 10:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free(adj):
                s = _max_triangle_free(adj)
                if s < best_score:
                    best_score = s
                    best_adj = adj[:]
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Convert to edge list
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    return result


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
