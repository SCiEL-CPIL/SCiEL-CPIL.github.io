"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 145.0
_VERTS = 18
_EDGE_COUNT = _VERTS * (_VERTS - 1) // 2  # 153

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * 17 - u * (u + 1) // 2 + v - u - 1

# Precompute edge indices
_EDGE_MAP = {}
_EDGE_LIST = []
for u in range(_VERTS):
    for v in range(u + 1, _VERTS):
        idx = len(_EDGE_LIST)
        _EDGE_LIST.append((u, v))
        _EDGE_MAP[(u, v)] = idx
        _EDGE_MAP[(v, u)] = idx

def _idx(u, v):
    return _EDGE_MAP[(u, v)]

def _is_k4_free(adj):
    """Check if graph (as 18x18 bool matrix) is K4-free."""
    n = _VERTS
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _largest_triangle_free(adj):
    """Find the largest triangle-free induced subgraph size via branch and bound."""
    n = _VERTS
    # Precompute common neighbors for fast triangle checking
    # cn[u][v] = bitmask of w != u,v such that adj[u][w] and adj[v][w]
    cn = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            mask = 0
            for w in range(n):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    mask |= (1 << w)
            cn[u][v] = mask
            cn[v][u] = mask

    best = [0]
    
    def can_add(vertex, current_mask):
        """Check if vertex can be added to current_mask without creating a triangle."""
        for u_bit in range(n):
            if current_mask & (1 << u_bit):
                u = u_bit
                if cn[u][vertex] & current_mask:
                    return False
        return True

    def branch(remaining, current_mask, current_size):
        # Upper bound: current_size + popcount(remaining)
        if current_size + bin(remaining).count('1') <= best[0]:
            return
        if remaining == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick vertex with highest degree in remaining subgraph
        # Actually, to find large independent-like sets, maybe pick vertex with
        # fewest conflicts? Or just iterate.
        # For speed, let's try adding vertices in order of index, but with pruning.
        # Better: pick a vertex v from remaining. Branch on include/exclude.
        
        # Pick vertex with max degree in remaining? Or min?
        # If we include it, we must remove its "triangle neighbors" from remaining?
        # No, triangle-free means no 3-clique.
        # If we include v, we cannot include any pair (u,w) in remaining 
        # such that u-w edge exists and u,w are both adjacent to v? No.
        # If we include v, and we already have u in current_mask, 
        # we cannot add w if w is adjacent to both u and v.
        
        # Let's stick to the simple check:
        # Iterate through remaining vertices.
        # To prune, we can order vertices.
        
        # Pick first set bit in remaining
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        remaining_without_v = remaining ^ lsb
        
        # Option 1: Include v
        if can_add(v, current_mask):
            new_mask = current_mask | lsb
            # What vertices become unavailable?
            # None immediately, but future additions must respect triangle-free.
            # However, if we add v, we cannot add any w in remaining_without_v 
            # such that there exists u in current_mask with w adjacent to u and v.
            # This is handled by can_add in recursive call.
            # But we can prune remaining: 
            # If w is in remaining_without_v, and there is u in current_mask 
            # such that adj[u][w] and adj[v][w], then w cannot be added.
            # So we can remove such w from remaining.
            new_remaining = remaining_without_v
            for u_bit in range(n):
                if current_mask & (1 << u_bit):
                    u = u_bit
                    # Remove w such that cn[u][v] has w? No, cn[u][v] is w adj to both u and v.
                    # If w is in cn[u][v], then w cannot be added.
                    new_remaining &= ~cn[u][v]
            
            branch(new_remaining, new_mask, current_size + 1)
            
        # Option 2: Exclude v
        branch(remaining_without_v, current_mask, current_size)

    # Start with all vertices
    all_mask = (1 << n) - 1
    branch(all_mask, 0, 0)
    
    return best[0]

def _score_graph(adj):
    """Compute the score: largest triangle-free induced subgraph size."""
    if not _is_k4_free(adj):
        return _VERTS + 1
    return _largest_triangle_free(adj)

def _greedy_construct(random_seed=None):
    """Greedy randomized construction of a K4-free graph."""
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random

    adj = [[False] * _VERTS for _ in range(_VERTS)]
    edges = []

    all_edges = [(u, v) for u in range(_VERTS) for v in range(u + 1, _VERTS)]
    rng.shuffle(all_edges)

    for (u, v) in all_edges:
        adj[u][v] = adj[v][u] = True
        if not _is_k4_free(adj):
            adj[u][v] = adj[v][u] = False
        else:
            edges.append([u, v])

    return edges, adj

def _local_search(edges, adj, max_time=30.0):
    """Local search: try edge swaps to reduce triangle-free set size."""
    global _DEADLINE
    start_time = time.time()
    best_score = _score_graph(adj)
    best_edges = [e[:] for e in edges]
    best_adj = [row[:] for row in adj]

    if best_score <= 8:
        return best_edges, best_adj

    all_edges = [(u, v) for u in range(_VERTS) for v in range(u + 1, _VERTS)]

    while time.time() < start_time + max_time and time.time() < _DEADLINE:
        improved = False
        rng = random.Random(int(time.time() * 1000) % 100000)
        rng.shuffle(all_edges)

        for (u, v) in all_edges:
            if time.time() >= start_time + max_time or time.time() >= _DEADLINE:
                break

            # Try adding edge (u,v) if not present
            if not adj[u][v]:
                adj[u][v] = adj[v][u] = True
                if _is_k4_free(adj):
                    score = _score_graph(adj)
                    if score < best_score:
                        best_score = score
                        best_edges = [e[:] for e in edges]
                        best_edges.append([u, v])
                        best_adj = [row[:] for row in adj]
                        improved = True
                    else:
                        adj[u][v] = adj[v][u] = False
                else:
                    adj[u][v] = adj[v][u] = False
            else:
                # Try removing edge (u,v)
                adj[u][v] = adj[v][u] = False
                score = _score_graph(adj)
                if score < best_score:
                    best_score = score
                    best_edges = [e[:] for e in best_edges if not (set(e) == {u, v})]
                    best_adj = [row[:] for row in adj]
                    improved = True
                else:
                    adj[u][v] = adj[v][u] = True

        if not improved:
            break

    return best_edges, best_adj

def construct():
    """Construct a K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
    global _DEADLINE
    start_time = time.time()

    best_edges = None
    best_score = _VERTS + 1
    best_adj = None

    # Try multiple random greedy constructions
    num_trials = 50
    for trial in range(num_trials):
        if time.time() >= _DEADLINE - 10:
            break

        edges, adj = _greedy_construct(random_seed=trial * 12345 + 67890)
        score = _score_graph(adj)

        if score < best_score:
            best_score = score
            best_edges = edges
            best_adj = adj

        # Local search on promising candidates
        if score <= best_score and time.time() < _DEADLINE - 20:
            edges2, adj2 = _local_search(edges, adj, max_time=15.0)
            score2 = _score_graph(adj2)
            if score2 < best_score:
                best_score = score2
                best_edges = edges2
                best_adj = adj2

        if best_score <= 8:
            break

    # If we haven't found a good graph, try a structured approach
    if best_score > 9 and time.time() < _DEADLINE - 5:
        for seed in range(20):
            if time.time() >= _DEADLINE - 3:
                break
            edges, adj = _greedy_construct(random_seed=seed * 999983 + 1)
            score = _score_graph(adj)
            if score < best_score:
                best_score = score
                best_edges = edges
                best_adj = adj

    if best_edges is None:
        # Fallback: complete tripartite
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3)
                      for a in parts[i] for b in parts[j]]

    return best_edges
