"""
K3,3-free graph on 20 vertices: maximize edge count.
Strategy: Start from a strong bipartite construction, then use local search
with kick + greedy fill to escape local optima.
"""
import time
import random
from itertools import combinations

N = 20
DEADLINE = 110  # safety margin before 120s limit


def has_k33(adj, edge_set):
    """Check if the graph contains a K3,3 subgraph.
    adj: list of sets, adj[i] = neighbors of i
    edge_set: set of frozenset({u,v}) for quick lookup
    Returns True if K3,3 found."""
    n = N
    # A K3,3 has two parts of size 3, all 9 cross edges present.
    # For each pair of disjoint 3-sets {a,b,c} and {d,e,f}, check all 9 edges.
    vertices = list(range(n))
    for part1 in combinations(vertices, 3):
        for part2 in combinations(vertices, 3):
            if set(part1) & set(part2):
                continue
            # Check all 9 cross edges
            ok = True
            for u in part1:
                for v in part2:
                    if v not in adj[u]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def has_k33_fast(adj):
    """Faster K3,3 check using common neighbors.
    For every triple of vertices, check if they share >= 3 common neighbors
    that are not in the triple.
    Actually, K3,3 means: exist disjoint sets A={a1,a2,a3}, B={b1,b2,b3}
    such that every ai is adjacent to every bj.
    Equivalent: for some triple A, |N(a1) ∩ N(a2) ∩ N(a3)| >= 3 and
    the three common neighbors are disjoint from A.
    """
    n = N
    vertices = list(range(n))
    for A in combinations(vertices, 3):
        a1, a2, a3 = A
        common = adj[a1] & adj[a2] & adj[a3]
        # Remove the vertices of A themselves (they can't be in common neighbors
        # since graph is simple and no self-loops, but just in case)
        common.difference_update(A)
        if len(common) >= 3:
            # Need to verify: the 3 common neighbors form a set B disjoint from A
            # and all edges between A and B exist. By definition of common neighbors,
            # every vertex in common is adjacent to all of a1,a2,a3.
            # So any 3 from common form a valid B.
            return True
    return False


def edge_count(adj):
    return sum(len(s) for s in adj) // 2


def add_edge(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)


def remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def try_add_edge(adj, u, v):
    """Try to add edge (u,v). Return True if it doesn't create K3,3."""
    if v in adj[u]:
        return True  # already present
    # Adding edge (u,v) creates K3,3 iff there exist a1,a2,a3 with
    # v in N(a1)∩N(a2)∩N(a3) and u in that common set, OR symmetric.
    # More precisely: after adding (u,v), check if any triple has >=3 common neighbors.
    # The new edge (u,v) can only affect triples that include u or v.
    # Actually it's simpler to just check: does adding (u,v) make some triple have 3 common neighbors?
    # The edge (u,v) contributes to common neighbor sets of triples containing both u and v? No.
    # Edge (u,v) means v is a neighbor of u and u is a neighbor of v.
    # For a triple T, v becomes a common neighbor of T if v is adjacent to all three vertices in T.
    # Before adding, v was adjacent to some subset. After adding (u,v), v is also adjacent to u.
    # So the only triples whose common-neighbor set changes are those that include u
    # and for which v was already adjacent to the other two.
    
    # Check triples containing u: for each pair (x,y) where x,y != u,
    # if v is adjacent to x and y (before adding), then after adding (u,v),
    # v becomes a common neighbor of {u,x,y} (if it wasn't already, which it wasn't since u-v wasn't edge).
    # Similarly for triples containing v.
    
    n = N
    for x in range(n):
        if x == u:
            continue
        for y in range(n):
            if y == u or y == x:
                continue
            # Triple {u, x, y}: after adding (u,v), is v a common neighbor?
            # Need v adjacent to u (yes, we're adding), v adjacent to x, v adjacent to y.
            if v in adj[x] and v in adj[y]:
                # v is now a common neighbor of {u,x,y}
                # Check if |N(u)∩N(x)∩N(y)| >= 3 after adding
                # N(u) gains v. So new common = (old common) ∪ {v if v in N(x)∩N(y)}
                common = adj[u] & adj[x] & adj[y]
                # v is in adj[x] and adj[y], and we're adding to adj[u]
                if v not in common:
                    common.add(v)
                # Remove u,x,y from common (they can't be common neighbors of themselves)
                common.difference_update({u, x, y})
                if len(common) >= 3:
                    return False
    
    # By symmetry, check triples containing v where u becomes a new common neighbor
    for x in range(n):
        if x == v:
            continue
        for y in range(n):
            if y == v or y == x:
                continue
            if u in adj[x] and u in adj[y]:
                common = adj[v] & adj[x] & adj[y]
                if u not in common:
                    common.add(u)
                common.difference_update({v, x, y})
                if len(common) >= 3:
                    return False
    
    return True


def greedy_fill(adj):
    """Add as many edges as possible in random order."""
    n = N
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(edges)
    added = 0
    for u, v in edges:
        if v not in adj[u]:
            if try_add_edge(adj, u, v):
                add_edge(adj, u, v)
                added += 1
    return added


def kick(adj, n_remove=3, n_add=5):
    """Remove random edges and re-add to escape local optima."""
    n = N
    # Collect current edges
    current_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                current_edges.append((u, v))
    
    if not current_edges:
        return
    
    # Remove some random edges
    to_remove = random.sample(current_edges, min(n_remove, len(current_edges)))
    for u, v in to_remove:
        remove_edge(adj, u, v)
    
    # Try to add back and more
    greedy_fill(adj)


def bipartite_seed():
    """Create a bipartite graph with parts of size 10.
    Use a structured pattern to maximize edges while keeping K3,3-free.
    In a bipartite graph with parts A and B, K3,3 exists iff there are
    3 vertices in A and 3 in B with all 9 cross-edges.
    To avoid this, for any 3 vertices in A, they should share at most 2 common neighbors in B.
    Equivalently, the neighborhood sets of vertices in A should have pairwise intersections
    such that no 3 sets have a common intersection of size >= 3.
    
    A simple approach: use a projective plane-like structure or just a random bipartite
    graph with degree ~5-6 and verify.
    
    Let's use: partition into A={0..9}, B={10..19}.
    For each a in A, connect to 5 random vertices in B, ensuring the constraint.
    """
    adj = [set() for _ in range(N)]
    A = list(range(10))
    B = list(range(10, 20))
    
    # Use a structured approach: each vertex in A connects to a subset of B
    # such that any 3 vertices in A share at most 2 common neighbors in B.
    # This is equivalent to a 3-(10,5,?) design or just use random with checking.
    
    # Try multiple random seeds
    best_adj = None
    best_count = 0
    
    for seed in range(50):
        rng = random.Random(seed)
        adj_try = [set() for _ in range(N)]
        
        # Assign neighborhoods for A vertices
        neighborhoods = []
        for a in A:
            # Choose 5 neighbors in B
            neighbors = rng.sample(B, 5)
            neighborhoods.append(set(neighbors))
            for b in neighbors:
                adj_try[a].add(b)
                adj_try[b].add(a)
        
        # Check K3,3 constraint (bipartite, so only need to check A triples and B triples)
        valid = True
        for triple in combinations(A, 3):
            common = adj_try[triple[0]] & adj_try[triple[1]] & adj_try[triple[2]]
            common.difference_update(triple)
            if len(common) >= 3:
                valid = False
                break
        
        if valid:
            # Also check B triples (should be symmetric in bipartite)
            for triple in combinations(B, 3):
                common = adj_try[triple[0]] & adj_try[triple[1]] & adj_try[triple[2]]
                common.difference_update(triple)
                if len(common) >= 3:
                    valid = False
                    break
        
        if valid:
            ec = edge_count(adj_try)
            if ec > best_count:
                best_count = ec
                best_adj = [set(s) for s in adj_try]
    
    if best_adj is None:
        # Fallback: simple K2,18
        adj = [set() for _ in range(N)]
        for a in range(2):
            for b in range(2, N):
                adj[a].add(b)
                adj[b].add(a)
        return adj
    
    return best_adj


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start from bipartite seed
    adj = bipartite_seed()
    
    # Greedy fill to add more edges
    greedy_fill(adj)
    
    best_adj = [set(s) for s in adj]
    best_count = edge_count(adj)
    
    # Local search with kicks
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Kick
        kick(adj, n_remove=random.randint(2, 5), n_add=random.randint(3, 8))
        
        current_count = edge_count(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = [set(s) for s in adj]
        
        # Occasional full greedy fill
        if iterations % 10 == 0:
            greedy_fill(adj)
            current_count = edge_count(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = [set(s) for s in adj]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final greedy fill on best
    greedy_fill(best_adj)
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges


if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
