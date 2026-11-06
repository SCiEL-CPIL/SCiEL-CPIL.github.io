"""
K3,3-free graph on 20 vertices: maximize edge count.
Strategy: Start from a strong bipartite construction, then use local search
with edge swaps (2-opt), kick + greedy fill to escape local optima.
"""
import time
import random
from itertools import combinations

N = 20
DEADLINE = 145  # safety margin before 150s limit


def has_k33_fast(adj):
    """Faster K3,3 check using common neighbors.
    For every triple of vertices, check if they share >= 3 common neighbors
    that are not in the triple.
    """
    n = N
    vertices = list(range(n))
    for A in combinations(vertices, 3):
        a1, a2, a3 = A
        common = adj[a1] & adj[a2] & adj[a3]
        common.difference_update(A)
        if len(common) >= 3:
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
    
    n = N
    # Check triples containing u: v becomes a new common neighbor
    for x in range(n):
        if x == u:
            continue
        for y in range(n):
            if y == u or y == x:
                continue
            if v in adj[x] and v in adj[y]:
                common = adj[u] & adj[x] & adj[y]
                if v not in common:
                    common.add(v)
                common.difference_update({u, x, y})
                if len(common) >= 3:
                    return False
    
    # Check triples containing v: u becomes a new common neighbor
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


def try_remove_edge(adj, u, v):
    """Check if removing edge (u,v) keeps graph K3,3-free.
    Removing an edge can only help, so this is always safe."""
    return True


def try_swap_edges(adj, u1, v1, u2, v2):
    """Try to swap: remove (u1,v1) and add (u2,v2).
    Return True if the swap is valid (removal is always safe, addition must be checked).
    We need to check that after removal, the addition doesn't create K3,3.
    """
    if u1 == u2 and v1 == v2:
        return False
    if v1 in adj[u1] is False:
        return False  # edge to remove doesn't exist
    if v2 in adj[u2]:
        return False  # edge to add already exists
    
    # Temporarily remove (u1,v1)
    adj[u1].discard(v1)
    adj[v1].discard(u1)
    
    # Try to add (u2,v2)
    result = try_add_edge(adj, u2, v2)
    
    if result:
        add_edge(adj, u2, v2)
        return True
    else:
        # Restore (u1,v1)
        add_edge(adj, u1, v1)
        return False


def greedy_fill(adj, time_budget=None):
    """Add as many edges as possible in random order."""
    n = N
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(edges)
    added = 0
    for u, v in edges:
        if time_budget is not None and time.time() > time_budget:
            break
        if v not in adj[u]:
            if try_add_edge(adj, u, v):
                add_edge(adj, u, v)
                added += 1
    return added


def edge_swap_local_search(adj, time_budget):
    """Perform local search using edge swaps (2-opt moves)."""
    n = N
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        # Collect current edges
        current_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges.append((u, v))
        
        # Collect non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    non_edges.append((u, v))
        
        if not current_edges or not non_edges:
            break
        
        # Try random swaps
        n_swaps = min(500, len(current_edges) * len(non_edges))
        for _ in range(n_swaps):
            if time.time() > time_budget:
                break
            e1 = random.choice(current_edges)
            e2 = random.choice(non_edges)
            # Make sure we're not swapping an edge with itself
            if e1 == e2:
                continue
            # Try the swap
            if try_swap_edges(adj, e1[0], e1[1], e2[0], e2[1]):
                improved = True
                # Update lists
                current_edges.remove(e1)
                current_edges.append(e2)
                non_edges.remove(e2)
                non_edges.append(e1)
                break  # Restart after a successful swap
    
    return improved


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
    """
    adj = [set() for _ in range(N)]
    A = list(range(10))
    B = list(range(10, 20))
    
    best_adj = None
    best_count = 0
    
    for seed in range(100):
        rng = random.Random(seed)
        adj_try = [set() for _ in range(N)]
        
        # Assign neighborhoods for A vertices with varying degrees
        for a in A:
            # Try different degree patterns
            deg = rng.randint(4, 6)
            neighbors = rng.sample(B, deg)
            for b in neighbors:
                adj_try[a].add(b)
                adj_try[b].add(a)
        
        # Check K3,3 constraint
        valid = True
        for triple in combinations(A, 3):
            common = adj_try[triple[0]] & adj_try[triple[1]] & adj_try[triple[2]]
            common.difference_update(triple)
            if len(common) >= 3:
                valid = False
                break
        
        if valid:
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
    
    # Local search with kicks and edge swaps
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        
        # Alternate between kick+greedy and edge swaps
        if iterations % 2 == 1:
            # Kick + greedy
            kick(adj, n_remove=random.randint(2, 5), n_add=random.randint(3, 8))
            greedy_fill(adj)
        else:
            # Edge swap local search
            time_budget = min(time.time() + 5, deadline - 10)
            edge_swap_local_search(adj, time_budget)
        
        current_count = edge_count(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = [set(s) for s in adj]
        
        # Occasional full greedy fill
        if iterations % 5 == 0:
            greedy_fill(adj)
            current_count = edge_count(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = [set(s) for s in adj]
    
    # Final polish: multiple greedy passes and edge swaps on best
    final_deadline = deadline - 2
    for _ in range(3):
        if time.time() > final_deadline - 5:
            break
        greedy_fill(best_adj)
        edge_swap_local_search(best_adj, final_deadline - 2)
    
    # Final greedy fill
    greedy_fill(best_adj)
    
    # Verify no K3,3
    if has_k33_fast(best_adj):
        # Fallback to a known valid graph
        adj_fallback = [set() for _ in range(N)]
        for a in range(2):
            for b in range(2, N):
                adj_fallback[a].add(b)
                adj_fallback[b].add(a)
        best_adj = adj_fallback
    
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
