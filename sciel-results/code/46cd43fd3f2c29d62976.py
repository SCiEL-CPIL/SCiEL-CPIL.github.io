"""K2,3-free graph on 20 vertices: maximize edges via structured init + bitmask greedy + local search."""
import time
import random

N = 20
DEADLINE_MARGIN = 15.0  # seconds before hard limit to stop


def _is_valid_addition(adj_bitmasks, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free."""
    for w in range(N):
        if w == u or w == v:
            continue
        if (adj_bitmasks[v] >> w) & 1:
            cnt = (adj_bitmasks[u] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
        if (adj_bitmasks[u] >> w) & 1:
            cnt = (adj_bitmasks[v] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
    return True


def _popcount(x):
    return x.bit_count()


def _count_edges(adj):
    total = 0
    for u in range(N):
        total += adj[u].bit_count()
    return total // 2


def _is_valid_graph(adj):
    """Check if graph is K2,3-free."""
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] & adj[v]).bit_count() >= 3:
                return False
    return True


def _greedy_fill(adj, degrees, edge_set, deadline):
    """Greedily add edges to maximize density while maintaining K2,3-free."""
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() < deadline:
        best_edge = None
        best_score = None
        
        for (u, v) in all_edges:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            
            if not _is_valid_addition(adj, u, v):
                continue
            
            new_du = degrees[u] + 1
            new_dv = degrees[v] + 1
            max_deg = max(new_du, new_dv)
            score = (max_deg, degrees[u] + degrees[v])
            
            if best_score is None or score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    return edge_set


def _local_search(adj, degrees, edge_set, deadline):
    """Try edge swaps: remove one edge, add up to two edges."""
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    current_count = _count_edges(adj)
    
    # Try single edge additions first
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for (u, v) in all_edges:
            if (u, v) in edge_set:
                continue
            if _is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
                current_count += 1
                changed = True
    
    # Try edge swaps: remove edge (a,b), add edges (c,d) and (e,f)
    while time.time() < deadline:
        improved = False
        for (a, b) in list(edge_set):
            if a > b:
                continue
            # Remove edge (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            degrees[a] -= 1
            degrees[b] -= 1
            edge_set.discard((a, b))
            edge_set.discard((b, a))
            
            # Try adding two edges
            added = []
            for (c, d) in all_edges:
                if (c, d) in edge_set or (d, c) in edge_set:
                    continue
                if c == a or c == b or d == a or d == b:
                    continue
                if _is_valid_addition(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    degrees[c] += 1
                    degrees[d] += 1
                    edge_set.add((c, d))
                    edge_set.add((d, c))
                    added.append((c, d))
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                # We removed 1, added 2, net +1
                improved = True
                current_count += 1
                # Keep this improvement
            else:
                # Revert: remove added edges
                for (c, d) in added:
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
                    degrees[c] -= 1
                    degrees[d] -= 1
                    edge_set.discard((c, d))
                    edge_set.discard((d, c))
                # Re-add original edge
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                degrees[a] += 1
                degrees[b] += 1
                edge_set.add((a, b))
                edge_set.add((b, a))
        
        if not improved:
            break
    
    return current_count


def _structured_init():
    """Create a structured initial graph using 5-cliques with specific inter-clique connections."""
    adj = [0] * N
    degrees = [0] * N
    edge_set = set()
    
    # Create 5 cliques of size 4
    cliques = [[i*4 + j for j in range(4)] for i in range(5)]
    
    # Add intra-clique edges
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                u, v = clique[i], clique[j]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Add inter-clique edges: connect clique i to clique (i+1)%5 with a matching
    for i in range(5):
        clique_a = cliques[i]
        clique_b = cliques[(i+1) % 5]
        # Connect first 2 vertices of each clique
        for k in range(2):
            u = clique_a[k]
            v = clique_b[k]
            if _is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Also connect clique 0 to clique 2
    clique_a = cliques[0]
    clique_b = cliques[2]
    for k in range(2):
        u = clique_a[k+2]
        v = clique_b[k]
        if _is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            edge_set.add((u, v))
            edge_set.add((v, u))
    
    return adj, degrees, edge_set


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    best_adj = [0] * N
    best_degrees = [0] * N
    best_edge_set = set()
    best_count = 0
    
    # Run multiple attempts with different strategies
    attempts = []
    
    # Attempt 1: Structured init + greedy fill
    adj1, deg1, es1 = _structured_init()
    es1 = _greedy_fill(adj1, deg1, es1, deadline)
    cnt1 = _count_edges(adj1)
    if cnt1 > best_count:
        best_adj = adj1[:]
        best_degrees = deg1[:]
        best_edge_set = es1.copy()
        best_count = cnt1
    
    # Attempt 2: Random restart with different edge orderings
    for attempt in range(3):
        if time.time() >= deadline:
            break
        
        adj2 = [0] * N
        deg2 = [0] * N
        es2 = set()
        
        # Randomized greedy
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() >= deadline:
                break
            if (u, v) in es2:
                continue
            if _is_valid_addition(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
                deg2[u] += 1
                deg2[v] += 1
                es2.add((u, v))
                es2.add((v, u))
        
        cnt2 = _count_edges(adj2)
        if cnt2 > best_count:
            best_adj = adj2[:]
            best_degrees = deg2[:]
            best_edge_set = es2.copy()
            best_count = cnt2
    
    # Local search on best solution
    if best_count > 0 and time.time() < deadline:
        local_deadline = min(deadline, time.time() + 30)
        best_count = _local_search(best_adj, best_degrees, best_edge_set, local_deadline)
    
    # Convert to edge list
    edges = []
    for (u, v) in best_edge_set:
        if u < v:
            edges.append([u, v])
    
    # Final validity check
    if not _is_valid_graph(best_adj):
        # Fallback: return empty or minimal valid graph
        return []
    
    return edges
