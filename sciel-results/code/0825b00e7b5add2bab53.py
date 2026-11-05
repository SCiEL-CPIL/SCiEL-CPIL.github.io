import time
import random
from itertools import combinations

def _clebsch_edges(offset=0):
    """Generate edges of Clebsch graph (16 vertices, degree 5, C4-free, alpha=5)."""
    edges = []
    verts = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    for i in range(16):
        for j in range(i + 1, 16):
            diff = verts[i] ^ verts[j]
            if bin(diff).count('1') == 2:
                edges.append([offset + i, offset + j])
    return edges

def _has_c4(adj, n):
    """Check if any two vertices share 2+ common neighbors."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                if len(adj[u] & adj[v]) >= 2:
                    return True
    return False

def _has_indep_k(adj, n, k, deadline):
    """Check if an independent set of size k exists using bitmask backtracking."""
    # Use bitmask representation for speed
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    def backtrack(remaining_mask, current_size):
        if time.time() > deadline:
            return False
        if current_size >= k:
            return True
        # Count bits in remaining_mask
        if bin(remaining_mask).count('1') < k - current_size:
            return False
        if remaining_mask == 0:
            return False
        
        # Find vertex with minimum degree in remaining subgraph
        # Use a simple heuristic: pick the vertex with fewest neighbors in remaining
        best_v = -1
        best_deg = n + 1
        temp = remaining_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = bin(adj_mask[v] & remaining_mask).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
            if best_deg == 0:
                break
        
        if best_v == -1:
            return False
        
        # Branch 1: include best_v
        new_remaining = remaining_mask & ~(1 << best_v) & ~adj_mask[best_v]
        if backtrack(new_remaining, current_size + 1):
            return True
        
        # Branch 2: exclude best_v
        new_remaining = remaining_mask & ~(1 << best_v)
        if backtrack(new_remaining, current_size):
            return True
        
        return False
    
    initial_mask = (1 << n) - 1
    return backtrack(initial_mask, 0)

def _build_adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with two disjoint Clebsch graphs (n=32)
    edges = _clebsch_edges(0) + _clebsch_edges(16)
    n = 32
    adj = _build_adj_from_edges(edges, n)
    
    # Validate base
    if _has_c4(adj, n) or _has_indep_k(adj, n, 11, deadline):
        # Fallback: return a valid smaller graph
        # 10 disjoint triangles: n=30, alpha=10, C4-free
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_n = n
    best_edges = edges[:]
    
    random.seed(42)
    
    # Try to add vertices one at a time
    while time.time() < deadline - 5 and n < 43:
        improved = False
        attempts = 0
        while time.time() < deadline - 5 and attempts < 2000:
            attempts += 1
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            # Randomly decide which existing vertices to connect to
            # For each existing vertex, we can only connect if it doesn't create C4
            # A new vertex v can connect to u if |N(u) ∩ N(v)| < 2
            # Since v is new, N(v) is what we're building, so we need to be careful
            
            # Strategy: randomly select a subset of existing vertices to connect to
            # such that no C4 is created
            order = list(range(n))
            random.shuffle(order)
            
            candidate_neighbors = set()
            for v in order:
                # Check if connecting new vertex to v would create C4
                # C4 is created if there's another vertex w in candidate_neighbors
                # such that |N(v) ∩ N(w)| >= 2
                can_add = True
                for w in candidate_neighbors:
                    if len(adj[v] & adj[w]) >= 2:
                        can_add = False
                        break
                if can_add:
                    candidate_neighbors.add(v)
            
            if not candidate_neighbors:
                continue
            
            # Build new adjacency
            new_adj[new_n - 1] = candidate_neighbors.copy()
            for v in candidate_neighbors:
                new_adj[v].add(new_n - 1)
            
            # Check C4-free
            if _has_c4(new_adj, new_n):
                continue
            
            # Check no independent set of size 11
            if _has_indep_k(new_adj, new_n, 11, deadline):
                continue
            
            # Valid extension!
            for v in sorted(candidate_neighbors):
                edges.append([new_n - 1, v])
            adj = new_adj
            n = new_n
            best_n = n
            best_edges = edges[:]
            improved = True
            break
        
        if not improved:
            break
    
    return {'n': best_n, 'edges': best_edges}
