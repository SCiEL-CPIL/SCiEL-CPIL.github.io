import time
import random
from collections import defaultdict

def _is_c4_free(n, adj):
    """Check if graph with adjacency sets is C4-free.
    A C4 exists if there are two vertices u,v with |N(u) ∩ N(v)| >= 2
    where u and v are not adjacent... actually no.
    A C4 is u-w-v-x-u. This exists if there exist u,v (not necessarily non-adjacent)
    such that |N(u) ∩ N(v)| >= 2 and u != v and u not in N(v) or v not in N(u) appropriately.
    
    Actually, a C4 exists iff there exist two vertices u, v with u != v such that
    |N(u) ∩ N(v)| >= 2 and u is not adjacent to v, OR more generally,
    there exist four distinct vertices forming a cycle.
    
    Simpler: a C4 exists iff there exist u, v with u != v and |N(u) ∩ N(v)| >= 2
    where u and v are NOT adjacent. Wait, that's not quite right either.
    
    Actually: A C4 is a cycle of length 4: u-w-v-x-u where all four are distinct.
    This means w and x are both neighbors of u and v, so w, x ∈ N(u) ∩ N(v),
    and u-v is not an edge (otherwise it'd be a triangle). Also u, v, w, x all distinct.
    
    So: C4 exists iff there exist non-adjacent u, v with |N(u) ∩ N(v)| >= 2.
    But we also need w != v and x != u, which is automatic if u,v non-adjacent and 
    w,x are neighbors of both.
    
    Wait, we also need w != x (they're distinct vertices in the cycle).
    So: C4 exists iff there exist non-adjacent u, v with |N(u) ∩ N(v)| >= 2.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:  # u and v not adjacent
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
    return True

def _count_c4s(n, adj):
    """Count number of C4s (as pairs of non-adjacent vertices with >=2 common neighbors)."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                common = len(adj[u] & adj[v])
                if common >= 2:
                    count += 1
    return count

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to build C4-free graphs with high minimum degree
    # Strategy: start from polarity graph, then use simulated annealing
    
    best_result = {'n': 39, 'edges': []}
    best_score = 39  # track best n achieved
    
    # First, try polarity graph constructions for various q
    # Polarity graph of projective plane of order q: n = q^2+q+1
    # For q=6: n=43
    
    def make_polarity_graph(q):
        """Construct polarity graph from projective plane of order q.
        Uses the standard construction for q a prime power.
        Vertices are points of PG(2,q), edges connect perpendicular points.
        """
        n = q * q + q + 1
        # For simplicity, use a known construction
        # Points are (x, y, 1) for affine, (x, 1, 0) for line at infinity, (0,0,1) ... 
        # This is complex. Let me use a different approach.
        # Actually, let me just use a greedy/SA approach instead.
        return None
    
    # Simulated annealing approach
    # State: set of edges in a graph on n vertices
    # Objective: maximize n subject to C4-free and min_degree >= n-39
    
    # Let's try multiple n values and use SA to find valid graphs
    
    def sa_for_n(n, max_iter=500000, seed=42):
        """Try to find a C4-free graph on n vertices with min degree >= n-39."""
        rng = random.Random(seed)
        required_min_deg = n - 39
        if required_min_deg < 1:
            required_min_deg = 1
        
        # Start with a random graph with appropriate density
        # We want min degree >= required_min_deg, so average degree should be higher
        # Expected degree ~ n * p, so p ~ required_min_deg / n + some margin
        
        # Start with a regular-ish graph
        # Use a random regular graph as starting point
        target_deg = required_min_deg + 1  # aim slightly above minimum
        
        # Build initial graph: try to make it (target_deg)-regular
        adj = [set() for _ in range(n)]
        
        # Simple approach: start with complete graph and remove edges to break C4s
        # Or start sparse and add edges
        
        # Let's start with a random graph with edge probability p
        p = min(0.9, (required_min_deg + 2) / n)
        
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Check initial state
        c4_count = _count_c4s(n, adj)
        min_deg = _min_degree(n, adj)
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.9999
        
        # Penalty function: we want c4_count = 0 and min_deg >= required_min_deg
        # Score = -c4_count * 1000 + max(0, min_deg - required_min_deg)
        
        def score(adj):
            c4 = _count_c4s(n, adj)
            md = _min_degree(n, adj)
            if c4 > 0:
                return -c4 * 10000 - md
            else:
                return md if md >= required_min_deg else md - (required_min_deg - md) * 1000
        
        current_score = score(adj)
        best_adj = [set(s) for s in adj]
        best_score_local = current_score
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            # Compute delta in score efficiently
            # Flipping edge (u,v) affects C4 count and degrees
            
            is_adj = v in adj[u]
            
            # Temporarily flip
            if is_adj:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
            
            new_score = score(adj)
            delta = new_score - current_score
            
            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_adj = [set(s) for s in adj]
            else:
                # Revert
                if is_adj:
                    adj[u].add(v)
                    adj[v].add(u)
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
            
            T = max(T * cooling, T_min)
            
            # Early termination if we found a valid solution
            if best_score_local >= required_min_deg and _count_c4s(n, best_adj) == 0:
                break
        
        # Check if best is valid
        if _count_c4s(n, best_adj) == 0 and _min_degree(n, best_adj) >= required_min_deg:
            edges = []
            for i in range(n):
                for j in best_adj[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        return None
    
    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        if time.time() > deadline:
            break
        
        result = sa_for_n(n, max_iter=200000, seed=n)
        if result is not None:
            if n > best_score:
                best_result = result
                best_score = n
            # Found a valid graph, try to improve or stop
            break  # Found the largest n, return it
    
    # If SA didn't work well, try a greedy construction
    if best_score <= 39:
        # Greedy: build graph vertex by vertex
        for n in range(47, 42, -1):
            if time.time() > deadline:
                break
            
            required_min_deg = max(1, n - 39)
            adj = [set() for _ in range(n)]
            
            # Add edges greedily, avoiding C4s
            # Order: try to add edges that maintain C4-freeness
            edge_order = []
            for i in range(n):
                for j in range(i + 1, n):
                    edge_order.append((i, j))
            random.Random(n + 100).shuffle(edge_order)
            
            for (u, v) in edge_order:
                if time.time() > deadline:
                    break
                # Check if adding (u,v) creates a C4
                # Adding (u,v) creates a C4 iff there exist w, x such that
                # w, x ∈ N(u) ∩ N(v) and w != x and w != v and x != u
                # Since u,v not yet adjacent, N(u) ∩ N(v) gives common neighbors
                common = adj[u] & adj[v]
                if len(common) < 2:
                    # Safe to add
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check result
            if _count_c4s(n, adj) == 0 and _min_degree(n, adj) >= required_min_deg:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                result = {'n': n, 'edges': edges}
                if n > best_score:
                    best_result = result
                    best_score = n
                break
    
    # Final fallback: try a specific construction
    # For n=43, use a known C4-free graph with high minimum degree
    # The polarity graph of the projective plane of order 6 doesn't exist (6 is not prime power)
    # But we can use order 5: n = 25+5+1 = 31, or order 7: n = 49+7+1 = 57 (too big)
    
    # Try a different approach: use a bipartite-like structure
    # Or use the fact that we can construct C4-free graphs with specific degree sequences
    
    # If we still have time, try more SA iterations for the best n found
    if best_score < 47 and time.time() < deadline - 5:
        for n in range(min(47, best_score + 3), best_score, -1):
            if time.time() > deadline - 2:
                break
            for seed in range(5):
                if time.time() > deadline - 2:
                    break
                result = sa_for_n(n, max_iter=100000, seed=seed * 1000 + n)
                if result is not None and n > best_score:
                    best_result = result
                    best_score = n
                    break
    
    return best_result
