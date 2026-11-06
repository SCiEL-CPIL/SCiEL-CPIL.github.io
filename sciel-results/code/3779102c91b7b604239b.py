"""Rewritten program using constraint propagation with conflict maximization."""
import itertools
import random
import time

def _is_regular_subgraph(vertices, adj_matrix, n):
    """Check if the induced subgraph on 'vertices' is regular."""
    k = len(vertices)
    if k < 2:
        return True  # 0 or 1 vertex graphs are regular
    # Compute degrees within the induced subgraph
    degs = []
    for v in vertices:
        d = 0
        for u in vertices:
            if u != v and adj_matrix[v][u]:
                d += 1
        degs.append(d)
    # Check if all degrees are equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def construct():
    # We want to maximize n such that no 6+ vertex subset induces a regular subgraph.
    # Known: 20-vertex graphs exist. Let's try to construct one via local search.
    # Strategy: start with a random graph on n vertices, then fix violations.
    
    # Precompute all 6-vertex subsets for a given n
    # For n=20, C(20,6) = 38760 subsets - manageable
    # For n=21, C(21,6) = 54264 - still manageable
    
    def build_and_optimize(n, max_time=110):
        start_time = time.time()
        
        # Initialize with a random graph
        # Use a moderately dense random graph as starting point
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Precompute all 6-vertex subsets
        subsets6 = list(itertools.combinations(range(n), 6))
        
        # Also need to check subsets of size > 6, but if we ensure no 6-vertex 
        # subset is regular, that's not sufficient. We need to check all subsets 
        # of size >= 6. However, checking all is expensive.
        # Key insight: if a subset of size k >= 6 is regular, it contains 6-vertex 
        # subsets that may or may not be regular. So we can't just check size-6.
        # We need to check all subsets of size >= 6.
        # For n=20, total subsets of size >= 6 is sum(C(20,k) for k=6..20) which is large.
        # But in practice, we can focus on size-6 subsets first, then verify.
        
        # Actually, let's be more careful. The condition is that NO subset of size >= 6 
        # induces a regular graph. Let's check size 6 first, then verify larger ones.
        
        def count_violations(adj):
            """Count number of violating subsets (size >= 6 that are regular)."""
            count = 0
            # Check all 6-vertex subsets
            for subset in subsets6:
                if _is_regular_subgraph(subset, adj, n):
                    count += 1
            # For larger subsets, we can't easily precompute, so we'll check a sample
            # or just rely on the 6-vertex check and final verification
            return count
        
        def get_violating_subsets(adj):
            """Return list of violating 6-vertex subsets."""
            viol = []
            for subset in subsets6:
                if _is_regular_subgraph(subset, adj, n):
                    viol.append(subset)
            return viol
        
        # Iterative improvement: find violations, flip edges to maximize resolved conflicts
        for iteration in range(20000):
            if time.time() - start_time > max_time:
                break
            
            viol = get_violating_subsets(adj)
            if not viol:
                # No 6-vertex violations; now verify no larger subset is regular
                # This is expensive, so do it carefully
                if _verify_no_large_regular(adj, n):
                    return adj, True
                # If there's a large regular subset, we need to break it
                # Find a violating large subset
                large_viol = _find_large_regular(adj, n)
                if large_viol is not None:
                    # Pick an edge within this subset to flip
                    subset = large_viol
                    # Find an edge to flip that helps
                    best_flip = None
                    best_score = -1
                    for i in range(len(subset)):
                        for j in range(i+1, len(subset)):
                            u, v = subset[i], subset[j]
                            # Try flipping this edge
                            adj[u][v] ^= 1
                            adj[v][u] ^= 1
                            # Count how many 6-subsets within this region are fixed
                            # Simplified: just count total violations after flip
                            score = count_violations(adj)
                            adj[u][v] ^= 1
                            adj[v][u] ^= 1
                            if score < best_score or best_flip is None:
                                best_score = score
                                best_flip = (u, v)
                    if best_flip:
                        u, v = best_flip
                        adj[u][v] ^= 1
                        adj[v][u] ^= 1
                continue
            
            # We have violations. Find the edge flip that maximizes resolved conflicts.
            # For each violating subset, consider flipping edges within it.
            # Track which edges appear in how many violations.
            edge_viol_count = {}
            for subset in viol:
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        u, v = subset[i], subset[j]
                        key = (min(u,v), max(u,v))
                        edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
            
            # Try flipping the most common edge in violations
            if not edge_viol_count:
                break
            
            # Sort by frequency, try top candidates
            sorted_edges = sorted(edge_viol_count.items(), key=lambda x: -x[1])
            improved = False
            for (u, v), _ in sorted_edges[:10]:
                # Try flipping
                old_val = adj[u][v]
                adj[u][v] = 1 - old_val
                adj[v][u] = 1 - old_val
                new_viol = get_violating_subsets(adj)
                if len(new_viol) < len(viol):
                    improved = True
                    break
                else:
                    adj[u][v] = old_val
                    adj[v][u] = old_val
            
            if not improved:
                # Try random flips to escape local optima
                for _ in range(5):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        adj[u][v] ^= 1
                        adj[v][u] ^= 1
                viol = get_violating_subsets(adj)
        
        return adj, False
    
    def _verify_no_large_regular(adj, n):
        """Verify no subset of size >= 6 induces a regular graph."""
        # Check sizes 6 to n
        # For efficiency, check size 6 first (already done), then 7, 8, ...
        for k in range(7, n+1):
            for subset in itertools.combinations(range(n), k):
                if _is_regular_subgraph(subset, adj, n):
                    return False
        return True
    
    def _find_large_regular(adj, n):
        """Find a subset of size >= 7 that induces a regular graph."""
        for k in range(7, n+1):
            for subset in itertools.combinations(range(n), k):
                if _is_regular_subgraph(subset, adj, n):
                    return subset
        return None
    
    # Try n=21 first (would be a breakthrough), then fall back to 20
    for n in [21, 20]:
        adj, success = build_and_optimize(n, max_time=50)
        if success:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a known small valid graph
    # Empty graph on 5 vertices is trivially valid
    return {'n': 5, 'edges': []}
