"""Packing trees T1..T20 into K20 - CSP with MRV and backtracking."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140.0
    trees = instance_trees(20, 611)
    
    # Precompute adjacency for K20
    K20_adj = [set() for _ in range(20)]
    for u in range(20):
        for v in range(u + 1, 20):
            K20_adj[u].add(v)
            K20_adj[v].add(u)
    
    def solve_tree_csp(tree_edges, k, used_edges, deadline, rng):
        """
        CSP-based solver for embedding a tree with k vertices into K20 avoiding used_edges.
        Uses MRV (Minimum Remaining Values) heuristic with forward checking.
        Returns a list of k distinct host vertices, or None.
        """
        if k == 1:
            return [0]
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None
        
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Determine embedding order using a root-based approach
        # Root the tree at vertex 0, then order by BFS
        visited = [False] * k
        order = [0]
        visited[0] = True
        parent = [-1] * k
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        
        # host[v] = which host vertex tree vertex v maps to
        host = [None] * k
        used_hosts = set()
        
        # Precompute for each tree vertex, which other tree vertices are its neighbors
        # that would need to be assigned before/after it
        # For MRV: when assigning vertex v, its candidates are host vertices h such that:
        #   - h not in used_hosts
        #   - for each already-assigned neighbor u of v in the tree: (host[u], h) not in used_edges
        
        def get_candidates(v):
            """Get candidate host vertices for tree vertex v."""
            cands = []
            for h in range(20):
                if h in used_hosts:
                    continue
                # Check against all already-assigned neighbors
                valid = True
                for u in adj[v]:
                    if host[u] is not None:
                        edge = (min(host[u], h), max(host[u], h))
                        if edge in used_edges:
                            valid = False
                            break
                if valid:
                    cands.append(h)
            return cands
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            v = order[idx]
            cands = get_candidates(v)
            if not cands:
                return False
            
            # MRV: pick the next unassigned vertex with fewest candidates
            # Actually, we're following the BFS order, but we can use MRV to reorder
            # Let's try: among all unassigned vertices, pick the one with fewest candidates
            
            # For efficiency, let's just use the current order but shuffle candidates
            rng.shuffle(cands)
            
            for h in cands:
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
            
            return False
        
        # Try with the BFS order
        if backtrack(0):
            return host
        
        return None

    def solve_tree_mrv(tree_edges, k, used_edges, deadline, rng):
        """
        CSP solver with true MRV: always pick the unassigned vertex with fewest candidates.
        """
        if k == 1:
            return [0]
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None
        
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        host = [None] * k
        used_hosts = set()
        assigned = [False] * k
        
        def get_candidates(v):
            cands = []
            for h in range(20):
                if h in used_hosts:
                    continue
                valid = True
                for u in adj[v]:
                    if host[u] is not None:
                        edge = (min(host[u], h), max(host[u], h))
                        if edge in used_edges:
                            valid = False
                            break
                if valid:
                    cands.append(h)
            return cands
        
        def backtrack(num_assigned):
            if time.time() > deadline:
                return False
            if num_assigned == k:
                return True
            
            # MRV: find unassigned vertex with fewest candidates
            best_v = -1
            best_cands = None
            best_count = 21  # max is 20
            
            for v in range(k):
                if assigned[v]:
                    continue
                cands = get_candidates(v)
                count = len(cands)
                if count == 0:
                    return False  # Dead end
                if count < best_count:
                    best_count = count
                    best_v = v
                    best_cands = cands
                    if count == 1:
                        break
            
            if best_v == -1:
                return True  # All assigned
            
            rng.shuffle(best_cands)
            
            for h in best_cands:
                host[best_v] = h
                used_hosts.add(h)
                assigned[best_v] = True
                
                if backtrack(num_assigned + 1):
                    return True
                
                host[best_v] = None
                used_hosts.remove(h)
                assigned[best_v] = False
            
            return False
        
        if backtrack(0):
            return host
        return None

    best_score = 0
    best_embeddings = [None] * 20
    
    # T1 is trivial
    best_embeddings[0] = [0]
    
    # Try multiple strategies
    strategies = [
        "desc",   # T20 first, then T19, ..., T2
        "asc",    # T2 first, then T3, ..., T20
        "alt",    # Alternate
    ]
    
    for restart in range(80):
        if time.time() > deadline - 3:
            break
        
        rng = Random(1000 + restart * 7)
        used_edges = set()
        embeddings = [None] * 20
        embeddings[0] = [0]
        
        # Determine order
        strat = strategies[restart % len(strategies)]
        if strat == "desc":
            order = list(range(19, 1, -1))  # T20 down to T2
        elif strat == "asc":
            order = list(range(1, 20))  # T2 up to T20
        else:  # alt
            order = []
            # Interleave: T20, T2, T19, T3, ...
            lo, hi = 1, 19
            while lo <= hi:
                if hi >= lo:
                    order.append(hi)
                    hi -= 1
                if lo <= hi:
                    order.append(lo)
                    lo += 1
        
        success = True
        for k_idx in order:
            k = k_idx + 1  # tree index (1-based), k vertices
            if time.time() > deadline - 3:
                success = False
                break
            
            # Choose solver based on size
            if k >= 15:
                res = solve_tree_mrv(trees[k_idx], k, used_edges, deadline, rng)
            else:
                res = solve_tree_csp(trees[k_idx], k, used_edges, deadline, rng)
            
            if res is None:
                success = False
                break
            
            embeddings[k_idx] = res
            for u, v in trees[k_idx]:
                hu, hv = res[u], res[v]
                used_edges.add((min(hu, hv), max(hu, hv)))
        
        if success:
            score = sum(k for k in range(1, 21) if embeddings[k-1] is not None) - 20  # k-1 edges each
            # Actually score = sum of (k-1) for each embedded tree
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
                
            if best_score == 190:
                break
    
    # Fallback: ensure T1 is set
    if best_embeddings[0] is None:
        best_embeddings[0] = [0]
    
    # Validate and clean up
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None
            elif len(set(best_embeddings[i])) != i + 1:
                best_embeddings[i] = None
            else:
                # Ensure all are Python ints
                for j in range(len(best_embeddings[i])):
                    best_embeddings[i][j] = int(best_embeddings[i][j])
    
    return best_embeddings
