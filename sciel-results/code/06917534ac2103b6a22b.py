"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    
    # Precompute all K20 edges
    K20_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            K20_edges.add((u, v))
            
    def solve_tree_csp(tree_edges, k, used_edges, deadline, rng):
        """
        CSP-based solver for embedding a tree into K20.
        Uses degree-ordered variable assignment with forward checking.
        """
        if k == 1:
            return [0]
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    if (u, v) not in used_edges:
                        return [u, v]
            return None

        # Build tree adjacency and degree
        tree_adj = [[] for _ in range(k)]
        tree_degree = [0] * k
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            tree_degree[u] += 1
            tree_degree[v] += 1
        
        # Order vertices by degree (descending) for better constraint propagation
        # But we need to respect tree structure: assign a vertex only after its parent
        # Strategy: Root at highest-degree vertex, then do BFS with degree-ordered children
        
        # Find best root (highest degree)
        root = max(range(k), key=lambda x: tree_degree[x])
        
        # BFS from root, ordering children by degree (descending)
        parent = [-1] * k
        visited = [False] * k
        visited[root] = True
        order = [root]
        q = [root]
        while q:
            u = q.pop(0)
            # Sort neighbors by degree descending
            neighbors = sorted([v for v in tree_adj[u] if not visited[v]], 
                               key=lambda x: -tree_degree[x])
            for v in neighbors:
                visited[v] = True
                parent[v] = u
                order.append(v)
                q.append(v)
        
        # Map tree vertex to position in order
        order_pos = {v: i for i, v in enumerate(order)}
        
        host = [None] * k
        used_hosts = set()
        
        # Precompute for each tree vertex, which tree vertices are its children
        children = [[] for _ in range(k)]
        for v in range(k):
            if parent[v] != -1:
                children[parent[v]].append(v)
        
        def get_candidates(v, host_map, used_hosts_set):
            """Get valid host candidates for tree vertex v given current assignment."""
            p = parent[v]
            hp = host_map[p] if p != -1 else None
            
            candidates = []
            for h in range(20):
                if h in used_hosts_set:
                    continue
                if hp is not None:
                    edge = (min(hp, h), max(hp, h))
                    if edge in used_edges:
                        continue
                candidates.append(h)
            return candidates
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                return True
            
            v = order[pos]
            
            # Get candidates
            candidates = get_candidates(v, host, used_hosts)
            if not candidates:
                return False
            
            # Order candidates: prefer those that leave more options for children
            # Score each candidate by how many valid hosts remain for its children
            def score_candidate(h):
                if not children[v]:
                    return 0
                # Count how many children have at least one valid host with h
                count = 0
                for c in children[v]:
                    for hc in range(20):
                        if hc in used_hosts or hc == h:
                            continue
                        edge = (min(h, hc), max(h, hc))
                        if edge not in used_edges:
                            count += 1
                            break
                return count
            
            # Sort by score descending, then randomize ties
            scored = [(score_candidate(h), h) for h in candidates]
            scored.sort(key=lambda x: -x[0])
            # Add randomness to break ties
            for i in range(len(scored) - 1, 0, -1):
                j = rng.randint(0, i)
                scored[i], scored[j] = scored[j], scored[i]
            
            for _, h in scored:
                host[v] = h
                used_hosts.add(h)
                
                if backtrack(pos + 1):
                    return True
                
                host[v] = None
                used_hosts.remove(h)
            
            return False
        
        if backtrack(0):
            return host
        return None

    # Strategy:
    # 1. T1 is trivial: []
    # 2. Embed trees from largest to smallest (T20, T19, ..., T2)
    #    This ensures the most constrained trees get first pick of edges.
    
    best_score = 0
    best_embeddings = [None] * 20
    
    # T1 always maps to a single vertex
    best_embeddings[0] = [0]
    
    # Try multiple restarts with different random seeds
    for restart in range(100):
        if time.time() > deadline - 2:
            break
            
        rng = Random(1234 + restart * 7 + 1)
        used_edges = set()
        embeddings = [None] * 20
        embeddings[0] = [0]  # T1
        
        success = True
        
        # Embed T20 down to T2
        for k in range(20, 1, -1):
            if time.time() > deadline - 2:
                success = False
                break
            
            res = solve_tree_csp(trees[k-1], k, used_edges, deadline, rng)
            if res is None:
                success = False
                break
            embeddings[k-1] = res
            for u, v in trees[k-1]:
                used_edges.add((min(res[u], res[v]), max(res[u], res[v])))
                
        if success:
            # Calculate score
            score = 0
            for k in range(1, 21):
                if embeddings[k-1] is not None:
                    score += k - 1
            if score > best_score:
                best_score = score
                best_embeddings = embeddings[:]
                
            if best_score == 190:
                break  # Perfect packing

    # Fallback: If T20 wasn't embedded, try identity mapping
    if best_embeddings[19] is None:
        # Check if identity mapping is edge-disjoint with any existing embeddings
        used_edges = set()
        for k in range(2, 20):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        identity_valid = True
        for u, v in trees[19]:
            if (min(u, v), max(u, v)) in used_edges:
                identity_valid = False
                break
        
        if identity_valid:
            best_embeddings[19] = list(range(20))
            best_score += 19
        else:
            # Try to find any valid embedding for T20
            rng = Random(9999)
            used_edges_fb = set()
            for k in range(2, 20):
                if best_embeddings[k-1] is not None:
                    for u, v in trees[k-1]:
                        used_edges_fb.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                          max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
            res20 = solve_tree_csp(trees[19], 20, used_edges_fb, deadline, rng)
            if res20 is not None:
                best_embeddings[19] = res20
                best_score += 19
    
    # Try to embed T2 if not already done
    if best_embeddings[1] is None:
        # Collect used edges from any existing embeddings
        used_edges = set()
        for k in range(2, 21):
            if best_embeddings[k-1] is not None:
                for u, v in trees[k-1]:
                    used_edges.add((min(best_embeddings[k-1][u], best_embeddings[k-1][v]), 
                                    max(best_embeddings[k-1][u], best_embeddings[k-1][v])))
        
        for u in range(20):
            for v in range(u+1, 20):
                if (u, v) not in used_edges:
                    best_embeddings[1] = [u, v]
                    best_score += 1
                    break
            if best_embeddings[1] is not None:
                break

    # Final validation and cleanup
    if best_embeddings[0] is None:
        best_embeddings[0] = [0]
        
    for i in range(20):
        if best_embeddings[i] is not None:
            if len(best_embeddings[i]) != i + 1:
                best_embeddings[i] = None
            else:
                if len(set(best_embeddings[i])) != i + 1:
                    best_embeddings[i] = None

    return best_embeddings

# EVOLVE-BLOCK-END
