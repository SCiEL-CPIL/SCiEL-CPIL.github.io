# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 145.0  # safety margin before 150s kill

def _gen_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    t0 = time.time()
    N = 20
    trees = _gen_trees(N, 611)
    
    # Precompute tree degrees and adjacency
    tree_info = []
    for k in range(1, N + 1):
        edges = trees[k - 1]
        deg = [0] * k
        adj = [set() for _ in range(k)]
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        tree_info.append((k, edges, deg, adj))
    
    # Host state
    used_edges = set()
    used_neighbors = [set() for _ in range(N)]
    
    result = [None] * N
    
    def avail_degree(v):
        return N - 1 - len(used_neighbors[v])
    
    def edge_free(u, v):
        if u > v:
            u, v = v, u
        return (u, v) not in used_edges
    
    def mark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.add((u, v))
        used_neighbors[u].add(v)
        used_neighbors[v].add(u)
    
    def unmark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.discard((u, v))
        used_neighbors[u].discard(v)
        used_neighbors[v].discard(u)
    
    def find_embedding(k_idx, rand_seed):
        """Find an embedding for tree k_idx using backtracking within the tree."""
        if time.time() - t0 > _DEADLINE:
            return None
        
        k = k_idx + 1
        edges, deg, adj = tree_info[k_idx][1], tree_info[k_idx][2], tree_info[k_idx][3]
        
        if k == 1:
            return [0]
        
        if k == 2:
            for u in range(N):
                for v in range(u + 1, N):
                    if edge_free(u, v):
                        return [u, v]
            return None

        # Order vertices by decreasing degree for better pruning
        v_order = sorted(range(k), key=lambda v: -deg[v])
        mapping = [None] * k
        assigned = [False] * k
        used_in_tree = set()
        
        def backtrack(pos):
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == k:
                return True
            
            tv = v_order[pos]
            d_needed = deg[tv]
            placed_neighbors = [nb for nb in adj[tv] if assigned[nb]]
            
            candidates = []
            for h in range(N):
                if h in used_in_tree:
                    continue
                if avail_degree(h) < d_needed:
                    continue
                ok = True
                for nb in placed_neighbors:
                    if not edge_free(h, mapping[nb]):
                        ok = False
                        break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort by most constrained first, with random tie-breaking
            rng = random.Random(rand_seed + pos * 7 + k_idx * 13)
            candidates.sort(key=lambda h: (avail_degree(h), rng.random()))
            
            for h in candidates:
                mapping[tv] = h
                assigned[tv] = True
                used_in_tree.add(h)
                
                for nb in placed_neighbors:
                    mark_edge(h, mapping[nb])
                
                if backtrack(pos + 1):
                    return True
                
                for nb in placed_neighbors:
                    unmark_edge(h, mapping[nb])
                
                used_in_tree.discard(h)
                mapping[tv] = None
                assigned[tv] = False
            
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    def solve(idx, rand_seed, depth):
        """Recursively embed trees from idx down to 0."""
        if time.time() - t0 > _DEADLINE:
            return False
        
        if idx < 0:
            return True
        
        emb = find_embedding(idx, rand_seed)
        if emb is None:
            return False
        
        # Apply embedding
        edges = tree_info[idx][1]
        for a, b in edges:
            mark_edge(emb[a], emb[b])
        
        result[idx] = emb
        
        if solve(idx - 1, rand_seed, depth + 1):
            return True
        
        # Unapply
        for a, b in edges:
            unmark_edge(emb[a], emb[b])
        result[idx] = None
        
        return False

    # Try multiple restarts with different random seeds
    # Embed largest trees first (index 19 down to 0)
    best_result = [None] * N
    best_edge_count = 0
    
    for restart in range(200):
        if time.time() - t0 > _DEADLINE:
            break
        
        # Reset state
        used_edges.clear()
        for i in range(N):
            used_neighbors[i].clear()
        result = [None] * N
        
        r_seed = restart * 7919 + 12345
        
        if solve(N - 1, r_seed, 0):
            # Full success - all 190 edges packed
            return result
        
        # Track best partial solution
        # Count edges in current result (before reset above cleared it, so use result which was unrolled)
        # Actually result is now all None after unrolling. We need to track during search.
        # Let's track it differently: count edges from the trees we managed to embed.
        # Since solve unrolls everything on failure, we lose the partial.
        # Alternative: run greedy to get a baseline, then try to improve.
        
        # For now, let's just keep trying restarts.
        pass
    
    # Fallback: Greedy sequential embedding
    # Reset state
    used_edges.clear()
    for i in range(N):
        used_neighbors[i].clear()
    result = [None] * N
    
    # Embed in order: largest first for better packing
    order = list(range(N - 1, -1, -1))
    
    for idx in order:
        if time.time() - t0 > _DEADLINE:
            break
        
        emb = find_embedding(idx, 42 + idx)
        if emb is not None:
            edges = tree_info[idx][1]
            for a, b in edges:
                mark_edge(emb[a], emb[b])
            result[idx] = emb
    
    # Convert to proper types
    final_result = []
    for i in range(N):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in result[i]])
    
    return final_result

# EVOLVE-BLOCK-END
