import time
from collections import deque
from random import Random
import sys

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

def _embed(tree_edges, k, free, rng, node_limit=5000000):
    """Find an injective host map for tree (vertices 0..k-1) using only free edges."""
    if k == 1:
        return [0]
    
    # Build adjacency list for the tree
    adj = [[] for _ in range(k)]
    degree = [0] * k
    for (u, v) in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        
    # Precompute all possible host edges for quick lookup
    # free is a set of frozensets
    host_adj = [set() for _ in range(20)]
    for e in free:
        u, v = tuple(e)
        host_adj[u].add(v)
        host_adj[v].add(u)
        
    host = [-1] * k
    used = [False] * 20
    counter = [0]
    
    # Heuristic: Start with the vertex with highest degree in the tree
    # This reduces the branching factor early on
    start_v = max(range(k), key=lambda x: degree[x])
    
    def get_candidates(v, host_v):
        """Get valid host vertices for tree vertex v given host_v is mapped to host_v? No, host_v is the candidate."""
        pass

    def dfs(mapped_count):
        if counter[0] > node_limit:
            return False
        if mapped_count == k:
            return True
            
        # Select next tree vertex to map
        # MRV: Find unmapped vertex with fewest valid host candidates
        # Valid host h for tree vertex v:
        # 1. h not in used
        # 2. If v has mapped neighbors, h must be connected to their hosts in 'free'
        
        best_v = -1
        best_cands = None
        min_cand_count = 21 # More than 20
        
        for v in range(k):
            if host[v] != -1:
                continue
                
            cands = []
            for h in range(20):
                if used[h]:
                    continue
                # Check constraints from mapped neighbors
                valid = True
                for nb in adj[v]:
                    if host[nb] != -1:
                        if h not in host_adj[host[nb]]:
                            valid = False
                            break
                if valid:
                    cands.append(h)
            
            if len(cands) < min_cand_count:
                min_cand_count = len(cands)
                best_v = v
                best_cands = cands
                if min_cand_count == 0:
                    return False
                if min_cand_count == 1:
                    break # Can't get better than 1
            
        if best_v == -1:
            # No unmapped vertices? Should not happen if mapped_count < k
            # Or all unmapped vertices have no constraints?
            # If k > 1, there are always constraints or we just pick any
            # Actually, if mapped_count < k, there is an unmapped vertex.
            # If the loop didn't find one, it means something is wrong.
            # Let's just pick the first unmapped one if best_v is -1
            for v in range(k):
                if host[v] == -1:
                    best_v = v
                    best_cands = [h for h in range(20) if not used[h]]
                    break
        
        if best_cands is None:
            return False
            
        # Shuffle candidates to break symmetry / randomize
        rng.shuffle(best_cands)
        
        for h in best_cands:
            counter[0] += 1
            if counter[0] > node_limit:
                return False
                
            host[best_v] = h
            used[h] = True
            
            if dfs(mapped_count + 1):
                return True
                
            host[best_v] = -1
            used[h] = False
            
        return False

    # Initialize start
    start_cands = [h for h in range(20)]
    rng.shuffle(start_cands)
    
    for h in start_cands:
        counter[0] = 0
        host[start_v] = h
        used[h] = True
        if dfs(1):
            return [int(x) for x in host]
        host[start_v] = -1
        used[h] = False
        
    return None

def construct():
    trees = instance_trees(20, 611)
    all_edges = set()
    for a in range(20):
        for b in range(a + 1, 20):
            all_edges.add(frozenset((a, b)))
    best = None
    best_val = -1
    start = time.time()
    deadline = start + 110.0
    attempt = 0
    
    while time.time() < deadline:
        attempt += 1
        rng = Random(attempt * 7919 + 13)
        free = set(all_edges)
        result = [None] * 20
        # largest (most constrained) trees first
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: -(i + 1))
        val = 0
        success = True
        for i in order_idx:
            if time.time() > deadline:
                break
            k = i + 1
            te = trees[i]
            if k == 1:
                result[i] = [0]
                continue
            emb = _embed(te, k, free, rng)
            if emb is not None and len(set(emb)) == k:
                ok = True
                es = []
                for (u, v) in te:
                    e = frozenset((emb[u], emb[v]))
                    if e not in free:
                        ok = False
                        break
                    es.append(e)
                if ok:
                    for e in es:
                        free.remove(e)
                    result[i] = [int(x) for x in emb]
                    val += k - 1
                else:
                    success = False
            else:
                success = False
        
        if val > best_val:
            best_val = val
            best = result
            
        if best_val >= 190:
            break
            
    if best is None:
        best = [None] * 19 + [list(range(20))]
        
    out = []
    for e in best:
        if e is None:
            out.append(None)
        else:
            out.append([int(x) for x in e])
    return out
