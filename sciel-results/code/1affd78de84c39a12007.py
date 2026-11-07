"""Packing trees T1..T20 into K20 using CSP backtracking with MRV."""
import time
from random import Random

def instance_trees(n=20, seed=611):
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
    start_time = time.time()
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    n = 20
    tree_edges = trees
    
    # Order trees largest first
    order = list(range(19, -1, -1))
    
    def try_packing(rng_seed):
        rng = Random(rng_seed)
        
        # available_edges: set of frozensets
        available = set()
        for i in range(n):
            for j in range(i + 1, n):
                available.add(frozenset((i, j)))
        
        assignment = [None] * 20
        
        def backtrack(tidx):
            if tidx == len(order):
                return True
            
            k = order[tidx]
            edges = tree_edges[k]
            tv_count = k + 1  # number of vertices in tree k
            
            if not edges:
                # T1: single vertex, no edges
                # Find any vertex (no constraint on vertices, only edges)
                # Just pick vertex 0
                assignment[k] = [0]
                if backtrack(tidx + 1):
                    return True
                assignment[k] = None
                return False
            
            # Build tree structure
            tree_adj = [[] for _ in range(tv_count)]
            for u, v in edges:
                tree_adj[u].append(v)
                tree_adj[v].append(u)
            
            # Find root and build tree
            root = 0
            parent = [-1] * tv_count
            children = [[] for _ in range(tv_count)]
            visited = [False] * tv_count
            stack = [root]
            visited[root] = True
            while stack:
                u = stack.pop()
                for v in tree_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        stack.append(v)
            
            # Backtracking assignment for this tree
            assign = [-1] * tv_count
            
            def embed(u):
                """Try to embed subtree rooted at u."""
                if not children[u]:
                    return True
                
                # Shuffle children for randomization
                child_list = list(children[u])
                rng.shuffle(child_list)
                
                for child in child_list:
                    # Try each available host for this child
                    # MRV: prefer hosts with fewer available edges for remaining
                    candidates = []
                    for h in range(n):
                        if h == assign[u]:
                            continue
                        e = frozenset((assign[u], h))
                        if e in available:
                            candidates.append(h)
                    
                    rng.shuffle(candidates)
                    
                    for h in candidates:
                        e = frozenset((assign[u], h))
                        assign[child] = h
                        available.remove(e)
                        if embed(child):
                            return True
                        available.add(e)
                        assign[child] = -1
                
                return False
            
            # Try each possible root host
            root_candidates = list(range(n))
            rng.shuffle(root_candidates)
            
            for rh in root_candidates:
                assign[root] = rh
                if embed(root):
                    assignment[k] = list(assign)
                    if backtrack(tidx + 1):
                        return True
                    # Undo
                    for u, v in edges:
                        available.add(frozenset((assign[u], assign[v])))
                    assignment[k] = None
                assign[root] = -1
            
            return False
        
        if backtrack(0):
            result = [None] * 20
            for k in range(20):
                if assignment[k] is not None:
                    result[k] = list(assignment[k])
            return result
        return None
    
    best_result = None
    best_edges = 0
    base_seed = 12345
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        seed = base_seed + attempts
        result = try_packing(seed)
        if result is not None:
            total_edges = sum(len(tree_edges[k]) for k in range(20) if result[k] is not None)
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = result
                if best_edges >= 190:
                    break
    
    if best_result is None:
        return [None] * 19 + [list(range(20))]
    
    return best_result
