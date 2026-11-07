"""Rewritten program: embed trees T1..T20 into K20 edge-disjointly via backtracking."""
import time
import random
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
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    trees = instance_trees()
    n = 20
    # Total edges in K20 is 190. Sum of (k-1) for k=1..20 is 190.
    # So we need a perfect packing: every edge of K20 used exactly once.
    
    # Strategy: use a randomized greedy/backtracking approach.
    # Order trees by size descending (harder to place larger trees).
    # For each tree, try to find an embedding using DFS/backtracking.
    
    # Precompute all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    used_edges = set()
    
    # Order: place larger trees first
    tree_order = list(range(n))  # index 0 is T1 (k=1), index 19 is T20 (k=20)
    # Sort by number of edges descending, so largest trees first
    tree_order.sort(key=lambda idx: len(trees[idx]), reverse=True)
    
    # Result: entry k-1 is None or list of k distinct ints
    result = [None] * n
    
    # For each tree in order, try to embed it
    def find_embedding(tree_edges, num_vertices):
        """Find an embedding of tree_edges into K20 avoiding used_edges."""
        # tree_edges is a list of (u, v) pairs in the abstract tree
        # We need to assign each abstract vertex a host vertex in 0..19
        # such that no two abstract vertices map to same host vertex
        # and no edge (a,b) in tree maps to an edge in used_edges
        
        if not tree_edges:
            # Single vertex tree: any unused vertex
            for v in range(num_vertices):
                return [v]
            return None
        
        # Use backtracking: assign abstract vertices to host vertices
        # Order abstract vertices by degree in the tree (higher degree first)
        adj = {}
        for u, v in tree_edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort vertices by degree descending
        verts = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
        
        # Try to assign using backtracking
        assignment = {}  # abstract vertex -> host vertex
        host_used = set()
        
        def backtrack(idx):
            if idx == len(verts):
                # Check all edges are valid
                for u, v in tree_edges:
                    hu, hv = assignment[u], assignment[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        return False
                return True
            
            v = verts[idx]
            # Try all possible host vertices
            # Prune: if v has already-assigned neighbors, host must be adjacent to their hosts
            # and the edge must not be in used_edges
            candidates = list(range(num_vertices))
            random.shuffle(candidates)
            
            for h in candidates:
                if h in host_used:
                    continue
                # Check against already-assigned neighbors
                valid = True
                for nb in adj[v]:
                    if nb in assignment:
                        hn = assignment[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                assignment[v] = h
                host_used.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                del assignment[v]
                host_used.remove(h)
            
            return False
        
        if backtrack(0):
            # Convert to list in order of abstract vertices 0..k-1
            return [assignment[v] for v in range(len(verts))]
        return None
    
    # Try multiple random restarts
    best_result = None
    best_count = -1
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        used_edges = set()
        result = [None] * n
        success = True
        
        for idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            
            k = idx + 1
            tree_edges = trees[idx]
            
            if not tree_edges:
                # T1: single vertex, any vertex
                result[idx] = [0]
                continue
            
            emb = find_embedding(tree_edges, n)
            if emb is None:
                success = False
                break
            
            result[idx] = emb
            # Mark edges as used
            for u, v in tree_edges:
                hu, hv = emb[u], emb[v]
                edge = (min(hu, hv), max(hu, hv))
                if edge in used_edges:
                    success = False
                    break
                used_edges.add(edge)
            if not success:
                break
        
        if success:
            count = len(used_edges)
            if count > best_count:
                best_count = count
                best_result = result[:]
            if best_count == 190:
                # Perfect packing
                break
    
    if best_result is not None:
        # Ensure types are correct: list of ints or None
        final = []
        for entry in best_result:
            if entry is None:
                final.append(None)
            else:
                final.append([int(x) for x in entry])
        return final
    else:
        # Fallback: just return the T20 identity embedding (score 1.0)
        return [None] * 19 + [list(range(20))]
