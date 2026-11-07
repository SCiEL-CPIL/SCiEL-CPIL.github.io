"""Improved packing of trees T1..T20 into K20."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

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

def _norm_edge(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Track used edges in K20 (normalized)
    used_edges = set()
    result = [None] * 20
    
    # Process trees from largest to smallest (most constrained first)
    # T20 has 19 edges, T19 has 18, etc.
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex, no edges. Pick any vertex.
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    e = _norm_edge(u, w)
                    if e not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add(e)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use backtracking with forward checking
        tree_n = k
        tree_edge_list = tree_edges
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Get BFS order of tree vertices starting from 0
        visited = [False] * tree_n
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)
        
        # For each tree vertex, precompute which tree neighbors are "before" it in BFS order
        # (i.e., already assigned when we assign this vertex)
        pos_in_bfs = [0] * tree_n
        for i, v in enumerate(bfs_order):
            pos_in_bfs[v] = i
        
        # assignment[tree_v] = host_v or None
        assignment = [None] * tree_n
        # Track which host vertices are used
        host_used = [False] * 20
        # Temporarily used edges during backtracking
        temp_used_edges = set()
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = bfs_order[pos]
            # Try each unused host vertex
            for host_v in range(20):
                if host_used[host_v]:
                    continue
                
                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in tree_adj[tree_v]:
                    if pos_in_bfs[nb] < pos:  # neighbor already assigned
                        e = _norm_edge(assignment[nb], host_v)
                        if e in used_edges or e in temp_used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Tentatively assign
                assignment[tree_v] = host_v
                host_used[host_v] = True
                # Mark edges as temporarily used
                temp_edges = set()
                for nb in tree_adj[tree_v]:
                    if pos_in_bfs[nb] < pos:
                        e = _norm_edge(assignment[nb], host_v)
                        temp_edges.add(e)
                
                if backtrack(pos + 1):
                    # Commit the edges
                    for e in temp_edges:
                        used_edges.add(e)
                    return True
                
                # Undo
                for e in temp_edges:
                    temp_used_edges.discard(e)
                host_used[host_v] = False
                assignment[tree_v] = None
        
        # Add temp edges to tracking
        # We need to properly manage temp_used_edges
        # Let me restructure: use a separate set that we add/remove
        
        # Actually, let me redo this with proper temp edge tracking
        temp_used_edges = set()
        
        def backtrack2(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = bfs_order[pos]
            # Try each unused host vertex
            for host_v in range(20):
                if host_used[host_v]:
                    continue
                
                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in tree_adj[tree_v]:
                    if pos_in_bfs[nb] < pos:  # neighbor already assigned
                        e = _norm_edge(assignment[nb], host_v)
                        if e in used_edges or e in temp_used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Tentatively assign
                assignment[tree_v] = host_v
                host_used[host_v] = True
                # Mark edges as temporarily used
                added_edges = []
                for nb in tree_adj[tree_v]:
                    if pos_in_bfs[nb] < pos:
                        e = _norm_edge(assignment[nb], host_v)
                        temp_used_edges.add(e)
                        added_edges.append(e)
                
                if backtrack2(pos + 1):
                    # Commit the edges
                    for e in added_edges:
                        used_edges.add(e)
                    return True
                
                # Undo
                for e in added_edges:
                    temp_used_edges.discard(e)
                host_used[host_v] = False
                assignment[tree_v] = None
        
        if backtrack2(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
