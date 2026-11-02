import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    seed_n, seed_edges = seed_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def greedy_cage_construct(k, g):
        """Construct a k-regular graph with girth >= g using greedy BFS tree completion."""
        r = (g - 1) // 2  # radius of BFS tree
        # Build BFS tree
        # Level 0: 1 vertex
        # Level i: k*(k-1)^(i-1) vertices for i >= 1
        levels = [1]
        for i in range(1, r + 1):
            levels.append(k * (k-1)**(i-1))
        n_tree = sum(levels)
        
        if n_tree > 5000:  # Safety
            return None
        
        # Assign vertex indices
        # Level 0: vertex 0
        # Level 1: vertices 1 to k
        # Level i: vertices from cumulative sum
        vertex_level = [0] * n_tree
        vertex_parent = [-1] * n_tree
        vertex_branch = [0] * n_tree  # level-1 ancestor
        
        idx = 0
        # Level 0
        vertex_level[0] = 0
        vertex_branch[0] = -1
        idx = 1
        # Level 1
        for i in range(k):
            vertex_level[idx] = 1
            vertex_parent[idx] = 0
            vertex_branch[idx] = idx
            idx += 1
        # Levels 2 to r
        for lvl in range(2, r + 1):
            count = k * (k-1)**(lvl-1)
            for i in range(count):
                vertex_level[idx] = lvl
                # Parent is at level lvl-1
                parent_idx = 1 + (i // (k-1))  # level 1 starts at index 1
                # Actually, let me be more careful
                # Level 1 vertices: indices 1 to k
                # Level 2: each level-1 vertex has (k-1) children
                # Level 3: each level-2 vertex has (k-1) children
                # etc.
                # For level lvl, vertex i has parent at level lvl-1, index = 1 + (i // (k-1)^(lvl-2))
                # This is getting complex. Let me use a different indexing.
                pass
            idx += count
        
        # This is getting too complex with the indexing. Let me use a simpler approach.
        # Just build the tree explicitly.
        tree_edges = []
        vertex_data = {}  # vertex_id -> (level, parent, branch)
        
        vid = 0
        root = vid
        vertex_data[vid] = (0, -1, -1)
        vid += 1
        
        # Level 1
        level1_vertices = []
        for i in range(k):
            v = vid
            vertex_data[v] = (1, root, v)
            level1_vertices.append(v)
            tree_edges.append((root, v))
            vid += 1
        
        # Levels 2 to r
        current_level = level1_vertices
        for lvl in range(2, r + 1):
            next_level = []
            for parent in current_level:
                branch = vertex_data[parent][2]
                for i in range(k - 1):
                    v = vid
                    vertex_data[v] = (lvl, parent, branch)
                    next_level.append(v)
                    tree_edges.append((parent, v))
                    vid += 1
            current_level = next_level
        
        n_tree = vid
        if n_tree != moore_bound:
            # Should be equal, but just in case
            pass
        
        # Now we need to add (n_tree * k - 2 * len(tree_edges)) / 2 more edges
        # = (n_tree * k - 2 * (n_tree - 1)) / 2 = n_tree * (k-1) / 2 + 1/2... 
        # Actually: tree has n_tree - 1 edges. We need n_tree * k / 2 edges total.
        # Additional edges: n_tree * k / 2 - (n_tree - 1) = (n_tree * k - 2*n_tree + 2) / 2
        
        # Free stubs: vertex v has k - (number of tree edges incident to v) free stubs
        # Root: k free stubs (all k edges go to level 1, so 0 free... wait)
        # Root has k edges to level 1, so 0 free stubs.
        # Level 1: each has 1 edge to root and (k-1) edges to level 2, so 0 free stubs.
        # ...
        # Level r: each has 1 edge to level r-1, so (k-1) free stubs.
        
        # Total free stubs = (k-1) * (number of level r vertices) = (k-1) * k * (k-1)^(r-1) = k * (k-1)^r
        # For k=5, r=4: 5 * 4^4 = 5 * 256 = 1280. Matches earlier calculation.
        
        # The constraint: two vertices u, v can be connected if their tree distance >= g - 1 = 8
        # (since adding edge (u,v) creates a cycle of length dist(u,v) + 1, which must be >= g)
        
        # Tree distance: for two vertices at levels lu, lv with LCA at level lca:
        # dist = (lu - lca) + (lv - lca) = lu + lv - 2*lca
        
        # For two level-r vertices: dist = 2*r - 2*lca = 2*(r - lca)
        # We need 2*(r - lca) >= g - 1 = 2*r (since g = 2*r + 1)
        # So r - lca >= r, i.e., lca = 0.
        # This means the LCA must be the root, i.e., they must be in different level-1 branches.
        
        # So level-r vertices can only connect to level-r vertices in a different level-1 branch.
        
        # Group level-r vertices by branch
        branch_groups = {}
        for v, (lvl, parent, branch) in vertex_data.items():
            if lvl == r:
                if branch not in branch_groups:
                    branch_groups[branch] = []
                branch_groups[branch].append(v)
        
        # Each level-r vertex has (k-1) free stubs
        # We need to pair up all free stubs such that paired vertices are in different branches
        
        # Total free stubs: (k-1) * sum(len(group) for group in branch_groups.values())
        # = (k-1) * k * (k-1)^(r-1) = k * (k-1)^r
        
        # For k=5, r=4: 5 * 256 = 1280 stubs, 640 edges.
        
        # Each branch has (k-1)^(r-1) = 64 vertices, each with 4 stubs = 256 stubs.
        # 5 branches, 256 stubs each, total 1280.
        
        # We need to create a perfect matching of stubs where no two stubs in the same edge are from the same branch.
        
        # Simple approach: randomly pair stubs from different branches.
        all_stubs = []
        for branch, vertices in branch_groups.items():
            for v in vertices:
                for s in range(k - 1):
                    all_stubs.append((v, branch))
        
        random.shuffle(all_stubs)
        
        # Greedy matching
        used = [False] * len(all_stubs)
        new_edges = []
        
        for i in range(len(all_stubs)):
            if used[i]:
                continue
            v1, b1 = all_stubs[i]
            found = False
            for j in range(i + 1, len(all_stubs)):
                if used[j]:
                    continue
                v2, b2 = all_stubs[j]
                if b1 == b2:
                    continue
                # Check tree distance
                # v1 and v2 are both at level r, in different branches (different level-1 ancestors)
                # So their LCA is the root (level 0), and dist = 2*r
                # 2*r = 8 >= g-1 = 8. OK!
                new_edges.append((v1, v2))
                used[i] = True
                used[j] = True
                found = True
                break
            if not found:
                return None  # Failed to match
        
        if len(new_edges) != len(all_stubs) // 2:
            return None
        
        all_edges = tree_edges + new_edges
        return n_tree, all_edges
    
    # Try greedy construction
    result = greedy_cage_construct(k, g)
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        # Verify
        if exact_girth_check(n, adj, g):
            return n, edges
    
    # Fallback to seed
    return seed_n, seed_edges
