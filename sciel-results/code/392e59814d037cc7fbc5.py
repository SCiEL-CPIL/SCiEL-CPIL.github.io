import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

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

    def check_girth(adj, n, g):
        """Verify no cycle shorter than g via non-backtracking walks from every vertex."""
        half = g // 2
        for s in range(n):
            # state: (node, prev) -> depth
            seen = {}
            dq = deque()
            for nb in adj[s]:
                key = nb * n + s
                seen[key] = 1
                dq.append((nb, s, 1))
            while dq:
                u, pu, dist = dq.popleft()
                if dist >= half:
                    continue
                nd = dist + 1
                for v in adj[u]:
                    if v == pu:
                        continue
                    if v == s:
                        if nd < g:
                            return False
                    else:
                        key = v * n + u
                        if key in seen:
                            if nd < g:
                                return False
                        else:
                            seen[key] = nd
                            dq.append((v, u, nd))
        return True

    def would_create_short_cycle(adj, u, v, g, max_dist):
        """Check if adding undirected edge (u,v) creates a cycle shorter than g.
        A cycle through (u,v) has length = dist(u,v in adj) + 1.
        So we need dist(u,v) >= g-1, i.e., no path of length < g-1 between u and v.
        We do BFS from u up to depth max_dist = g-2, avoiding going back through v
        (since v is not yet connected to u in adj)."""
        if u == v:
            return True
        if v in adj[u]:
            return True
        # BFS from u, depth limit = g-2 (a path of length g-1 + edge = cycle of length g, OK)
        # We need to detect any path of length <= g-2 from u to v
        limit = g - 2
        visited = {u: 0}
        dq = deque([u])
        while dq:
            node = dq.popleft()
            d = visited[node]
            if d >= limit:
                continue
            for nb in adj[node]:
                if nb == v:
                    return True  # path of length d+1 <= limit = g-2, cycle length <= g-1 < g
                if nb not in visited:
                    visited[nb] = d + 1
                    dq.append(nb)
        return False

    def build_tree_3reg(g, seed):
        """Build a 3-regular graph by growing a tree of depth (g-2)//2 from a root,
        then adding edges among leaves to complete 3-regularity.
        Returns (n, edges) or None."""
        rng = random.Random(seed)
        # For girth g, a tree of depth h from root has no cycles.
        # If we make the tree such that every vertex has degree 3 within the tree
        # (except leaves which have degree 1 in tree), then we need to add
        # 2 edges per leaf to make it 3-regular.
        # But that's complex. Simpler: build a tree where internal nodes have
        # 3 children (degree 4 in tree including parent, except root with degree 3).
        # Actually for 3-regular: each node has 3 neighbors.
        # 
        # Alternative approach: build a "spider" - a path of length L, then
        # attach pairs of leaves. No, that doesn't give 3-regular.
        #
        # Better: Use a binary-tree-like structure but with degree 3.
        # Start with root. Root has 3 neighbors. Each non-leaf has 2 children
        # (since one edge goes to parent). Leaves have 1 edge (to parent).
        # To make 3-regular, each leaf needs 2 more edges.
        # We pair leaves and add edges between pairs. But we need even number of leaves
        # and the pairing must not create short cycles.
        #
        # For girth g: the tree must be deep enough that any two leaves have
        # tree-distance >= g-1 (so that an edge between them creates a cycle
        # of length >= g). Actually, if two leaves have tree-distance D, then
        # adding an edge creates a cycle of length D+1. We need D+1 >= g, so D >= g-1.
        #
        # In a tree of depth h (root at depth 0, leaves at depth h), the distance
        # between two leaves is at most 2h (going up to root and back down).
        # So we need 2h >= g-1, i.e., h >= (g-1)/2.
        # For g=16: h >= 7.5, so h >= 8.
        #
        # Number of leaves in a 3-regular tree of depth h:
        # Level 0: 1 node (root, degree 3 in tree)
        # Level 1: 3 nodes
        # Level 2: 3*2 = 6 nodes
        # Level i (i>=1): 3*2^(i-1) nodes
        # Level h: 3*2^(h-1) leaves
        # Total nodes: 1 + 3 + 6 + ... + 3*2^(h-1) = 1 + 3*(2^h - 1) = 3*2^h - 2
        # For h=8: 3*256 - 2 = 766 nodes. 3*2^7 = 384 leaves.
        # We need to pair 384 leaves (192 edges). Total edges = tree_edges + 192.
        # Tree edges = total_nodes - 1 = 765. Total edges = 765 + 192 = 957.
        # Check: 3*n/2 = 3*766/2 = 1149. That's wrong.
        #
        # Wait, I'm confusing myself. Let me redo.
        # In a 3-regular tree (each internal node has 3 neighbors in the tree):
        # Root: 3 children
        # Each non-root internal node: 2 children + 1 parent = 3 neighbors
        # Leaves: 1 parent = 1 neighbor in tree
        #
        # Number of nodes at level i:
        # Level 0: 1
        # Level 1: 3
        # Level i (i>=1): 3 * 2^(i-1)
        #
        # Total nodes up to level h: 1 + sum_{i=1}^{h} 3*2^(i-1) = 1 + 3*(2^h - 1) = 3*2^h - 2
        # For h=8: 3*256 - 2 = 766
        # Leaves (level h): 3*2^(h-1) = 3*128 = 384
        # Internal nodes: 766 - 384 = 382
        #
        # Edges in tree: 766 - 1 = 765
        # Sum of degrees in tree: 2*765 = 1530
        # Internal nodes contribute: 382 * 3 = 1146
        # Leaves contribute: 384 * 1 = 384
        # Total: 1146 + 384 = 1530. Checks out.
        #
        # To make 3-regular: each leaf needs 2 more edges. Total new edges = 384*2/2 = 384.
        # We need to add 384 edges among the 384 leaves, each leaf getting exactly 2.
        # This is a 2-regular graph on 384 vertices = a union of cycles.
        # We need to ensure no cycle in the full graph is shorter than g.
        #
        # The distance between two leaves in the tree: if they share a parent, distance = 2.
        # If their LCA is at depth d, distance = 2*(h-d).
        # For g=16, we need all leaf-pairs that get an edge to have tree-distance >= 15.
        # 2*(h-d) >= 15 => h-d >= 7.5 => d <= h-8 = 0. So LCA must be root!
        # That means we can only connect leaves from different subtrees of the root.
        # The root has 3 subtrees, each with 384/3 = 128 leaves.
        # We can connect leaves from different subtrees (distance = 2*h = 16 >= 15). OK!
        # But leaves within the same subtree of root have LCA at depth >= 1,
        # so distance <= 2*(h-1) = 14 < 15. Can't connect those.
        #
        # So we need a 2-regular graph on 384 vertices where edges only go
        # between different root-subtrees. Each subtree has 128 leaves.
        # Each leaf needs degree 2, and all neighbors must be in other subtrees.
        # This is like a 3-partite 2-regular graph. Possible!
        # E.g., cycle through all 384 leaves, ensuring consecutive leaves are in different parts.
        #
        # But wait: even if tree-distance >= 15, the cycle formed might be shorter
        # if there's another path. No - in a tree, there's exactly one path between
        # two nodes. So the cycle length = tree_distance + 1. If tree_distance >= g-1,
        # cycle length >= g. Good.
        #
        # However, we also need to check that the 2-regular graph on leaves doesn't
        # create shorter cycles through the tree. A cycle in the full graph either:
        # (a) is entirely in the tree (impossible, it's a tree), or
        # (b) uses some leaf-edges. If it uses 2 leaf-edges, it goes leaf1 - leaf2 (edge),
        #     then through tree from leaf2 to leaf3, then leaf3 - leaf4 (edge), etc.
        #     Actually a cycle using exactly 2 leaf-edges: leaf_a -- leaf_b (added edge),
        #     then tree path from leaf_b to leaf_a. Length = 1 + tree_dist(leaf_a, leaf_b).
        #     We need this >= g. Since tree_dist >= g-1, length >= g. Good.
        #     A cycle using 3+ leaf-edges would be even longer. So we're safe!
        #
        # Great, so the construction works. Let me implement it.
        
        h = (g - 1) + 1  # depth, need 2h >= g-1... let me be precise
        # Need: for any two leaves connected by an added edge, tree_dist >= g-1
        # Maximum tree_dist between leaves in same root-subtree = 2*(h-1)
        # We only connect leaves from different root-subtrees, so min tree_dist = 2*h
        # Need 2*h >= g-1 => h >= (g-1)/2
        h = (g - 1) // 2 + 1  # ceil((g-1)/2) + 1 to be safe... actually (g-1)//2 might not suffice
        # For g=16: (15)//2 + 1 = 7+1 = 8. 2*8 = 16 >= 15. Good.
        # Actually we need 2h >= g-1 = 15, so h >= 8 (since h is integer). (g-1)//2 = 7, +1 = 8.
        
        # Build the tree
        # Level 0: root = 0
        # Level i (i>=1): nodes get 2 children each (parent + 2 children = 3)
        # Root gets 3 children
        
        adj = []  # adjacency list, will build incrementally
        parent = []  # parent of each node
        depth = []  # depth of each node
        children = []  # list of children for each node
        
        # BFS to build tree
        # Node 0 is root at depth 0
        adj.append([])
        parent.append(-1)
        depth.append(0)
        children.append([])
        
        current_level = [0]
        for d in range(1, h + 1):
            next_level = []
            for node in current_level:
                num_children = 2 if d >= 1 else 3  # root has 3, others have 2
                # Wait: root is at depth 0, its children are at depth 1.
                # Nodes at depth d (d>=1) have 2 children (at depth d+1).
                # But root (depth 0) has 3 children.
                # Let me restructure:
                pass
            
            # Actually let me redo this more carefully
            break
        
        # Let me rebuild more carefully
        adj = []
        parent = []
        node_depth = []
        children_list = []
        
        # Add root
        adj.append([])
        parent.append(-1)
        node_depth.append(0)
        children_list.append([])
        
        # Root's children: 3 nodes
        for i in range(3):
            new_id = len(adj)
            adj.append([])
            parent.append(0)
            node_depth.append(1)
            children_list.append([])
            children_list[0].append(new_id)
            adj[0].append(new_id)
            adj[new_id].append(0)
        
        # Levels 2 through h
        for d in range(2, h + 1):
            # All nodes at level d-1 get 2 children
            level_nodes = [i for i in range(len(adj)) if node_depth[i] == d - 1]
            for node in level_nodes:
                for _ in range(2):
                    new_id = len(adj)
                    adj.append([])
                    parent.append(node)
                    node_depth.append(d)
                    children_list.append([])
                    children_list[node].append(new_id)
                    adj[node].append(new_id)
                    adj[new_id].append(node)
        
        n_tree = len(adj)
        leaves = [i for i in range(n_tree) if node_depth[i] == h]
        num_leaves = len(leaves)
        
        # Each leaf needs 2 more edges
        # Group leaves by which root-child's subtree they're in
        # Root's children are 1, 2, 3 (indices)
        root_children = children_list[0]  # should be [1, 2, 3]
        
        # Assign each leaf to a group based on which root-child is its ancestor
        leaf_groups = {rc: [] for rc in root_children}
        for leaf in leaves:
            # Find which root-child is the ancestor
            node = leaf
            while node_depth[node] > 1:
                node = parent[node]
            # node is now at depth 1, i.e., a root-child
            leaf_groups[node].append(leaf)
        
        # Now we need to add 2 edges per leaf, with edges only between different groups
        # Each group has num_leaves/3 leaves (should be even for pairing)
        group_sizes = [len(leaf_groups[rc]) for rc in root_children]
        
        # We need a 2-regular graph on the leaves where each vertex has degree 2
        # and edges only go between different groups.
        # Simple approach: create a cycle that visits all leaves, alternating groups.
        # If each group has the same size, we can do: g1, g2, g3, g1, g2, g3, ...
        
        # Let's create the 2-regular structure as a single cycle
        # Shuffle leaves within each group
        for rc in root_children:
            rng.shuffle(leaf_groups[rc])
        
        # Create a cycle: interleave the three groups
        # If groups have sizes s1, s2, s3 (all equal = num_leaves/3)
        # Cycle: g1[0], g2[0], g3[0], g1[1], g2[1], g3[1], ..., g1[s-1], g2[s-1], g3[s-1]
        # This ensures consecutive elements are in different groups.
        
        s = num_leaves // 3
        if s * 3 != num_leaves:
            # Groups not equal size - use a different strategy
            # Just try random pairing with constraint
            pass
        
        cycle_order = []
        for i in range(s):
            cycle_order.append(leaf_groups[root_children[0]][i])
            cycle_order.append(leaf_groups[root_children[1]][i])
            cycle_order.append(leaf_groups[root_children[2]][i])
        
        # Add edges along the cycle
        for i in range(len(cycle_order)):
            u = cycle_order[i]
            v = cycle_order[(i + 1) % len(cycle_order)]
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n_tree):
            if len(adj[i]) != 3:
                return None
        
        # Check for multi-edges
        for i in range(n_tree):
            if len(adj[i]) != len(set(adj[i])):
                return None
        
        # Build edge list
        edges = []
        for u in range(n_tree):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return n_tree, edges, adj

    # Strategy:
    # 1. Try the tree-based construction for various depths
    # 2. If girth check passes, return it
    # 3. Otherwise, fall back to seed
    
    # For g=16: tree of depth 8 gives 766 vertices
    # Let's try it
    
    result = build_tree_3reg(g, 42)
    if result is not None:
        n, edges, adj = result
        # Verify girth
        if check_girth(adj, n, g):
            return n, edges
    
    # Try with different random seeds for the leaf ordering
    for seed in range(1, 20):
        if time.time() > deadline - 5:
            break
        result = build_tree_3reg(g, seed)
        if result is not None:
            n, edges, adj = result
            if check_girth(adj, n, g):
                return n, edges
    
    # Try a slightly different tree structure: depth h-1 but with more branching
    # Actually, let's try a different approach: use a path-based construction
    # where we create a "ladder" of cycles
    
    # Alternative: use the known fact that for 3-regular graphs,
    # we can use a "wiring" of a complete bipartite structure
    
    # Let's try: take a cycle of length L, and for each vertex, add two
    # "chords" that connect to vertices far away on the cycle.
    # This gives a 3-regular graph (cycle edges + 2 chords per vertex).
    # Wait, that's 2+2=4 per vertex. No.
    # Cycle gives degree 2. Add 1 chord per vertex = degree 3.
    # But chords must be paired (if u connects to v, v connects to u).
    # So we need n/2 chords. Each vertex gets exactly 1 chord.
    # Girth depends on chord placement.
    
    def build_chord_cycle(n, seed):
        """Build 3-regular graph: a cycle of n vertices + n/2 chords."""
        if n < 4 or n % 2 != 0:
            return None
        rng = random.Random(seed)
        adj = [[] for _ in range(n)]
        
        # Cycle edges
        for i in range(n):
            adj[i].append((i + 1) % n)
            adj[i].append((i - 1) % n)
        
        # Add chords: pair up vertices
        # For girth g, chords should connect vertices that are far apart
        # on the cycle (distance >= g-1 along cycle, or the chord creates
        # a cycle of length >= g)
        
        # A chord (i, j) creates a cycle of length min(|i-j|, n-|i-j|) + 1... no.
        # Actually, chord (i,j) with |i-j| = d (cyclic distance) creates
        # two cycles of length d+1 and n-d+1. We need both >= g.
        # So d >= g-1 and n-d >= g-1, i.e., g-1 <= d <= n-g+1.
        
        # For g=16: d in [15, n-15]. Need n >= 30 for this to be possible.
        # With n=766: d in [15, 751]. Plenty of room.
        
        # Random pairing with constraint
        vertices = list(range(n))
        rng.shuffle(vertices)
        used = [False] * n
        chords = []
        
        for i in range(0, n, 2):
            u = vertices[i]
            # Find a v such that cyclic distance is in [g-1, n-g+1]
            candidates = []
            for j in range(i + 1, n):
                v = vertices[j]
                if used[v]:
                    continue
                d = abs(u - v)
                d = min(d, n - d)
                if g - 1 <= d <= n - g + 1:
                    candidates.append(v)
            if not candidates:
                return None
            v = rng.choice(candidates)
            used[u] = True
            used[v] = True
            chords.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
            if len(adj[i]) != len(set(adj[i])):
                return None
        
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return n, edges, adj

    # Try chord cycles with various n
    for n in [766, 800, 834, 868, 902]:
        if time.time() > deadline - 5:
            break
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = build_chord_cycle(n, seed)
            if result is not None:
                nn, edges, adj = result
                if check_girth(adj, nn, g):
                    return nn, edges
    
    # Fallback to seed construction
    return seed_construct(k, g)
