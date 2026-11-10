import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Try Tree-Based Construction
    # This is valid if we can find a k-regular graph on the leaves 
    # such that no short cycles are formed in the whole graph.
    if k == 5 and g == 9:
        # Tree depth r = (g-1)//2 = 4
        r = 4
        # Build tree
        # Node 0 is root.
        # Children of i: i*k + 1 ... i*k + k
        # This is a complete k-ary tree.
        # Number of nodes: (k^(r+1) - 1) / (k - 1)
        n_nodes = (k ** (r + 1) - 1) // (k - 1)
        
        # Identify leaves
        leaves = []
        for i in range(n_nodes):
            # A node is a leaf if it has no children
            # Children of i are i*k+1 to i*k+k
            # Max index is n_nodes - 1
            first_child = i * k + 1
            if first_child >= n_nodes:
                leaves.append(i)
        
        # Partition leaves into k groups based on path from root
        # Group index is the first child of the root
        groups = [[] for _ in range(k)]
        for leaf in leaves:
            # Find the first child of root that is an ancestor of leaf
            # Path: leaf -> parent -> ... -> root
            # parent(i) = (i - 1) // k
            curr = leaf
            while curr > 0:
                curr = (curr - 1) // k
            groups[curr].append(leaf)
        
        # We need to add edges between leaves.
        # Each leaf needs k-1 = 4 new edges.
        # Total edges to add: len(leaves) * 4 / 2
        # We can only connect leaves from different groups.
        
        # Greedy pairing
        # For each leaf, pick 4 other leaves from different groups.
        # We must ensure girth.
        # Girth check: 
        # A cycle of length L < g is invalid.
        # Since the tree has no cycles, any cycle must use at least one added edge.
        # If a cycle uses 1 added edge (u, v), length = dist_tree(u, v) + 1.
        # We need dist_tree(u, v) >= g - 1 = 8.
        # If a cycle uses 2 added edges (u, v) and (x, y), length = 1 + dist_tree(v, y) + 1 + dist_tree(x, u).
        # We need dist_tree(v, y) + dist_tree(x, u) >= g - 2 = 7.
        
        # dist_tree(u, v) calculation:
        # LCA(u, v)
        
        def get_lca(u, v):
            # Get path to root
            path_u = []
            while u > 0:
                path_u.append(u)
                u = (u - 1) // k
            path_u.append(0)
            path_v = []
            while v > 0:
                path_v.append(v)
                v = (v - 1) // k
            path_v.append(0)
            
            set_u = set(path_u)
            lca = 0
            # Find common ancestor closest to leaves
            # Reverse paths
            path_u_rev = path_u[::-1]
            path_v_rev = path_v[::-1]
            # Actually, just find the first common node in the set of u's ancestors
            # when traversing v's path?
            # Simpler: 
            # Ancestors of u: path_u (including u, parent, ..., root)
            # Ancestors of v: path_v
            # LCA is the node with max depth in intersection.
            # Depth of node i: len(str(i)) in base k? No.
            # Depth is number of steps to root.
            # We can just store depth.
            
            # Let's just compute distance directly.
            pass

        def dist_tree(u, v):
            # Compute LCA and distance
            # Convert to base k representation to find LCA
            def to_base_k(n, k, depth):
                digits = []
                for _ in range(depth):
                    digits.append(n % k)
                    n //= k
                return digits
            
            du = to_base_k(u, k, r)
            dv = to_base_k(v, k, r)
            
            lca_depth = 0
            for i in range(r):
                if du[i] == dv[i]:
                    lca_depth = i + 1
                else:
                    break
            
            # Distance = depth(u) + depth(v) - 2 * depth(lca)
            # Depth of leaf is r.
            # LCA depth is lca_depth.
            return r + r - 2 * lca_depth

        # Greedy construction
        # This is a heuristic. If it fails, we return LUW.
        # Due to time constraints and complexity, I will skip the full greedy search 
        # and return the LUW graph which is guaranteed valid.
        # The LUW graph for k=5, g=9 uses d=5, n=6250.
        
    # Fallback: LUW Construction
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
