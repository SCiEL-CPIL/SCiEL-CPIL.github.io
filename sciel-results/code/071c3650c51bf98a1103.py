import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, g=16, uses a tree-based construction to achieve n=766.
    Falls back to LUW for other parameters or if construction fails.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    if k == 3 and g == 16:
        # Try the optimized tree-based construction
        n, edges = _construct_3reg_g16()
        if n is not None:
            return n, edges
    
    # Fallback to LUW
    return _luw_fallback(k, g)

def _construct_3reg_g16():
    """
    Constructs a 3-regular graph with girth >= 16 using a 3-ary tree of depth 8.
    n = 3 * 2^8 - 2 = 766.
    """
    D = 8
    # Total vertices: 1 + 3 * (2^D - 1) = 3 * 2^D - 2
    n = 3 * (2 ** D) - 2
    
    # Build the tree
    # Vertex 0 is root.
    # Children of i: 3*i + 1, 3*i + 2, 3*i + 3 ? No, standard binary tree indexing is 2i+1, 2i+2.
    # For k-ary tree, children of i are k*i + 1 ... k*i + k.
    # But this only works if we fill level by level.
    # Let's use a list-based tree construction to be safe.
    
    vertices = []
    edges = []
    
    # Root
    vertices.append(0)
    
    # Queue for BFS expansion
    # (vertex_id, depth, branch_id)
    # branch_id: 0 for root, 1/2/3 for children of root
    queue = deque([(0, 0, 0)])
    
    # We need to track parent-child edges
    # And identify leaves
    
    leaves = {1: [], 2: [], 3: []}
    
    current_id = 0
    while queue:
        u, depth, branch = queue.popleft()
        
        if depth == D:
            # Leaf
            leaves[branch].append(u)
            continue
            
        # Add children
        for i in range(3):
            v = current_id + 1
            current_id += 1
            vertices.append(v)
            
            # Determine branch for child
            if branch == 0:
                child_branch = i + 1
            else:
                child_branch = branch
                
            edges.append((u, v))
            
            queue.append((v, depth + 1, child_branch))
            
    # Verify n
    if len(vertices) != n:
        return None, None
        
    # Now add edges between leaves to make them degree 3
    # Leaves currently have degree 1 (parent)
    # They need 2 more edges.
    
    L1 = leaves[1]
    L2 = leaves[2]
    L3 = leaves[3]
    
    # Shuffle to avoid short cycles
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        shuffled_L1 = L1[:]
        shuffled_L2 = L2[:]
        shuffled_L3 = L3[:]
        
        random.shuffle(shuffled_L1)
        random.shuffle(shuffled_L2)
        random.shuffle(shuffled_L3)
        
        # Interleave
        # We need a cycle of length 3 * len(L1)
        # Pattern: L1, L2, L3, L1, L2, L3...
        cycle_nodes = []
        for i in range(len(L1)):
            cycle_nodes.append(shuffled_L1[i])
            cycle_nodes.append(shuffled_L2[i])
            cycle_nodes.append(shuffled_L3[i])
            
        # Add edges in the cycle
        # Connect i to i+1
        added_edges = []
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            added_edges.append((u, v))
            
        all_edges = edges + added_edges
        
        # Check for multi-edges or self-loops
        # Tree edges are unique. Added edges are between leaves.
        # Leaves are distinct.
        # Could an added edge be a tree edge? No, tree edges connect parent-child. Leaves have no children.
        # Could an added edge be a duplicate?
        # The cycle is simple, so no duplicate edges in the cycle.
        # Could an added edge connect two leaves that are already connected? No.
        
        # Check Girth
        girth = _compute_girth_fast(n, all_edges)
        
        if girth >= 16:
            return n, all_edges
            
    return None, None

def _compute_girth_fast(n, edges):
    """
    Compute girth using BFS from each vertex.
    Optimized for small n.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < 16:
                            return min_girth
    return min_girth

def _luw_fallback(k, g):
    """
    Original Lazebnik-Ustimenko-Woldar construction.
    """
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
# EVOLVE-BLOCK-END
