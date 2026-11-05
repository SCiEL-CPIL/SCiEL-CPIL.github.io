"""Deterministic greedy tree-folding search for the (4,10)-cage."""
import sys
import time
from collections import deque

# --- Finite-field helpers for GF(4) (char-2) and GF(k) (prime) ---
def _gf4_ops():
    # GF(4) = {0,1,2,3} with x^2 = x + 1; use representation: 0=0,1=1,2=alpha,3=alpha+1
    # addition is XOR; multiplication table below.
    add = lambda a, b: a ^ b
    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    mul = lambda a, b: table[a][b]
    return add, mul

def _gfp_ops(p):
    add = lambda a, b: (a + b) % p
    mul = lambda a, b: (a * b) % p
    return add, mul

# --- LUW / Lazebnik–Ustimenko–Woldar construction (fallback) ---
def _luw_construct(k, g):
    """Build the LUW graph CD(d, q) with q = k, choosing d so girth >= g.
    Returns (n, edges)."""
    q = k
    # For odd girth g, the LUW graph has girth at least d + 5 when d is odd.
    # We want d + 5 >= g  =>  d >= g - 5.  Choose smallest odd d >= g - 5.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    if q == 4:
        add, mul = _gf4_ops()
        sub = add  # in char 2, sub == add
    else:
        add, mul = _gfp_ops(q)
        sub = lambda a, b: (a - b) % q

    # Rule for LUW: for c = 2..d, (a_c, b_c)
    # For c = 2: (1,1); c=3: (2,1); c=4: (1,2)
    # For c >= 5: periodic pattern with period 4
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Build graph via BFS from the zero point on side 0
    # A vertex is (side, x) where side in {0,1}, x is a tuple of length d+1
    # x[0] is unused (None), x[1..d] are field elements
    # Point ~ Line iff for all c in 2..d: line[c] - point[c] = line[a_c] * point[b_c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p, find line l: l[c] = p[c] + l[a]*p[b]  -- but l[a] is unknown
                    # Actually: l[c] - p[c] = l[a] * p[b]
                    # We need to solve for l given p. The LUW construction defines:
                    # Given point p, line l is determined by l[1] = first, and for c=2..d:
                    # l[c] = p[c] + l[a] * p[b]
                    # But l[a] may refer to an already-determined coordinate.
                    # We build l incrementally: l[1] is set, then for c=2..d compute l[c].
                    # But l[a] might not be computed yet if a >= c. Let's check the rule.
                    # For the standard LUW, a_c < c always holds.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l, find point p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a]*p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# --- Greedy tree-folding search ---
def _greedy_fold_search(k, g, time_limit):
    """Attempt to find a 4-regular graph of girth >= g via greedy folding of the infinite tree.
    Returns (n, edges) or None if no valid graph found within time_limit."""
    if k != 4 or g != 10:
        return None
        
    # Construct a 4-regular tree of depth 5 (radius 5)
    # Levels 0 to 5.
    # Level 0: 1 node (root)
    # Level 1: 4 nodes
    # Level 2: 16 nodes
    # Level 3: 64 nodes
    # Level 4: 256 nodes
    # Level 5: 1024 nodes (leaves)
    
    # Total nodes: 1 + 4 + 16 + 64 + 256 + 1024 = 1365
    
    # We will identify the leaves of the first 4 branches (Level 5 nodes whose parent is in Level 4, 
    # whose parent is in Level 3, whose parent is in Level 2, whose parent is in Level 1 branch 0) 
    # with leaves of the second 4 branches? 
    # No, we connect leaves from Branch 0 to Branch 1, and Branch 2 to Branch 3.
    # This creates a 4-regular graph if we add 3 edges per leaf.
    
    # Let's define the tree structure explicitly.
    # Node ID: 0 for root.
    # Children of node u: u*4 + i for i in 0..3? No, that's for a full 4-ary tree.
    # Let's use a list of nodes.
    
    nodes = []
    parent = []
    children = [[] for _ in range(1365)] # Pre-allocate max size
    
    # Generate tree
    # Level 0
    nodes.append(0)
    parent.append(-1)
    
    # Level 1
    for i in range(4):
        nodes.append(len(nodes))
        parent.append(0)
        children[0].append(nodes[-1])
        
    # Level 2
    for u in nodes[1:5]:
        for i in range(4):
            nodes.append(len(nodes))
            parent.append(u)
            children[u].append(nodes[-1])
            
    # Level 3
    for u in nodes[5:21]:
        for i in range(4):
            nodes.append(len(nodes))
            parent.append(u)
            children[u].append(nodes[-1])
            
    # Level 4
    for u in nodes[21:85]:
        for i in range(4):
            nodes.append(len(nodes))
            parent.append(u)
            children[u].append(nodes[-1])
            
    # Level 5
    for u in nodes[85:341]:
        for i in range(4):
            nodes.append(len(nodes))
            parent.append(u)
            children[u].append(nodes[-1])
            
    n = len(nodes)
    
    # Edges from tree
    edges = []
    for u in range(1, n):
        p = parent[u]
        edges.append((p, u))
        
    # Now add edges between leaves to make it 4-regular.
    # Leaves are nodes 341 to 1364.
    # Branch 0 leaves: parents are 85..100 (Level 4 nodes in Branch 0)
    # Branch 1 leaves: parents are 101..116
    # Branch 2 leaves: parents are 117..132
    # Branch 3 leaves: parents are 133..148
    
    # We pair Branch 0 with Branch 1, and Branch 2 with Branch 3.
    # For each pair, we create a 3-regular bipartite graph.
    
    def add_bipartite_edges(leaves_A, leaves_B):
        # leaves_A and leaves_B are lists of node IDs, each of size 256.
        # We need to add edges such that each node in A and B gets 3 new edges.
        # Constraint: For any path u-v-w (u,w in A, v in B), dist_tree(u,w) >= 8.
        # This is satisfied if u and w are in different "sub-branches" of A.
        # Sub-branches of A are determined by the Level 2 ancestor.
        # Leaves in A are indexed 0..255.
        # Index i corresponds to path from Level 2 ancestor.
        # First digit of path (Level 3 choice) determines sub-branch.
        # i // 64 gives the sub-branch index (0..3).
        
        # We connect u in A to v in B if subbranch_A(u) != subbranch_A(w) for any w connected to v.
        # This is hard to guarantee globally with a simple rule.
        # Instead, we use a rule that ensures local distance.
        # Connect u (idx i) to v (idx j) if (i + j) % 4 != 0? No.
        
        # Let's try: Connect u_i to v_{i}, v_{i+64}, v_{i+128} (mod 256).
        # Subbranch of u_i is i // 64.
        # Subbranch of v_j is j // 64.
        # If we connect u_i to v_j, and v_j to u_k, we need subbranch(i) != subbranch(k).
        # This requires that the neighbors of any v_j have distinct subbranches.
        # If we define the bipartite graph by:
        # Edge (i, j) exists if j == (i + offset) % 256 for offset in {0, 64, 128}?
        # Then v_j is connected to u_{j}, u_{j-64}, u_{j-128}.
        # Subbranches: j//64, (j-64)//64, (j-128)//64.
        # If j = 0: subbranches 0, 3, 2. Distinct.
        # If j = 1: subbranches 0, 3, 2. Distinct.
        # This works!
        
        offsets = [0, 64, 128]
        for i in range(256):
            u = leaves_A[i]
            for off in offsets:
                j = (i + off) % 256
                v = leaves_B[j]
                edges.append((u, v))
                
    # Identify leaves for each branch
    # Branch 0 leaves: children of nodes 85..100
    leaves_0 = []
    for u in range(85, 101):
        leaves_0.extend(children[u])
        
    leaves_1 = []
    for u in range(101, 117):
        leaves_1.extend(children[u])
        
    leaves_2 = []
    for u in range(117, 133):
        leaves_2.extend(children[u])
        
    leaves_3 = []
    for u in range(133, 149):
        leaves_3.extend(children[u])
        
    add_bipartite_edges(leaves_0, leaves_1)
    add_bipartite_edges(leaves_2, leaves_3)
    
    return n, edges

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    start_time = time.time()
    
    # Try the greedy fold search first (as per directive)
    result = _greedy_fold_search(k, g, time_limit=5.0)
    
    if result is not None:
        n, edges = result
        # Validate the result
        if _validate_graph(n, edges, k, g):
            return n, edges
    
    # Fall back to the LUW construction
    return _luw_construct(k, g)

def _validate_graph(n, edges, k, g):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    # Check simple: no loops, no repeated edges
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        e = (min(u, v), max(u, v))
        if e in edge_set:
            return False
        edge_set.add(e)
    
    # Check degree
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if any(d != k for d in degree):
        return False
    
    # Check girth >= g using BFS from each vertex
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for src in range(n):
        # BFS from src
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif abs(dist[v] - dist[u]) <= 1 and dist[v] != dist[u] + 1:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
