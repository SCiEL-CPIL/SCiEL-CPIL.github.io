import time
import random
import collections

def construct(k, g):
    if k != 3:
        # Fallback for other k
        return luw_construct(k, g)
        
    if g > 16:
        # For larger g, this specific tree depth might not suffice or might be too big
        # But for the task g=16 is the target.
        pass

    start_time = time.time()
    time_limit = 100
    
    # Target girth g
    # Build tree of depth d such that max distance in tree is >= g-1
    # Max distance in tree of depth d is 2*d.
    # We need cycles of length >= g.
    # A cycle using one matching edge has length dist(u,v) + 1.
    # So we need dist(u,v) >= g-1.
    # Max dist in tree is 2*d. So 2*d >= g-1 => d >= (g-1)/2.
    # For g=16, d >= 7.5 => d=8.
    
    d = (g - 1 + 1) // 2 # ceil((g-1)/2)
    if (g-1) % 2 != 0:
        d = (g-1)//2 + 1
    else:
        d = (g-1)//2
        
    # For g=16, d=8.
    
    # Build Tree
    # Root is 0.
    # Children of v: 3*v + 1, 3*v + 2, 3*v + 3?
    # Let's use a simple indexing.
    # Level 0: [0]
    # Level 1: [1, 2, 3]
    # Level 2: [4..9]
    # ...
    
    nodes = [0]
    for level in range(1, d + 1):
        start_idx = len(nodes)
        for _ in range(3 * (2 ** (level - 1))):
            nodes.append(-1) # Placeholder
        # Connect to parents
        # This is getting complicated to index.
        # Let's use a list of (parent, children)
        
    # Easier: Generate tree structure
    tree = {}
    nodes = [0]
    tree[0] = []
    for level in range(1, d + 1):
        new_nodes = []
        for node in nodes:
            children = [len(tree) + i for i in range(3)]
            tree[node] = children
            for c in children:
                tree[c] = []
                new_nodes.append(c)
        nodes = new_nodes
        
    n = len(tree)
    leaves = nodes # Nodes at level d
    
    # Verify tree size
    # Level 0: 1
    # Level 1: 3
    # Level d: 3 * 2^(d-1)
    # Total: 1 + 3 * (2^d - 1)
    
    # Calculate distances between all pairs of leaves?
    # Too expensive for 384 leaves (384^2 / 2 ~ 70k pairs). OK.
    # But we need to match 2 stubs per leaf.
    
    # Let's use a randomized greedy matching.
    # Stubs: (leaf, stub_id) where stub_id in {0, 1}
    # Total stubs: 2 * len(leaves)
    
    # Precompute distance between any two leaves
    # dist[u][v]
    # We can compute LCA or just BFS from each leaf?
    # BFS from each leaf is 384 * 766 ~ 300k ops. Fast.
    
    dist_matrix = {}
    for leaf in leaves:
        # BFS from leaf
        dist = {leaf: 0}
        queue = collections.deque([leaf])
        while queue:
            u = queue.popleft()
            # Get neighbors in tree
            # Tree is directed parent->children. We need undirected.
            # Let's build adjacency list for tree.
            pass
