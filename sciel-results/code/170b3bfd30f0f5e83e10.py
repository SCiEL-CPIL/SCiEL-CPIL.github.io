import random

def construct(k, g):
    # Specific optimized construction for k=4, g=10
    if k == 4 and g == 10:
        return construct_4_10()
    
    # Fallback for other parameters (not expected in this task instance)
    # Use a simple cycle or similar to avoid crashing
    # But the task is specifically about the cage problem n(4,10)
    # So we assume k=4, g=10.
    # If other values are passed, we can try a generic greedy approach or return a trivial graph.
    # Given the context, I will implement a generic greedy for other cases just in case, 
    # but optimize for 4,10.
    
    # For now, if not 4,10, raise an exception or return a basic graph.
    # The prompt says "the evaluator passes this task's degree k and girth g".
    # So it will be 4 and 10.
    
    # Generic fallback: Return a 4-regular graph on some vertices.
    # e.g. Circulant graph C_N(1,2) which has girth 4.
    # This is invalid for g=10.
    # But since the task is specific, I'll just handle 4,10.
    raise ValueError("Optimized only for k=4, g=10")

def construct_4_10():
    random.seed(42) # For reproducibility
    
    depth = 5
    # Build 4-regular tree
    # 0: root
    # 1-4: level 1
    # ...
    
    nodes = [0]
    parent = {0: -1}
    children = {0: []}
    depth_map = {0: 0}
    
    # BFS
    queue = [0]
    while queue:
        u = queue.pop(0)
        d = depth_map[u]
        if d == depth:
            continue
            
        if u == 0:
            num_children = 4
        else:
            num_children = 3
            
        start_idx = len(nodes)
        for i in range(num_children):
            v = start_idx + i
            nodes.append(v)
            parent[v] = u
            if u not in children:
                children[u] = []
            children[u].append(v)
            depth_map[v] = d + 1
            queue.append(v)
            
    n = len(nodes)
    edges = []
    
    # Add tree edges
    for v in range(1, n):
        edges.append((parent[v], v))
        
    # Identify leaves and their branch
    leaves = []
    for v in range(n):
        if depth_map[v] == depth:
            # Find branch (child of root)
            u = v
            while parent[u] != 0:
                u = parent[u]
            # u is the child of root (1..4)
            branch = u - 1 # 0..3
            leaves.append((v, branch))
            
    # Each leaf needs 3 connections
    stubs = []
    for v, b in leaves:
        for _ in range(3):
            stubs.append((v, b))
            
    # Shuffle stubs
    random.shuffle(stubs)
    
    # Pair stubs, ensuring different branches
    # We have 3 * 324 = 972 stubs.
    # We need to pair them up.
    # If we just pair sequentially, we might get same branch.
    # We will do a greedy fix.
    
    used = [False] * len(stubs)
    new_edges = []
    
    for i in range(0, len(stubs), 2):
        v1, b1 = stubs[i]
        v2, b2 = stubs[i+1]
        
        if b1 == b2:
            # Find a j > i+1 such that stubs[j][1] != b1
            # and stubs[j] is not used
            # Since branches are balanced, there should be plenty of others.
            swapped = False
            for j in range(i+2, len(stubs), 2):
                if not used[j]:
                    v3, b3 = stubs[j]
                    if b3 != b1:
                        # Swap stubs[i+1] and stubs[j]
                        stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                        v2, b2 = stubs[i+1]
                        swapped = True
                        break
            if not swapped:
                # Should not happen if distribution is good
                # Force a connection to a random other branch stub
                # Find any unused stub with different branch
                for j in range(len(stubs)):
                    if not used[j] and j != i and j != i+1:
                        if stubs[j][1] != b1:
                            stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                            v2, b2 = stubs[i+1]
                            break
        
        # Add edge
        if v1 != v2: # Should be distinct leaves
            new_edges.append((v1, v2))
            
        used[i] = True
        used[i+1] = True
        
    edges.extend(new_edges)
    
    return n, edges
