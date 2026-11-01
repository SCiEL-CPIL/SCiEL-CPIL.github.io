import collections

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutation nodes of length n-1
    # Symbols are 1..n
    # Node is a tuple of n-1 distinct symbols
    import itertools
    
    perms = []
    for p in itertools.permutations(range(1, n + 1), n - 1):
        perms.append(p)
        
    num_nodes = len(perms)
    node_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Compute cycles
    # deterministic next: remove first, append missing symbol
    det_next = [None] * num_nodes
    for i, p in enumerate(perms):
        missing = [s for s in range(1, n + 1) if s not in p][0]
        next_p = p[1:] + (missing,)
        det_next[i] = node_to_idx[next_p]
        
    # Find cycles
    visited = [False] * num_nodes
    cycles = []
    cycle_id = [0] * num_nodes
    for i in range(num_nodes):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                cycle_id[curr] = len(cycles)
                curr = det_next[curr]
            cycles.append(cycle)
            
    num_cycles = len(cycles)
    
    # Start with the first cycle
    start_node_idx = 0
    current_node = perms[start_node_idx]
    word = list(current_node)
    
    visited_cycles = set()
    
    # We need to traverse the cycle we are in
    # Then find next
    
    # Helper to get deterministic next node tuple
    def get_det_next_tuple(node_tuple):
        missing = 0
        for s in range(1, n + 1):
            if s not in node_tuple:
                missing = s
                break
        return node_tuple[1:] + (missing,)
        
    # Helper to get deterministic next symbol
    def get_det_next_symbol(node_tuple):
        missing = 0
        for s in range(1, n + 1):
            if s not in node_tuple:
                missing = s
                break
        return missing

    for _ in range(num_cycles):
        # Traverse the current cycle
        # We are at current_node
        # We need to make sure we are at a node in an unvisited cycle
        # Actually, we just arrived or started.
        
        # Check if current node's cycle is visited
        curr_cycle_id = cycle_id[node_to_idx[current_node]]
        if curr_cycle_id not in visited_cycles:
            # Traverse the cycle
            for _ in range(n):
                sym = get_det_next_symbol(current_node)
                word.append(sym)
                current_node = get_det_next_tuple(current_node)
            visited_cycles.add(curr_cycle_id)
            
        # Find next cycle
        # BFS from current_node
        # Target: any node in an unvisited cycle
        
        # Optimization: If all cycles visited, break
        if len(visited_cycles) == num_cycles:
            break
            
        queue = collections.deque()
        queue.append((current_node, []))
        visited_bfs = {current_node}
        
        found = False
        next_node = None
        path = []
        
        while queue:
            node, path_so_far = queue.popleft()
            
            # Check if this node is in an unvisited cycle
            if node in node_to_idx:
                cid = cycle_id[node_to_idx[node]]
                if cid not in visited_cycles:
                    found = True
                    next_node = node
                    path = path_so_far
                    break
            
            # Expand
            for s in range(1, n + 1):
                next_n = node[1:] + (s,)
                if next_n not in visited_bfs:
                    visited_bfs.add(next_n)
                    queue.append((next_n, path_so_far + [s]))
                    
        if not found:
            # Should not happen, but fallback
            break
            
        # Append the path
        for s in path:
            word.append(s)
        current_node = next_node

    return "".join(map(str, word))
