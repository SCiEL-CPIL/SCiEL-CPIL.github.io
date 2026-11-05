import sys
from collections import deque

def construct(n):
    if n <= 4:
        # Standard recursive is optimal
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 5, use the Greedy TSP heuristic on the de Bruijn graph
    # Map symbols 1..n to 0..n-1
    # Node is a tuple of n-1 digits
    
    # Pre-compute all permutation nodes and their cycles
    # Number of permutations is n!
    # We need to map each permutation (tuple of n digits) to a node (tuple of n-1 digits)
    
    # Generate all permutations
    import itertools
    perms = list(itertools.permutations(range(n)))
    
    # Map permutation to node (first n-1 digits)
    # And identify the missing digit
    perm_nodes = {}
    for p in perms:
        node = tuple(p[:n-1])
        # The edge goes to tuple(p[1:])
        # The label is p[-1]
        perm_nodes[node] = p[-1] # The digit to append
        
    # Identify cycles in the permutation subgraph
    # Node u -> Node v where v = u[1:] + (u[0] shifted?)
    # Actually, if u is a permutation node, the next permutation node is determined by the appended digit.
    # Wait, the node is the prefix.
    # If we are at node u (prefix), and we append d, we go to node v = u[1:] + (d,).
    # For u to be a permutation node, u has distinct digits.
    # The appended digit d must be the one missing from u to form a full permutation.
    # So the next node v is uniquely determined.
    
    visited_nodes = set()
    cycles = []
    for p in perms:
        node = tuple(p[:n-1])
        if node in visited_nodes:
            continue
        # Trace the cycle
        cycle = []
        curr = node
        while curr not in visited_nodes:
            visited_nodes.add(curr)
            cycle.append(curr)
            # Find next node
            # The digit to append is the one missing from curr
            missing = [d for d in range(n) if d not in curr][0]
            next_node = tuple(list(curr[1:]) + [missing])
            curr = next_node
        cycles.append(cycle)
        
    # Now we have a list of cycles. Each cycle is a list of nodes.
    # We need to visit all cycles.
    # Start with an arbitrary cycle
    current_node = cycles[0][0]
    visited_cycles = set([0]) # Index of visited cycles
    word_digits = list(current_node)
    
    # Helper to get neighbors in de Bruijn graph
    def get_neighbors(node):
        # node is tuple of n-1 digits
        # Append d in 0..n-1
        # Next node is node[1:] + (d,)
        return [tuple(list(node[1:]) + [d]) for d in range(n)]
    
    # BFS to find nearest unvisited cycle
    # Returns (distance, path_nodes, target_cycle_idx)
    def bfs_nearest(start_node, visited_cycles_set):
        # Queue: (node, path)
        # We want to find the first node that belongs to an unvisited cycle.
        # We need to know which cycle a node belongs to.
        # Pre-compute node_to_cycle_idx
        pass

    # Pre-compute node to cycle index
    node_to_cycle = {}
    for i, c in enumerate(cycles):
        for node in c:
            node_to_cycle[node] = i
            
    def bfs_nearest(start_node, visited_cycles_set):
        q = deque()
        q.append((start_node, [start_node]))
        visited_bfs = {start_node}
        while q:
            node, path = q.popleft()
            # Check if this node is in an unvisited cycle
            if node in node_to_cycle:
                c_idx = node_to_cycle[node]
                if c_idx not in visited_cycles_set:
                    return path, c_idx
            # Expand
            for neighbor in get_neighbors(node):
                if neighbor not in visited_bfs:
                    visited_bfs.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return None, None

    # Greedy loop
    for _ in range(len(cycles)):
        # Find nearest unvisited cycle
        path, target_idx = bfs_nearest(current_node, visited_cycles)
        if path is None:
            break # Should not happen
            
        # Append the path digits to word
        # The path is a list of nodes.
        # The first node is current_node, which is already in word.
        # We need to append the digits that generate the transitions.
        # Node u -> Node v implies we appended the last digit of v.
        # So for each step in path, we append the last digit of the node.
        for node in path[1:]:
            word_digits.append(node[-1])
            
        # Now we are at path[-1], which is in cycle target_idx.
        # We must traverse this cycle.
        # We enter at path[-1].
        # We should traverse the cycle and exit.
        # Let's traverse the cycle starting from path[-1].
        # We need to add the digits for the cycle edges.
        # The cycle is a list of nodes.
        # Find the position of path[-1] in the cycle.
        c_nodes = cycles[target_idx]
        entry_idx = c_nodes.index(path[-1])
        
        # Traverse the cycle
        # We are at c_nodes[entry_idx].
        # The next node is c_nodes[(entry_idx + 1) % len(c_nodes)].
        # The digit to append is the last digit of the next node.
        for j in range(len(c_nodes)):
            next_node_idx = (entry_idx + j + 1) % len(c_nodes)
            next_node = c_nodes[next_node_idx]
            word_digits.append(next_node[-1])
            
        # After traversing the cycle, we are back at path[-1].
        current_node = path[-1]
        visited_cycles.add(target_idx)
        
    return "".join(map(str, [d+1 for d in word_digits]))
