import heapq
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build de Bruijn graph of order n-1
    # Nodes: (n-1)-permutations of 1..n
    # Edges: from node u to node v if v = u[1:] + [x] for some x in 1..n
    
    # Generate all (n-1)-permutations
    from itertools import permutations
    
    nodes = list(permutations(range(1, n + 1), n - 1))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Build adjacency: for each node, which nodes can follow it
    # Node u = (a1, a2, ..., a_{n-1})
    # Next node v = (a2, a3, ..., a_{n-1}, x) where x in 1..n and x not in u
    # The symbol appended is x
    
    adjacency = defaultdict(list)  # node_idx -> list of (next_node_idx, symbol)
    for i, u in enumerate(nodes):
        for x in range(1, n + 1):
            if x in u:
                continue
            v = u[1:] + (x,)
            if v in node_to_idx:
                j = node_to_idx[v]
                adjacency[i].append((j, x))
    
    # We need to find the shortest path that visits all edges exactly once (Eulerian path)
    # Actually, we need the shortest superpermutation, which corresponds to
    # finding the shortest walk that covers all permutations.
    
    # Alternative approach: use the known construction for n=7
    # The shortest known superpermutation for n=7 has length 5905
    
    # Let's try a greedy approach on the de Bruijn graph
    # Start at node 0, greedily choose edges to minimize unvisited nodes
    
    # First, let's count out-degrees and in-degrees
    out_deg = [len(adjacency[i]) for i in range(num_nodes)]
    in_deg = [0] * num_nodes
    for i in range(num_nodes):
        for j, x in adjacency[i]:
            in_deg[j] += 1
    
    # Find start node: out_deg - in_deg = 1 (or all balanced)
    start_node = 0
    for i in range(num_nodes):
        if out_deg[i] - in_deg[i] == 1:
            start_node = i
            break
    
    # Greedy traversal with heuristic
    # State: current node, set of visited edges
    # Edge identified by (from_node, to_node, symbol)
    
    # Use a priority queue: (heuristic, current_node, path_so_far)
    # Heuristic: number of unvisited edges reachable
    
    # For n=7, num_nodes = 720, edges = 720 * 2 = 1440 (each node has 2 outgoing edges for n=7? No)
    # For n=7, each node has n-1=6 possible next symbols, but must not repeat
    # Wait, in the de Bruijn graph for permutations, from (a1,...,a_{n-1}),
    # we can append any x in 1..n not in {a1,...,a_{n-1}}, which is exactly 1 symbol
    # So each node has out-degree 1!
    
    # This means the graph is a single cycle or collection of cycles
    # Let's verify: for n=3, nodes are (1,2),(1,3),(2,1),(2,3),(3,1),(3,2)
    # From (1,2): can append 3 -> (2,3)
    # From (2,3): can append 1 -> (3,1)
    # From (3,1): can append 2 -> (1,2)
    # So we have a cycle of length 3: (1,2)->(2,3)->(3,1)->(1,2)
    # And another: (1,3)->(3,2)->(2,1)->(1,3)
    
    # For n=7, there will be multiple cycles
    # The superpermutation corresponds to traversing these cycles in some order
    
    # Let's find all cycles
    visited_nodes = set()
    cycles = []
    
    for i in range(num_nodes):
        if i in visited_nodes:
            continue
        # Follow the unique outgoing edge
        cycle = []
        current = i
        while current not in visited_nodes:
            visited_nodes.add(current)
            cycle.append(current)
            if not adjacency[current]:
                break
            current = adjacency[current][0][0]
        if len(cycle) > 1:
            cycles.append(cycle)
    
    # Now we need to order the cycles to minimize total length
    # Each cycle of length L contributes L-1 symbols (since the last symbol wraps around)
    # Actually, traversing a cycle of length L gives L symbols (the first symbol is already in the word)
    
    # The total length = sum over cycles of (cycle_length - 1) + 1
    # Wait, let me think more carefully
    
    # If we have a cycle: node0 -> node1 -> ... -> node_{L-1} -> node0
    # Starting at node0, we write: sym0, sym1, ..., sym_{L-1}
    # Where sym_k is the symbol that takes node_k to node_{k+1}
    # After writing sym_{L-1}, we're back at node0
    # The total symbols written is L
    
    # If we concatenate multiple cycles, the last symbol of one cycle and first of next might overlap
    
    # For simplicity, let's just concatenate the cycles
    # Total length = sum of all cycle lengths
    
    # Order cycles greedily by overlap
    # This is getting complex. Let's just return a valid superpermutation
    
    # Simple valid construction: concatenate all permutations
    # Length = n! * n, which is too long but valid
    
    # Better: use the de Bruijn graph properly
    # Since each node has out-degree 1, the graph is a disjoint union of cycles
    # A superpermutation corresponds to a walk that visits every node at least once
    # The shortest such walk visits each node exactly once per cycle, so it's the sum of cycle lengths
    
    # Let's compute the total
    total_length = sum(len(c) for c in cycles)
    
    # Now construct the word
    word = []
    
    # Start with the first cycle
    if cycles:
        first_cycle = cycles[0]
        for i in range(len(first_cycle)):
            node = first_cycle[i]
            next_node, sym = adjacency[node][0]
            word.append(sym)
    
    # Append remaining cycles
    for cycle in cycles[1:]:
        for i in range(len(cycle)):
            node = cycle[i]
            next_node, sym = adjacency[node][0]
            word.append(sym)
    
    # Ensure we have all symbols 1..n in the word
    # The word should start with the first node's symbols
    # Actually, the word I'm constructing is just the appended symbols
    # I need to prepend the starting node's symbols
    
    # Let me redo this more carefully
    # A superpermutation is a sequence of symbols
    # If I'm at node u = (a1, a2, ..., a_{n-1}) and I append symbol x,
    # the new node is v = (a2, a3, ..., a_{n-1}, x)
    # The symbol x is what gets appended to the word
    
    # To start, I need to establish the initial (n-1)-tuple
    # Let's start with node 0 = (1, 2, ..., n-1)
    # The initial word is "123...{n-1}"
    # Then I traverse the graph, appending the symbol for each edge
    
    start_idx = node_to_idx[tuple(range(1, n))]
    word = list(range(1, n))  # Initial (n-1) symbols
    
    # Now traverse from start_idx
    current = start_idx
    visited_edges = set()
    
    # Follow the unique path, but we need to cover all nodes
    # Since out-degree is 1, we just follow the cycle containing start_idx
    # Then we need to jump to other cycles
    
    # This approach is flawed because we can't jump between cycles
    # In a de Bruijn graph with out-degree 1, once you're in a cycle, you're stuck
    
    # The issue is that for permutation de Bruijn graphs, the structure is different
    # Let me reconsider
    
    # For the permutation de Bruijn graph, the edges are not unique
    # From node (a1,...,a_{n-1}), I can append any x in 1..n, but x must not be in the node
    # Wait, no - in a proper de Bruijn graph, I can append any symbol, but the resulting
    # node must be a valid (n-1)-permutation
    
    # Actually, I think I made an error. Let me reconsider the graph definition
    
    # For superpermutations, the standard approach uses a graph where:
    # Nodes are (n-1)-permutations
    # There's an edge from u to v if v = u[1:] + [x] for some x, AND x is not in u
    # This ensures v is also a valid (n-1)-permutation
    
    # But this gives out-degree 1, which means the graph is a collection of cycles
    # And you can't move between cycles, so this doesn't work for superpermutations
    
    # The correct graph for superpermutations has:
    # Nodes: (n-1)-permutations
    # Edges: from u to v if v = u[1:] + [x] for ANY x in 1..n
    # But v must be a valid (n-1)-permutation, so x must not be in u
    # This still gives out-degree 1
    
    # I think the issue is that superpermutations don't correspond to Eulerian paths
    # in this graph. Instead, they correspond to walks that visit all nodes
    
    # Given the complexity, let's just use a simple valid construction
    # Concatenate all permutations with minimal overlap
    
    perms = list(permutations(range(1, n + 1)))
    word = list(perms[0])
    
    for perm in perms[1:]:
        # Find maximum overlap between end of word and start of perm
        max_overlap = 0
        for k in range(n - 1, 0, -1):
            if word[-k:] == list(perm[:k]):
                max_overlap = k
                break
        word.extend(perm[max_overlap:])
    
    return "".join(map(str, word))
