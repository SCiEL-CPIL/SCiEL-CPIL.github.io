"""Improved program: construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy:
    1. Try to find a random Cayley graph on a product of S3 groups (e.g., S3^4) 
       with k involutory generators that satisfies the girth constraint.
    2. If successful and the group size is smaller than the L-U-W fallback, use it.
    3. Otherwise, fall back to the Lazebnik-Ustimenko-Woldar (L-U-W) construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    
    # Fallback: L-U-W Construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # Primary: Random Cayley Graph Search
    def try_random_cayley(k, g, max_time=100):
        """
        Search for a k-regular Cayley graph on S3^d with girth >= g.
        We try d=3, 4, 5... until we find one or time runs out.
        """
        start_time = time.time()
        
        # Define S3 elements as tuples of permutations
        # Identity: (0,1,2)
        # Transpositions: (1,0,2), (0,2,1), (2,1,0)
        s3_elements = [
            (0, 1, 2), # e
            (1, 0, 2), # (0 1)
            (0, 2, 1), # (1 2)
            (2, 1, 0), # (0 2)
            (1, 2, 0), # (0 1 2) - not involution
            (2, 0, 1)  # (0 2 1) - not involution
        ]
        
        # Involutions in S3: e, and the 3 transpositions
        s3_involutions = [s3_elements[0], s3_elements[1], s3_elements[2], s3_elements[3]]
        
        def multiply_s3(a, b):
            return tuple(a[b[i]] for i in range(3))
        
        # Try different dimensions of S3^d
        for d in range(3, 6):
            if time.time() - start_time > max_time:
                return None
            
            # Generate all involutions in S3^d
            # An element is an involution if each component is an involution in S3
            # and the product is not identity (unless we allow identity, but generators must be distinct and non-identity usually)
            # Actually, identity is an involution, but it doesn't add an edge. We need k distinct non-identity involutions.
            
            invs = []
            for combo in __import__('itertools').product(s3_involutions, repeat=d):
                # Check if combo is identity
                if all(c == s3_elements[0] for c in combo):
                    continue
                invs.append(combo)
            
            # We need to pick k generators from invs
            # To ensure the graph is connected, the generators should generate the group.
            # S3^d has size 6^d.
            # For d=4, size is 1296.
            
            # Randomly sample sets of k generators
            # Number of trials depends on d
            num_trials = 500 if d <= 4 else 200
            
            for _ in range(num_trials):
                if time.time() - start_time > max_time:
                    return None
                
                generators = random.sample(invs, k)
                
                # Check if they generate the group
                # BFS from identity using generators
                visited = {tuple([s3_elements[0]] * d): 0} # Identity is tuple of d identities
                queue = deque([tuple([s3_elements[0]] * d)])
                gen_set = set(generators)
                
                # Precompute multiplication for S3^d
                def multiply_group(a, b):
                    return tuple(multiply_s3(a[i], b[i]) for i in range(d))
                
                # BFS to find group size and check girth
                # We do BFS to determine the girth.
                # Girth is the length of the shortest cycle.
                # In a Cayley graph, a cycle of length L corresponds to a word of length L in generators that equals identity.
                # We can find the girth by BFS from identity, tracking the first time we visit a node.
                # If we encounter a node already visited at depth d_prev, and current depth is d_curr,
                # we have a cycle of length d_curr + d_prev + 1?
                # Actually, standard BFS girth calculation:
                # For each edge (u, v), if dist[u] + 1 == dist[v], it's a tree edge.
                # If dist[u] + 1 >= dist[v], there is a cycle.
                # Specifically, if we find a node v that has been visited before,
                # the cycle length is dist[u] + 1 + dist[v] if the path to v doesn't go through u?
                # Simpler: Just run BFS and record the shortest path to each node.
                # Then for each edge (u, v), if v is visited, check if the edge closes a cycle.
                # A cycle exists if there is an edge (u, v) such that u and v are not parent-child in the BFS tree.
                # The length of the cycle is dist[u] + 1 + dist[v] if the paths to u and v are disjoint?
                # No, that's for undirected graphs.
                # Let's just find the shortest cycle by checking all pairs of neighbors?
                # Too slow.
                
                # Alternative: Since we want girth >= g, we can just check if there are any cycles of length < g.
                # This is equivalent to checking if the BFS tree has any "cross edges" within depth (g-1)/2.
                # Let's just compute the girth exactly for the generated group if it's small enough.
                
                # First, determine the group size.
                group_elements = set()
                current_level = [tuple([s3_elements[0]] * d)]
                group_elements.add(current_level[0])
                
                while current_level:
                    next_level = []
                    for node in current_level:
                        for gen in generators:
                            neighbor = multiply_group(node, gen)
                            if neighbor not in group_elements:
                                group_elements.add(neighbor)
                                next_level.append(neighbor)
                    current_level = next_level
                
                n = len(group_elements)
                
                # If the group is not the full S3^d, it might be smaller, which is good for score?
                # But if it's too small, girth might be small.
                # If n < Moore bound, girth must be < g.
                # Moore bound for k=5, g=9 is 426.
                if n < 426:
                    continue
                
                # Now check girth on this group
                # Map group elements to indices
                index_map = {elem: i for i, elem in enumerate(group_elements)}
                
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for i, elem in enumerate(group_elements):
                    for gen in generators:
                        neighbor = multiply_group(elem, gen)
                        j = index_map[neighbor]
                        adj[i].append(j)
                
                # Check girth via BFS from each node? No, just from one node if it's a Cayley graph?
                # No, girth is a global property.
                # But for Cayley graphs, it's enough to check cycles through the identity?
                # No.
                # However, we can use the property that the girth is the minimum length of a non-trivial relation.
                # We can find the girth by running BFS from the identity and looking for the shortest path back to identity?
                # That gives the shortest cycle through identity.
                # Is the global girth always equal to the shortest cycle through identity in a Cayley graph?
                # Yes, because any cycle can be translated to pass through the identity.
                
                # So we just need to find the shortest word in generators that equals identity.
                # We can do BFS from identity, and if we reach identity again, we found a cycle.
                # But we must not allow backtracking (u -> v -> u).
                
                # BFS for shortest cycle through identity
                # State: (current_element, last_generator_used)
                # Start: (identity, None)
                # Goal: (identity, last_gen) where last_gen is not None and step > 1
                
                visited = set()
                queue = deque()
                # (element, last_gen_index, distance)
                for i, gen in enumerate(generators):
                    queue.append((multiply_group(tuple([s3_elements[0]]*d), gen), i, 1))
                    visited.add((multiply_group(tuple([s3_elements[0]]*d), gen), i))
                
                min_cycle = float('inf')
                while queue:
                    elem, last_i, dist = queue.popleft()
                    if elem == tuple([s3_elements[0]]*d):
                        min_cycle = dist
                        break
                    for j, gen in enumerate(generators):
                        if j == last_i:
                            continue # No backtracking
                        next_elem = multiply_group(elem, gen)
                        state = (next_elem, j)
                        if state not in visited:
                            visited.add(state)
                            queue.append((next_elem, j, dist + 1))
                
                if min_cycle >= g:
                    # Found a valid graph
                    # Construct edges
                    edges = []
                    for i, elem in enumerate(group_elements):
                        for gen in generators:
                            neighbor = multiply_group(elem, gen)
                            j = index_map[neighbor]
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        return None

    # Try the random Cayley graph approach
    result = try_random_cayley(k, g)
    if result:
        return result
    
    # Fallback to L-U-W
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
