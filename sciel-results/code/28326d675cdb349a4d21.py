"""Combined program: construct k-regular graph of girth g using random Cayley graphs on PSL(2,q) with L-U-W fallback."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Try to find a Cayley graph on PSL(2, q) for small q with sufficient order.
    2. Fall back to the Lazebnik-Ustimenko-Woldar construction if random search fails.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Fallback: L-U-W construction
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

    # Random Cayley Graph on PSL(2, q)
    def try_psl2_cayley(k, g, max_time=100):
        """
        Attempt to find a k-regular Cayley graph on PSL(2, q) with girth >= g.
        """
        start_time = time.time()
        
        # We need a group large enough.
        # Order of PSL(2, q) is q(q^2-1)/2 for q odd.
        # We need n >= moore_bound.
        
        # Try q = 11, 13, 17, 19, 23, etc.
        # For k=5, g=9, mb=428.
        # q=11 -> n=660.
        # q=13 -> n=1092.
        
        for q in [11, 13, 17, 19, 23]:
            if (q * (q*q - 1)) // 2 < mb:
                continue
                
            # Generate PSL(2, q) elements
            # Elements are 2x2 matrices with determinant 1, modulo q.
            # We represent them as tuples (a, b, c, d) for [[a,b],[c,d]].
            # Identity: (1, 0, 0, 1)
            
            def mat_mul(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q, 
                        (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
            
            def mat_inv(m):
                a, b, c, d = m
                # det = 1
                return (d, -b, -c, a)
            
            # Find involutions: m^2 = I
            involutions = []
            # Iterate over possible matrices
            # a^2 + bc = 1
            # d = (1 - a^2 - bc) / c ? No, det=1 => ad-bc=1.
            # For involution: m^2 = I => (a+d)^2 + 4bc = 0 ?
            # In PSL(2, q), an element is an involution if its trace is 0 (for q odd, char != 2).
            # Trace = a + d. So d = -a.
            # det = a(-a) - bc = -a^2 - bc = 1 => a^2 + bc = -1.
            
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # a^2 + bc = -1 mod q
                        val = (a*a + b*c) % q
                        if val == (q - 1) % q:
                            d = (-a) % q
                            m = (a, b, c, d)
                            # Check if it's identity (it shouldn't be if trace 0 and q>2)
                            # Check if m^2 = I
                            m2 = mat_mul(m, m)
                            if m2 == (1, 0, 0, 1):
                                # Avoid duplicates? PSL identifies m and -m.
                                # We can just use the representative with a > 0 or a=0, b>0 etc.
                                # For simplicity, just add if not in list (small q)
                                if m not in involutions:
                                    involutions.append(m)
            
            # Filter out identity just in case
            involutions = [m for m in involutions if m != (1, 0, 0, 1)]
            
            if len(involutions) < k:
                continue
                
            # We need to select k involutions that generate the group.
            # Random sampling
            best_graph = None
            best_n = float('inf')
            
            # Limit iterations to fit in time
            max_iters = 500
            for _ in range(max_iters):
                if time.time() - start_time > max_time:
                    break
                
                gens = random.sample(involutions, k)
                
                # Check if they generate the group
                # BFS from identity using these generators
                visited = set()
                queue = deque()
                queue.append((1, 0, 0, 1))
                visited.add((1, 0, 0, 1))
                parent = {(1, 0, 0, 1): None}
                
                # We also need to check girth.
                # Non-backtracking BFS
                # State: (node, prev_node)
                # Start: (I, None)
                
                # Actually, we can combine generation check and girth check.
                # If the graph is connected, it's a Cayley graph on the subgroup generated.
                # We want the whole group, so we check if |visited| == |G|.
                
                # Girth check:
                # BFS from (I, None).
                # If we reach (I, prev) where prev != None and depth < g, we have a cycle.
                # Wait, a cycle of length L corresponds to a path of length L from I to I.
                # Non-backtracking means we don't go I -> g1 -> I.
                # So we look for the shortest non-backtracking closed walk.
                
                # Let's do a standard BFS on the state space (node, prev_gen_index)
                # to find the shortest cycle.
                
                # However, for large groups, this might be slow.
                # Let's first check if the generators produce a large enough subgroup.
                
                # Simple BFS to get subgroup size
                sub_group = set([(1, 0, 0, 1)])
                queue = deque([(1, 0, 0, 1)])
                while queue:
                    curr = queue.popleft()
                    for gen in gens:
                        next_node = mat_mul(curr, gen)
                        if next_node not in sub_group:
                            sub_group.add(next_node)
                            queue.append(next_node)
                            # Also add inverse? Generators are involutions, so inv = gen.
                            # But we need to close under multiplication.
                            # Since gens are involutions, Cay(G, S) is undirected.
                            # The subgroup is generated by S.
                            
                if len(sub_group) < mb:
                    continue
                
                # If subgroup is large enough, check girth.
                # We only need to check on the generated subgroup if it's the whole group,
                # or if the subgroup itself has girth >= g.
                # But we want a graph on the *whole* group ideally, or at least a valid cage.
                # If the generated subgroup is proper but has girth >= g and size >= mb, it's a valid cage.
                # However, Cayley graphs are usually defined on the whole group.
                # If S doesn't generate G, the graph is disconnected.
                # The task requires a simple graph, usually connected? "The graph must be simple... every vertex degree k".
                # It doesn't explicitly say connected, but "cage" implies connected.
                # So we require S to generate the group (or the subgroup we use).
                
                # Let's assume we use the generated subgroup H.
                # If H is proper, we get a disconnected graph on G? No, we construct the graph on H.
                # So we need to map H to 0..|H|-1.
                
                # Check girth of Cay(H, S)
                # BFS from (I, None)
                # State: (node, prev_node)
                # Start: (I, None)
                # Visited states: set of (node, prev_node)
                
                # We want to find if there is a cycle of length < g.
                # A cycle of length L is a sequence g1...gL such that g1...gL = I and no backtracking.
                # Backtracking: gi != g_{i-1}^{-1}. Since involutions, gi != g_{i-1}.
                
                # BFS on (node, last_gen)
                # Start: (I, -1)
                # Expand: (node, last_gen) -> (node * gen, gen) for gen in S, gen != last_gen.
                # If we reach (I, gen) with depth < g, we found a cycle of length depth.
                # Actually, if we reach (I, gen) at depth d, the cycle length is d.
                # Wait, if we start at I, go to g1, ..., gd=I.
                # The path is I -> g1 -> ... -> I.
                # The last step was from g_{d-1} to I via gen g_d.
                # So state is (I, g_d).
                # We need d >= g.
                
                visited_states = set()
                queue = deque()
                queue.append(((1, 0, 0, 1), -1, 0)) # (node, last_gen_idx, depth)
                visited_states.add(((1, 0, 0, 1), -1))
                
                found_short_cycle = False
                while queue:
                    node, last_idx, depth = queue.popleft()
                    
                    if depth >= g:
                        # We are safe for this path, but we need to ensure NO path has length < g.
                        # If we finish BFS without finding (I, gen) at depth < g, girth is >= g.
                        pass
                    
                    for i, gen in enumerate(gens):
                        if i == last_idx:
                            continue
                        next_node = mat_mul(node, gen)
                        next_state = (next_node, i)
                        
                        if next_state in visited_states:
                            continue
                            
                        visited_states.add(next_state)
                        
                        # Check if we hit Identity with a non-zero depth
                        if next_node == (1, 0, 0, 1):
                            # Cycle found of length depth + 1
                            if depth + 1 < g:
                                found_short_cycle = True
                                break
                        else:
                            queue.append((next_node, i, depth + 1))
                    
                    if found_short_cycle:
                        break
                        
                if found_short_cycle:
                    continue
                
                # If we get here, girth is >= g.
                # Construct the graph on the subgroup H.
                # We need to list all elements of H.
                # We already have sub_group from the previous BFS?
                # No, that BFS was just for size. Let's re-do it or store it.
                # Actually, the girth BFS visited states, not just nodes.
                # Let's just use the sub_group calculated earlier.
                
                # Map elements of H to indices
                h_list = list(sub_group)
                h_index = {h: i for i, h in enumerate(h_list)}
                n = len(h_list)
                
                edges = []
                for i, h in enumerate(h_list):
                    for gen in gens:
                        next_h = mat_mul(h, gen)
                        j = h_index[next_h]
                        if i < j:
                            edges.append((i, j))
                            
                # Check if n < best_n
                if n < best_n:
                    best_n = n
                    best_graph = (n, edges)
                    
            if best_graph:
                return best_graph
                
    # Try the random Cayley approach
    result = try_psl2_cayley(k, g, max_time=100)
    if result:
        return result
        
    # Fallback
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
