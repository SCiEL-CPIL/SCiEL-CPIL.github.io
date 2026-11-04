import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def luw_construct(k, g):
        # ... (existing LUW code) ...
        pass

    def check_girth(n, edges, g):
        # ... (existing check_girth code) ...
        pass

    def build_psl_cayley(q):
        # Build Cayley graph of PSL(2, q) with 3 involutions
        # Elements are matrices [[a, b], [c, d]] mod q with det=1
        # We represent them as tuples (a, b, c, d)
        
        # Generate all elements of PSL(2, q)
        # Actually, we don't need to generate all first, we can BFS from identity
        
        # Identity
        identity = (1, 0, 0, 1)
        
        # Involutions (Trace 0, Det 1)
        # M = [[a, b], [c, -a]] such that -a^2 + bc = 1 => bc - a^2 = 1
        # We need 3 distinct ones.
        
        # Standard involutions:
        # g1 = [[0, 1], [-1, 0]] -> a=0, b=1, c=-1, d=0. bc-a^2 = -1 - 0 = -1 != 1.
        # Wait, in SL(2, q), det = ad - bc.
        # For g1 = [[0, 1], [-1, 0]], det = 0 - (-1) = 1. Trace = 0.
        # So g1 is in SL(2, q). In PSL(2, q), it is an involution.
        
        # Let's define generators as matrices in SL(2, q)
        # g1 = (0, 1, -1, 0)
        # g2 = (1, 1, -1, -1) -> det = -1 - (-1) = 0? No.
        # det(g2) = 1*(-1) - 1*(-1) = 0. Not in SL.
        # Need det=1.
        # a=1, b=1. d=-1. c? 1*(-1) - 1*c = 1 => -1-c=1 => c=-2.
        # g2 = (1, 1, -2, -1).
        
        # g3: a=2. d=-2. 2*(-2) - b*c = 1 => -4 - bc = 1 => bc = -5.
        # Pick b=1, c=-5.
        # g3 = (2, 1, -5, -2).
        
        # Check if these are valid for all q?
        # They must be distinct in PSL(2, q).
        # And they must generate the group? Not strictly required for girth, but for connectivity.
        # If they don't generate, the graph is disconnected. We need a connected graph.
        # For PSL(2, q), any two non-commuting involutions usually generate.
        
        gens = [
            (0, 1, -1, 0),
            (1, 1, -2, -1),
            (2, 1, -5, -2)
        ]
        
        # Normalize to mod q
        gens = [tuple(x % q for x in gen) for gen in gens]
        
        # BFS to build graph
        # State: (a, b, c, d)
        # We need to track visited nodes and parent to check girth later?
        # Or just build the edge list and run check_girth.
        
        # Map from matrix tuple to index
        index = {identity: 0}
        order = [identity]
        edges = []
        
        # Queue for BFS
        queue = [identity]
        qi = 0
        
        while qi < len(queue):
            node = queue[qi]
            qi += 1
            idx = index[node]
            
            for gen in gens:
                # Multiply node * gen
                # [[a1, b1], [c1, d1]] * [[a2, b2], [c2, d2]]
                a1, b1, c1, d1 = node
                a2, b2, c2, d2 = gen
                
                a3 = (a1*a2 + b1*c2) % q
                b3 = (a1*b2 + b1*d2) % q
                c3 = (c1*a2 + d1*c2) % q
                d3 = (c1*b2 + d1*d2) % q
                
                new_node = (a3, b3, c3, d3)
                
                if new_node not in index:
                    index[new_node] = len(order)
                    order.append(new_node)
                    queue.append(new_node)
                
                # Add edge
                # Only add if not already added?
                # In undirected graph, (u, v) is same as (v, u).
                # We add edge (idx, index[new_node])
                # To avoid duplicates, we can just collect all and dedupe later,
                # or check if idx < index[new_node] ? No, BFS tree edges are unique.
                # But there might be non-tree edges.
                # We want the full graph.
                # So we add the edge every time we see it?
                # No, that creates multi-edges in the list.
                # We should store edges in a set.
                
                u = idx
                v = index[new_node]
                if u < v:
                    edges.append((u, v))
                else:
                    edges.append((v, u))
        
        # Deduplicate edges
        edges = list(set(edges))
        
        n = len(order)
        
        # Check if connected (BFS visited all)
        # If n < expected size, it's not connected or generators don't generate.
        # Expected size of PSL(2, q) is q(q^2-1)/2
        expected = q * (q*q - 1) // 2
        if n != expected:
            return None
            
        return n, edges

    # Main logic
    best_result = None
    best_n = float('inf')
    
    # Try PSL(2, q) for small primes
    for q in [11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline - 20:
            break
            
        result = build_psl_cayley(q)
        if result is not None:
            n, edges = result
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = result
                # If we found one close to Moore bound, we might stop?
                # But we want the smallest.
                # Since q increases n, we can break if we find a valid one?
                # No, larger q means larger n. We want smallest n.
                # So we should check smaller q first.
                # If q=11 works, n=660.
                # If q=13 works, n=1092.
                # We want the smallest valid n.
                # So we iterate q in increasing order and return the first valid one?
                # Yes, because n(q) is increasing with q.
                if best_result is not None:
                    break

    if best_result is not None:
        return best_result

    # Fallback to LUW
    if k != 3:
        return luw_construct(k, g)

    return luw_construct(k, g)
