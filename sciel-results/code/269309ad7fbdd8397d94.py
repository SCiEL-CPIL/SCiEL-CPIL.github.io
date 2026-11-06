"""Rewritten program: polarity graph of PG(2,5) + best single edge, then local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    q = 5
    n = 31
    
    # Enumerate points of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for t in range(q):
        points.append((1, t, 0))
    points.append((0, 1, 0))
    
    # Build polarity graph: edge if dot product is 0 mod q
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % q
            if dot == 0:
                edges.append((i, j))
    
    # Build adjacency as bitmasks for fast common-neighbor counting
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def common_count(a, b):
        """Count common neighbors of a and b using bitmasks."""
        return bin(adj[a] & adj[b]).count('1')
    
    def count_c4():
        """Count 4-cycles: for each pair (i,j), C(common(i,j), 2), sum, divide by 2."""
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = common_count(i, j)
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    # The polarity graph of PG(2,5) has 90 edges and is C4-free.
    # We need exactly 91 edges, so add one more edge.
    # Choose the edge that minimizes the number of new 4-cycles.
    # A new edge (u,v) creates C4s with every pair of common neighbors.
    # Since the base graph is C4-free, common_count(u,v) in the base graph
    # determines new C4s: C(common_count(u,v), 2).
    
    if len(edges) < 91:
        deficit = 91 - len(edges)
        best_edge = None
        best_c4 = float('inf')
        
        # Try all non-edges
        existing = set(edges)
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in existing:
                    c = common_count(u, v)
                    new_c4 = c * (c - 1) // 2
                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_edge = (u, v)
        
        if best_edge is not None:
            edges.append(best_edge)
            u, v = best_edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Now we have exactly 91 edges. Perform local search to reduce C4 count.
    current_c4 = count_c4()
    
    # Local search: try edge swaps (remove one, add another) to reduce C4
    # Use a smarter approach: for each edge, compute its "cost" in terms of C4 contribution
    # and try swapping high-cost edges with low-cost non-edges
    
    def edge_c4_contribution(u, v):
        """Estimate how many C4s contain edge (u,v).
        A C4 containing (u,v) has form u-w-x-v-u or u-x-w-v-u where w,x are common neighbors of u,v
        and also... actually let's just use the full count."""
        # Number of C4s containing edge (u,v) = sum over w in N(u)∩N(v) of |N(w)∩N(u)∩... |
        # Simpler: for each w in common(u,v), count x in N(w)∩N(u)∩N(v) \ {w} ... 
        # Actually: C4 through (u,v): pick w in N(u)∩N(v), then x must be in N(w)∩N(u)∩N(v) minus {w}
        # But that's not quite right either. Let me just compute it properly.
        # A 4-cycle containing edge (u,v): u-a-b-v-u where a in N(u)∩N(v)\{...} no.
        # 4-cycle: u, a, v, b, u where a in N(u)∩N(v), b in N(u)∩N(v), a≠b, and a-b not necessarily edge.
        # Wait: 4-cycle u-a-v-b-u requires: a~u, a~v, v~b, b~u, and a≠b, and u≠v.
        # So a and b are both in N(u)∩N(v), a≠b.
        # Number of such 4-cycles = C(|N(u)∩N(v)|, 2)
        # But this counts each C4 twice (once for each diagonal edge), and each C4 has 2 diagonal edges.
        # Actually no: for a fixed edge (u,v), the C4s containing it are determined by pairs of common neighbors.
        # Each pair {a,b} ⊂ N(u)∩N(v), a≠b, gives a C4 u-a-v-b-u (if a~v and b~u, which they are).
        # But we need to make sure this is a valid 4-cycle (no repeated vertices, which is satisfied).
        # So the number of C4s containing edge (u,v) is C(|N(u)∩N(v)|, 2).
        c = common_count(u, v)
        return c * (c - 1) // 2
    
    def try_improve():
        """Try to find an improving edge swap. Returns new C4 count or None."""
        nonlocal current_c4
        if time.time() > deadline:
            return None
        
        # Compute C4 contribution for each edge
        edge_costs = {}
        for idx, (u, v) in enumerate(edges):
            edge_costs[idx] = edge_c4_contribution(u, v)
        
        # Sort edges by cost descending
        sorted_edges = sorted(edge_costs.items(), key=lambda x: -x[1])
        
        existing = set(edges)
        
        # Try removing high-cost edges and adding low-cost non-edges
        for idx, cost in sorted_edges[:20]:  # Try top 20 most expensive edges
            if time.time() > deadline:
                return None
            u, v = edges[idx]
            
            # Temporarily remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop(idx)
            existing.discard((u, v))
            
            # Try adding various non-edges
            best_new_c4 = current_c4
            best_add = None
            
            # Sample some non-edges to try
            random.shuffle(range(n))
            tried = 0
            for a in range(n):
                for b in range(a + 1, n):
                    if (a, b) not in existing and (b, a) not in existing:
                        # Check if this is a non-edge
                        if not (adj[a] >> b) & 1:
                            # Estimate new C4s
                            c = common_count(a, b)
                            new_c4_est = current_c4 - cost + c * (c - 1) // 2
                            if new_c4_est < best_new_c4:
                                # Verify with actual count
                                adj[a] |= (1 << b)
                                adj[b] |= (1 << a)
                                actual_c4 = count_c4()
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                if actual_c4 < best_new_c4:
                                    best_new_c4 = actual_c4
                                    best_add = (a, b)
                            tried += 1
                            if tried >= 300:
                                break
                if tried >= 300:
                    break
            
            if best_add is not None and best_new_c4 < current_c4:
                a, b = best_add
                edges.append((a, b))
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                existing.add((a, b))
                current_c4 = best_new_c4
                return current_c4
            else:
                # Restore
                edges.insert(idx, (u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                existing.add((u, v))
        
        return None
    
    # Run local search
    no_improve = 0
    while time.time() < deadline:
        result = try_improve()
        if result is not None:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                break
    
    # Return edges as list of lists
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
