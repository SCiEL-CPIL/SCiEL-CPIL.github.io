import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    num_edges = len(all_edges)  # 120
    
    # Q3 structure: 8 vertices, 12 edges
    # Q3 is the cube graph: vertices 0-7, edges between vertices differing in exactly one bit
    q3_vertices = list(range(8))
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff & (diff - 1) == 0:  # power of 2, differ in one bit
                q3_edges.append((i, j))
    # q3_edges has 12 edges
    
    q3_adj = {v: [] for v in range(8)}
    for u, v in q3_edges:
        q3_adj[u].append(v)
        q3_adj[v].append(u)
    
    # Precompute: for each pair of Q3 edges (i,j), determine if they can coexist in Q3
    # and what constraints they impose
    # For fast detection: given a graph with adj_bits, check if edge (u,v) creates a Q3
    
    # Strategy: Precompute all 8! / 8 = 5040 distinct Q3 embeddings up to symmetry? 
    # Actually, there are many more. Let's use a different approach.
    
    # For each potential Q3 embedding (mapping of Q3 vertices to graph vertices),
    # we can precompute the set of 12 edges it requires.
    # But 16P8 is too large.
    
    # Better: For each edge (u,v) in the graph, check if it can be part of a Q3.
    # A Q3 containing edge (u,v) means: there exists an embedding f: Q3 -> G such that
    # {f(a), f(b)} = {u, v} for some edge (a,b) in Q3.
    
    # Fast check for "does adding (u,v) create a Q3":
    # We need to find 6 more vertices w1..w6 such that together with u,v they form Q3.
    # The structure: u and v are adjacent in Q3. Their common neighbors in Q3: 
    # each has 2 other neighbors, and those 4 + u + v form a 6-cycle with 2 chords? No.
    # In Q3, edge (a,b): a has neighbors {b, a1, a2}, b has neighbors {a, b1, b2}
    # a1 and b1 are adjacent (form a 4-cycle a-b-b1-a1-a), same for a2,b2.
    # And a1 is NOT adjacent to b2, a2 NOT adjacent to b1.
    # The 8th vertex c is adjacent to a1, a2, b1, b2.
    
    # So to check if (u,v) is in a Q3:
    # 1. Find common neighbors of u and v in G: these could be {a1,b1} or {a2,b2} or {c}
    #    Wait, in Q3, u and v share NO common neighbors (it's bipartite, no triangles).
    #    Actually Q3 IS bipartite, so u and v (adjacent) have no common neighbors.
    #    The 4-cycle: u - a1 - b1 - v - u? No: u-a1, a1-b1, b1-v, v-u. Yes that's a 4-cycle.
    #    So a1 and b1 are both neighbors of u AND v? No!
    #    Let me re-examine: Q3 vertices as 3-bit strings.
    #    000-001: 000 neighbors: 001,010,100. 001 neighbors: 000,011,101.
    #    Common neighbors of 000 and 001: none (000's other neighbors are 010,100; 001's are 011,101).
    #    4-cycle: 000-010-011-001-000. So 010 and 011 are the "other" vertices.
    #    010 is neighbor of 000 and 011. 011 is neighbor of 010 and 001.
    #    So the 4-cycle is u - x - y - v - u where x is neighbor of u only (among {x,y}), y neighbor of v only.
    #    Actually x is neighbor of u and y, y is neighbor of v and x. x is NOT neighbor of v, y NOT neighbor of u.
    
    # So: for edge (u,v) to be in a Q3, we need:
    # - Two vertices x,y such that: x~u, x~y, y~v, and x NOT~ v, y NOT~ u
    #   (forming 4-cycle u-x-y-v-u)
    # - Another such pair x',y' (the other 4-cycle through u,v)
    # - An 8th vertex z adjacent to x, x', y, y'
    # - And z NOT adjacent to u or v
    
    # Let me verify: Q3 = 000,001,010,011,100,101,110,111
    # Edge 000-001. 
    # 4-cycles through this edge: 
    #   000-010-011-001-000 (x=010, y=011)
    #   000-100-101-001-000 (x=100, y=101)
    # 8th vertex: 110. Adjacent to 010,100,011,101? 
    #   110~010 (differ in bit 2? 110 vs 010: bit 0 differs. Yes)
    #   110~100 (110 vs 100: bit 1 differs. Yes)
    #   110~011 (110 vs 011: bits 0 and 1 differ. No! Not adjacent.)
    # Hmm, let me recount. 110 in binary: bits 2,1 set (value 6). 
    # 010 = 2, 100 = 4, 011 = 3, 101 = 5.
    # 110(6) vs 010(2): 6^2=4, power of 2? 4=2^2. Yes, adjacent.
    # 110(6) vs 100(4): 6^4=2, power of 2? Yes. Adjacent.
    # 110(6) vs 011(3): 6^3=5, not power of 2. NOT adjacent.
    # 110(6) vs 101(5): 6^5=3, not power of 2. NOT adjacent.
    # So 110 is adjacent to 010 and 100 but NOT to 011 and 101.
    # 
    # I think I mislabeled. Let me redo:
    # The two 4-cycles: (000,010,011,001) and (000,100,101,001).
    # The 8th vertex 110 should be adjacent to one vertex from each 4-cycle (not u or v).
    # 110 adjacent to 010 (from first cycle) and 100 (from second cycle).
    # So z=110 is adjacent to x=010 and x'=100, where x,x' are the "u-side" vertices.
    
    # So the check for edge (u,v) in Q3:
    # Find all 4-cycles u-x-y-v where x~u, x~y, y~v, x!~v, y!~u.
    # Need at least 2 such cycles (giving pairs (x1,y1) and (x2,y2)).
    # Then check if there's a vertex z adjacent to x1 and x2 (or x1 and y2, etc.) 
    # and not adjacent to u or v, and the full structure works.
    
    # This is getting complex. Let me just use a direct backtracking check that's fast enough.
    
    def creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) to graph with adj_bits creates a Q3 subgraph."""
        # u and v are the new edge. Check all Q3 edges as potential image of (u,v).
        for (a, b) in q3_edges:
            # Try mapping a->u, b->v
            if _check_partial(a, b, u, v, adj_bits):
                return True
            # Try mapping a->v, b->u
            if _check_partial(a, b, v, u, adj_bits):
                return True
        return False
    
    def _check_partial(qa, qb, gu, gv, adj_bits):
        """Check if there's a Q3 embedding with qa->gu, qb->gv."""
        mapping = [-1] * 8
        mapping[qa] = gu
        mapping[qb] = gv
        used = [False] * n
        used[gu] = True
        used[gv] = True
        
        # Order remaining Q3 vertices by degree of constraints
        remaining = [x for x in range(8) if x != qa and x != qb]
        # Sort by number of already-mapped neighbors (more constrained first)
        def constraint_count(qv):
            return sum(1 for nb in q3_adj[qv] if mapping[nb] != -1)
        remaining.sort(key=constraint_count, reverse=True)
        
        def backtrack(idx):
            if idx == len(remaining):
                # Verify all Q3 edges exist in G
                for (x, y) in q3_edges:
                    gx, gy = mapping[x], mapping[y]
                    if not (adj_bits[gx] & (1 << gy)):
                        return False
                return True
            
            qv = remaining[idx]
            # Candidate graph vertices
            for gc in range(n):
                if used[gc]:
                    continue
                # Check all mapped neighbors of qv
                ok = True
                for qnb in q3_adj[qv]:
                    gnb = mapping[qnb]
                    if gnb != -1:
                        if not (adj_bits[gc] & (1 << gnb)):
                            ok = False
                            break
                if not ok:
                    continue
                # Also: gc must not be gu or gv (already in used)
                mapping[qv] = gc
                used[gc] = True
                if backtrack(idx + 1):
                    return True
                mapping[qv] = -1
                used[gc] = False
            return False
        
        return backtrack(0)
    
    def edge_count(adj_bits):
        return sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    def is_q3_free(adj_bits):
        """Verify the graph is Q3-free."""
        # Check all 120 edges
        for (u, v) in all_edges:
            if adj_bits[u] & (1 << v):
                # Temporarily remove edge and check
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                if creates_q3(u, v, adj_bits):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    return False
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        return True
    
    def greedy_complete(adj_bits):
        """Try to add all possible missing edges greedily."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for (u, v) in all_edges:
                if not (adj_bits[u] & (1 << v)):
                    if not creates_q3(u, v, adj_bits):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        changed = True
        return adj_bits
    
    def simulated_annealing(adj_bits, max_iters=2000):
        """Simulated annealing with edge swaps."""
        if time.time() > deadline:
            return adj_bits, edge_count(adj_bits)
        
        current = adj_bits[:]
        current_count = edge_count(current)
        best = current[:]
        best_count = current_count
        
        T = 2.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Propose: remove a random existing edge, add a random missing edge
            existing = []
            missing = []
            for (u, v) in all_edges:
                if current[u] & (1 << v):
                    existing.append((u, v))
                else:
                    missing.append((u, v))
            
            if not existing or not missing:
                break
            
            # Remove random existing edge
            ru, rv = random.choice(existing)
            current[ru] &= ~(1 << rv)
            current[rv] &= ~(1 << ru)
            current_count -= 1
            
            # Add random missing edge
            au, av = random.choice(missing)
            if not creates_q3(au, av, current):
                current[au] |= (1 << av)
                current[av] |= (1 << au)
                current_count += 1
            else:
                # Revert the addition (keep removal)
                pass
            
            delta = current_count - best_count
            
            if current_count > best_count:
                best = current[:]
                best_count = current_count
            elif delta >= 0 or random.random() < math.exp(delta / T):
                # Accept
                pass
            else:
                # Reject: revert
                current[au] &= ~(1 << av)
                current[av] &= ~(1 << au)
                current_count -= 1
                current[ru] |= (1 << rv)
                current[rv] |= (1 << ru)
                current_count += 1
            
            T *= cooling
            if T < 0.01:
                break
        
        return best, best_count
    
    def local_search_add_only(adj_bits):
        """Try to add any missing edge that doesn't create Q3."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for (u, v) in all_edges:
                if not (adj_bits[u] & (1 << v)):
                    if not creates_q3(u, v, adj_bits):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        changed = True
        return adj_bits
    
    # Multi-start with different initial structures
    best_adj = [0] * n
    best_count = 0
    
    num_starts = 15
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        random.seed(1000 + s * 7919)
        
        adj = [0] * n
        
        if s % 3 == 0:
            # Start with K_{3,13}
            for u in range(3):
                for v in range(3, 16):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif s % 3 == 1:
            # Start with K_{4,12}
            for u in range(4):
                for v in range(4, 16):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        else:
            # Start with random dense graph (p=0.5)
            for (u, v) in all_edges:
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Greedily complete
        adj = greedy_complete(adj)
        count = edge_count(adj)
        
        # Local search (add only)
        adj = local_search_add_only(adj)
        count = edge_count(adj)
        
        # Simulated annealing
        adj, count = simulated_annealing(adj, max_iters=1500)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Final greedy completion
    best_adj = local_search_add_only(best_adj)
    best_count = edge_count(best_adj)
    
    # Build result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
