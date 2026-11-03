import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # LUW construction for q=4, d=5 as fallback (2048 vertices, girth >= 10)
    q = 4
    d = 5
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def gf_add(x, y): return x ^ y
    def gf_sub(x, y): return x ^ y
    def gf_mul(x, y): return table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    best_n = len(order)
    best_edges = edges

    if k == 4 and g == 10:
        # Base graph: K6 minus perfect matching
        # Vertices 0..5
        # Edges:
        # (0,1), (0,2), (0,4), (0,5)
        # (1,2), (1,3), (1,5)
        # (2,3), (2,4)
        # (3,4), (3,5)
        # (4,5)
        # Wait, let's list them systematically.
        # Pairs (i,j) with i<j, not in {(0,3),(1,4),(2,5)}
        base_edges = []
        pm = {(0,3), (1,4), (2,5)}
        for i in range(6):
            for j in range(i+1, 6):
                if (i,j) not in pm:
                    base_edges.append((i, j))
        
        # Map edge tuple to index
        edge_index = {}
        for idx, e in enumerate(base_edges):
            edge_index[e] = idx
            edge_index[(e[1], e[0])] = idx
        
        # Precompute all closed non-backtracking walks of length 3 to 9
        # Walk is a sequence of vertices v0, v1, ..., vL such that v0=vL, 
        # vi != vi+2, and (vi, vi+1) is an edge.
        # We represent the walk by the sequence of edge indices and directions.
        # Coefficient for edge e is +1 if traversed u->v (canonical), -1 if v->u.
        
        # Build adjacency list for base
        base_adj = [[] for _ in range(6)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
            
        walks = [] # List of np.array of shape (E,)
        
        # DFS to find closed NB walks
        # State: (prev, curr, length, path_edge_coeffs)
        # We start from each vertex? No, we can just find all cycles.
        # To avoid duplicates, we can enforce v0 < v1 < ... < vmax? 
        # Or just collect all and deduplicate?
        # Number of walks is small.
        
        # Let's do a DFS from each start node, but only record if start is the min vertex in the cycle?
        # Actually, for voltage check, direction matters.
        # A cycle and its reverse have opposite voltage sums.
        # If sum != 0, -sum != 0.
        # So checking one direction is enough if we check != 0.
        
        def find_walks():
            found = set()
            for start in range(6):
                # DFS
                # stack: (prev, curr, length, coeffs_list)
                # coeffs_list is list of (edge_idx, sign)
                stack = []
                for nb in base_adj[start]:
                    e_idx = edge_index[(start, nb)]
                    sign = 1 if (start, nb) == base_edges[e_idx] else -1
                    stack.append((start, nb, 1, [(e_idx, sign)]))
                
                while stack:
                    prev, curr, length, coeffs = stack.pop()
                    if length >= 3:
                        if curr == start:
                            # Closed walk
                            # Check if it's non-backtracking? 
                            # Our DFS naturally avoids backtracking to prev.
                            # But does it avoid other backtracking?
                            # "Non-backtracking" means you don't go back to the vertex you came from.
                            # The DFS condition `if nb == prev: continue` ensures this.
                            # However, a walk like A-B-C-B-A is not allowed.
                            # Our DFS:
                            # A->B (prev A, curr B)
                            # B->C (prev B, curr C)
                            # C->B (prev C, curr B) -- Wait, C's neighbors include B.
                            # If we allow C->B, then B->A.
                            # This is a backtracking walk?
                            # Standard definition: No edge is traversed in both directions.
                            # Or: No vertex is visited twice? No, that's simple cycle.
                            # Non-backtracking walk: $v_{i-1} \neq v_{i+1}$.
                            # So C->B is allowed if B != C's previous?
                            # In A-B-C-B, the step C->B goes to B. Previous was C? No, previous was C?
                            # Path: A, B, C, B.
                            # Step 1: A->B. Prev A, Curr B.
                            # Step 2: B->C. Prev B, Curr C.
                            # Step 3: C->B. Prev C, Curr B.
                            # Is this non-backtracking?
                            # At C, we came from B. We go to B.
                            # This is backtracking.
                            # So we must ensure `nb != prev` in the DFS.
                            # My DFS above does `for nb in base_adj[curr]: if nb == prev: continue`.
                            # So C->B is blocked because prev is B?
                            # Wait. In state (prev=B, curr=C), we iterate neighbors of C.
                            # If nb == B, and prev == B, we skip.
                            # So yes, it prevents immediate backtracking.
                            
                            # Convert coeffs to array
                            arr = np.zeros(len(base_edges), dtype=np.int8)
                            for e_idx, sign in coeffs:
                                arr[e_idx] += sign
                            # Deduplicate? 
                            # The same cycle can be found from different starts or rotations.
                            # But the voltage sum is the same (cyclic shift doesn't change sum).
                            # Reversal changes sign.
                            # We just need to ensure we check all unique cycles.
                            # Using a set of tuples of (e_idx, sign) might be large.
                            # But E=12, length<=9.
                            # Let's just add to list and dedupe later?
                            # Or just add. Duplicates don't hurt correctness, just speed.
                            walks.append(arr)
                            
                    if length < 9:
                        for nb in base_adj[curr]:
                            if nb == prev:
                                continue
                            e_idx = edge_index[(curr, nb)]
                            sign = 1 if (curr, nb) == base_edges[e_idx] else -1
                            stack.append((curr, nb, length + 1, coeffs + [(e_idx, sign)]))
        
        find_walks()
        
        # Deduplicate walks to speed up check
        # Two walks are equivalent if one is a rotation of the other?
        # For voltage sum, rotation doesn't matter.
        # We can normalize by finding the lexicographically smallest rotation?
        # Or just use a set of tuples.
        unique_walks = set()
        for w in walks:
            unique_walks.add(tuple(w))
        
        if unique_walks:
            walk_matrix = np.array(list(unique_walks)) # Shape (K, E)
            K = len(walk_matrix)
        else:
            walk_matrix = None
            K = 0
            
        # Try different M values
        # Smaller M -> smaller n.
        # Start from small M.
        for M in range(30, 100):
            if time.time() > deadline - 10.0:
                break
            n_total = 6 * M
            if n_total >= best_n:
                continue
            
            remaining = deadline - 15.0 - time.time()
            if remaining <= 0:
                break
            
            # Scale attempts with time
            max_attempts = min(5000, max(100, int(remaining * 50)))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                # Random voltages
                V = np.random.randint(0, M, size=len(base_edges))
                
                if K > 0:
                    # Compute sums
                    # Sums = walk_matrix @ V % M
                    # If any sum is 0, fail.
                    sums = (walk_matrix @ V) % M
                    if 0 in sums:
                        continue
                
                # If we get here, girth is likely >= 10 (assuming no walks of length <3, which is true for simple graph)
                # Wait, we checked lengths 3..9.
                # What about length 1, 2?
                # Simple graph has no loops (len 1) or multi-edges (len 2).
                # So girth >= 3 is guaranteed.
                
                # Build the lifted graph
                adj = [[] for _ in range(n_total)]
                for e_idx, (u, v) in enumerate(base_edges):
                    vol = V[e_idx]
                    for w in range(M):
                        a = u * M + w
                        b = v * M + ((w + vol) % M)
                        adj[a].append(b)
                        adj[b].append(a)
                
                # Verify 4-regularity and simplicity
                # Since base is simple and lift is regular, it should be fine.
                # But we must ensure no multi-edges in lift.
                # This happens if two different base edges connect the same lifted vertices.
                # Or if a base edge creates a loop? No, u!=v.
                # Multi-edges in lift:
                # If (u,v) and (u',v') are different base edges, can they map to same (a,b)?
                # a = u*M + w. b = v*M + (w+vol)%M.
                # If u != u', then a != a' (unless M=1, but M>=30).
                # So no multi-edges from different base edges.
                # What about same base edge?
                # w1 != w2 -> a1 != a2.
                # So simple.
                
                # Verify degree
                # Each vertex has degree 4 in base, so degree 4 in lift.
                
                # Update best
                if n_total < best_n:
                    best_n = n_total
                    edge_set = set()
                    for i in range(n_total):
                        for j in adj[i]:
                            if i < j:
                                edge_set.add((i, j))
                    best_edges = list(edge_set)
                    break # Found valid for this M, try smaller M? No, M is increasing.
                    # Wait, I loop M from 30 to 100.
                    # If I find one for M=30, n=180.
                    # I should break the M loop?
                    # No, I want the *smallest* n.
                    # Smallest n corresponds to smallest M.
                    # So if I find one for M=30, that's the best possible for this base.
                    # I can break the M loop.
                    break

    return best_n, best_edges
