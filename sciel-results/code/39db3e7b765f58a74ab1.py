import random
import time

def construct():
    # Try to find n=30 using Local Search
    try:
        result = _find_30()
        if result is not None:
            return {'n': 30, 'colors': result}
    except Exception:
        pass

    # Fallback to n=29
    n = 29
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

def _find_30():
    n = 30
    num_edges = n * (n - 1) // 2
    start_time = time.time()
    limit = 105 # Leave buffer
    
    # Precompute edge endpoints
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Precompute for each edge, the list of other edges that form a triangle with it
    # Triangle (i,j,k) involves edges (i,j), (i,k), (j,k)
    # For a given edge (i,j), the third vertex k can be any other vertex.
    # The other two edges are (i,k) and (j,k).
    
    # We need a fast way to get the index of an edge (u,v)
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # For each edge e=(u,v), store list of (e1, e2) such that e, e1, e2 form a triangle
    # e1 = (u,k), e2 = (v,k)
    triangle_neighbors = [[] for _ in range(num_edges)]
    for k in range(n):
        for i in range(n):
            if i == k: continue
            for j in range(n):
                if j == k or j == i: continue
                # Triangle i, j, k
                e1 = (i, j)
                e2 = (i, k)
                e3 = (j, k)
                
                idx1 = edge_index.get(e1)
                if idx1 is None: continue
                # Add (e2, e3) to triangle_neighbors[idx1]
                idx2 = edge_index.get(e2)
                idx3 = edge_index.get(e3)
                if idx2 is not None and idx3 is not None:
                    triangle_neighbors[idx1].append((idx2, idx3))
                    
    def get_cost(colors):
        count = 0
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            for idx2, idx3 in triangle_neighbors[idx]:
                if colors[idx2] == c and colors[idx3] == c:
                    count += 1
        return count // 3 # Each triangle counted 3 times
        
    def has_k222(colors):
        # Build adjacency matrices
        A0 = [[0]*n for _ in range(n)]
        A1 = [[0]*n for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            if colors[idx] == 0:
                A0[u][v] = 1
                A0[v][u] = 1
            else:
                A1[u][v] = 1
                A1[v][u] = 1
                
        # Check for K2,2,2 in A0 and A1
        # K2,2,2 is a subgraph if there are 3 disjoint edges (a,b), (c,d), (e,f)
        # such that all cross edges are present.
        # This is equivalent to:
        # There exist 6 distinct vertices v1..v6 such that
        # G contains a subgraph isomorphic to K2,2,2.
        # We can iterate over all 6-subsets.
        from itertools import combinations
        for verts in combinations(range(n), 6):
            # Check if induced subgraph on verts contains K2,2,2 in color 0
            # Or color 1
            # We can just check if the subgraph induced by verts in A0 contains K2,2,2
            # And same for A1
            
            # To speed up, we can check if the degree sequence or something matches?
            # No, just check.
            # There are 15 partitions of 6 vertices into 3 pairs.
            # For each partition, check if all 12 cross edges are 1 in A0.
            
            # Generate partitions
            # This is a bit slow in Python if done naively.
            # But C(30,6) is ~600k. 15 partitions * 12 checks = 180 ops.
            # 600k * 180 = 100M ops. Might be slow.
            
            # Optimization:
            # If the number of edges in the induced subgraph is less than 12, it can't contain K2,2,2.
            
            def check_subgraph(A, verts):
                # Map verts to 0..5
                # Build 6x6 matrix
                # Count edges
                edge_count = 0
                for i in range(6):
                    for j in range(i+1, 6):
                        if A[verts[i]][verts[j]]:
                            edge_count += 1
                if edge_count < 12:
                    return False
                
                # Check all 15 partitions
                # Partitions of 6 into 3 pairs:
                # (0,1), (2,3), (4,5)
                # (0,1), (2,4), (3,5)
                # ...
                
                # Hardcode the 15 partitions of indices 0..5
                partitions = [
                    ((0,1),(2,3),(4,5)),
                    ((0,1),(2,4),(3,5)),
                    ((0,1),(2,5),(3,4)),
                    ((0,2),(1,3),(4,5)),
                    ((0,2),(1,4),(3,5)),
                    ((0,2),(1,5),(3,4)),
                    ((0,3),(1,2),(4,5)),
                    ((0,3),(1,4),(2,5)),
                    ((0,3),(1,5),(2,4)),
                    ((0,4),(1,2),(3,5)),
                    ((0,4),(1,3),(2,5)),
                    ((0,4),(1,5),(2,3)),
                    ((0,5),(1,2),(3,4)),
                    ((0,5),(1,3),(2,4)),
                    ((0,5),(1,4),(2,3))
                ]
                
                for p in partitions:
                    ok = True
                    for pair in p:
                        u, v = pair
                        # Check all edges from u to other pairs
                        # Actually, we need to check all 12 cross edges.
                        # Edges between pair 1 and pair 2
                        # Edges between pair 2 and pair 3
                        # Edges between pair 3 and pair 1
                        
                        p1, p2, p3 = p
                        # Check p1-p2
                        if not (A[verts[p1[0]]][verts[p2[0]]] and A[verts[p1[0]]][verts[p2[1]]] and
                                A[verts[p1[1]]][verts[p2[0]]] and A[verts[p1[1]]][verts[p2[1]]]):
                            ok = False
                            break
                        if not ok: break
                        
                        # Check p2-p3
                        if not (A[verts[p2[0]]][verts[p3[0]]] and A[verts[p2[0]]][verts[p3[1]]] and
                                A[verts[p2[1]]][verts[p3[0]]] and A[verts[p2[1]]][verts[p3[1]]]):
                            ok = False
                            break
                        if not ok: break
                        
                        # Check p3-p1
                        if not (A[verts[p3[0]]][verts[p1[0]]] and A[verts[p3[0]]][verts[p1[1]]] and
                                A[verts[p3[1]]][verts[p1[0]]] and A[verts[p3[1]]][verts[p1[1]]]):
                            ok = False
                            break
                        if not ok: break
                        
                    if ok:
                        return True
                return False
                
            if check_subgraph(A0, verts):
                return True
            if check_subgraph(A1, verts):
                return True
                
        return False

    # Local Search
    for attempt in range(50):
        if time.time() - start_time > limit:
            break
            
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        cost = get_cost(colors)
        
        for step in range(2000):
            if time.time() - start_time > limit:
                return None
                
            idx = random.randint(0, num_edges - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            colors[idx] = new_val
            
            # Calculate cost change
            # Only triangles involving edge idx change
            delta = 0
            for idx2, idx3 in triangle_neighbors[idx]:
                # Before flip:
                # Triangle was mono if colors[idx]==colors[idx2]==colors[idx3]
                c_old = old_val
                c2 = colors[idx2]
                c3 = colors[idx3]
                if c_old == c2 and c_old == c3:
                    delta -= 1 # Was mono, now not (because idx flipped)
                
                # After flip:
                c_new = new_val
                if c_new == c2 and c_new == c3:
                    delta += 1 # Now mono, was not
                
            cost += delta
            
            if cost == 0:
                if not has_k222(colors):
                    return colors[:]
            
            # Accept or revert
            # Simple hill climbing: accept if better, else revert
            # Or Simulated Annealing?
            # Let's stick to Hill Climbing for simplicity and speed
            if cost > 0 and delta > 0:
                # Revert if it got worse?
                # Actually, if delta > 0, cost increased.
                colors[idx] = old_val
                cost -= delta
            elif cost < 0:
                # Should not happen if logic is correct
                colors[idx] = old_val
                cost -= delta
            # If delta <= 0, keep the change

    return None
