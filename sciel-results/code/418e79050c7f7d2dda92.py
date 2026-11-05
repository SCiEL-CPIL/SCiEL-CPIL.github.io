import random
import time
import itertools

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_EDGE_SET = set()
for _a, _b in _Q3_EDGES:
    _Q3_EDGE_SET.add((_a, _b))
    _Q3_EDGE_SET.add((_b, _a))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    # Start with K3,13
    adj = [[False] * n for _ in range(n)]
    for u in range(3):
        for v in range(3, 16):
            adj[u][v] = True
            adj[v][u] = True
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # Only need to check Q3s that use the new edge (u,v)
        # The new edge must correspond to one of the 12 Q3 edges
        # For each Q3 edge (a,b), check if there's a mapping where (u,v) maps to (a,b)
        # and all other Q3 edges are present
        
        # Try both orientations: u->a,v->b and u->b,v->a
        for a, b in _Q3_EDGES:
            for mapping_uv in [(a, u, b, v), (b, u, a, v)]:
                qa, ua, qb, ub = mapping_uv
                # qa, qb are Q3 vertices; ua, ub are graph vertices
                # We need to assign the other 6 Q3 vertices to 6 of the remaining 14 graph vertices
                # such that all Q3 edges are satisfied
                
                # Q3 vertices not yet assigned: all except qa, qb
                unassigned_q3 = [i for i in range(8) if i != qa and i != qb]
                # For each unassigned Q3 vertex, find which graph vertices it must connect to
                
                # Build constraints: for each unassigned Q3 vertex x, 
                # it must connect to all assigned neighbors and unassigned neighbors
                
                # This is still complex. Let's just do a full check with the new edge added temporarily
                adj[ua][ub] = True
                adj[ub][ua] = True
                
                # Now check if there's a Q3 containing edge (ua, ub)
                # The Q3 must use vertices ua, ub and 6 others
                # For efficiency, only check 6-subsets from remaining vertices that could complete a Q3
                
                # For each candidate 6-subset from remaining 14 vertices:
                # C(14,6) = 3003, times 12 Q3 edges * 2 orientations = 72072 checks
                # Each check is a small backtracking - might be too slow
                
                # Alternative: just check if the graph (with new edge) has ANY Q3
                # But that's the expensive check
                
                # Let's do a targeted check: Q3 must contain edge (ua,ub)
                # In Q3, each edge connects two vertices. The 6 other vertices must form
                # the rest of the cube structure.
                
                # For a given Q3 edge (qa,qb) mapped to (ua,ub):
                # qa's neighbors in Q3: 3 vertices (one is qb, two others)
                # qb's neighbors in Q3: 3 vertices (one is qa, two others)
                
                # Let's get the Q3 structure
                q3_adj = {i: set() for i in range(8)}
                for x, y in _Q3_EDGES:
                    q3_adj[x].add(y)
                    q3_adj[y].add(x)
                
                # Neighbors of qa (excluding qb):
                qa_other_neighbors = [x for x in q3_adj[qa] if x != qb]
                # Neighbors of qb (excluding qa):
                qb_other_neighbors = [x for x in q3_adj[qb] if x != qa]
                
                # The 6 unassigned Q3 vertices are: qa_other_neighbors + qb_other_neighbors + remaining
                # Actually: all 8 vertices minus {qa, qb} = 6 vertices
                # These 6 include: qa_other_neighbors (2), qb_other_neighbors (2), and 2 more
                
                # For the mapping to work:
                # - Each neighbor of qa in Q3 must be adjacent to ua in graph
                # - Each neighbor of qb in Q3 must be adjacent to ub in graph
                
                # Candidate graph vertices for qa's other neighbors: must be adjacent to ua
                # Candidate graph vertices for qb's other neighbors: must be adjacent to ub
                
                # This is getting complex. Let me just do the full has_q3 check.
                
                result = _has_q3_full(adj, n)
                adj[ua][ub] = False
                adj[ub][ua] = False
                if result:
                    return True
        return False
    
    def _has_q3_full(adj, n):
        """Full check for Q3 in graph."""
        for subset in itertools.combinations(range(n), 8):
            # Quick filter: all vertices must have degree >= 3 in induced subgraph
            ok = True
            for v in subset:
                deg = sum(1 for u in subset if u != v and adj[v][u])
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            subset_list = list(subset)
            assignment = [-1] * 8
            graph_to_q3 = [-1] * n
            
            def backtrack(q3_idx):
                if time.time() > deadline - 5:
                    return False
                if q3_idx == 8:
                    for a, b in _Q3_EDGES:
                        if not adj[assignment[a]][assignment[b]]:
                            return False
                    return True
                for gv in subset_list:
                    if graph_to_q3[gv] != -1:
                        continue
                    feasible = True
                    for a in range(q3_idx):
                        if (a, q3_idx) in _Q3_EDGE_SET:
                            if not adj[gv][assignment[a]]:
                                feasible = False
                                break
                    if not feasible:
                        continue
                    assignment[q3_idx] = gv
                    graph_to_q3[gv] = q3_idx
                    if backtrack(q3_idx + 1):
                        return True
                    assignment[q3_idx] = -1
                    graph_to_q3[gv] = -1
                return False
            
            if backtrack(0):
                return True
        return False
    
    # Try to add missing edges
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                missing.append((i, j))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 15:
            break
        if not creates_q3(u, v):
            adj[u][v] = True
            adj[v][u] = True
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
