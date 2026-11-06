import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    
    # Generate PG(2,3) Incidence Graph
    # Points: 0..12, Lines: 13..25
    # Total 26 vertices.
    # Lines in PG(2,3):
    # There are 13 lines.
    # We can generate them by iterating through slopes and intercepts?
    # Or just use a known list.
    # Let's generate them.
    # Points are (x,y) in GF(3)^2 plus point at infinity?
    # No, PG(2,3) has 13 points.
    # Coordinates (x,y,1) and (1,0,0), (0,1,0)?
    # Let's just hardcode the incidence or generate via algebra.
    # GF(3) = {0, 1, 2}
    # Points: (x,y) for x,y in {0,1,2} (9 points) + 3 points at infinity.
    # Lines: y = mx + c for m,c in {0,1,2} (9 lines) + 3 lines at infinity.
    # Wait, lines at infinity are x=inf, y=inf, 1=inf?
    # Standard construction:
    # Points: 0..8 are (x,y). 9,10,11 are directions (slopes). 12 is (inf, inf)?
    # Actually, easier to use a library or known structure.
    # Since I can't import external graph libs, I'll construct it.
    
    def generate_pg23():
        # 26 vertices: 0-12 points, 13-25 lines
        # Points:
        # 0: (0,0), 1: (1,0), 2: (2,0)
        # 3: (0,1), 4: (1,1), 5: (2,1)
        # 6: (0,2), 7: (1,2), 8: (2,2)
        # 9: inf_x (vertical), 10: inf_y (horizontal), 11: inf_1 (slope 1), 12: inf_2 (slope 2)
        # Wait, PG(2,3) has 13 points.
        # Lines:
        # 13: y=0, 14: y=1, 15: y=2
        # 16: x=0, 17: x=1, 18: x=2
        # 19: x+y=0, 20: x+y=1, 21: x+y=2
        # 22: x+2y=0, 23: x+2y=1, 24: x+2y=2
        # 25: Line at infinity? No, we need 13 lines.
        # The 9 finite points are covered by 12 finite lines?
        # Each finite point is on 4 lines.
        # Total incidences = 13 * 4 = 52.
        # Let's define lines by their sets of points.
        
        # Finite lines (y = mx + c mod 3)
        # m=0: y=0 (0,3,6), y=1 (1,4,7), y=2 (2,5,8) -> Lines 13,14,15
        # m=1: x+y=0 (0,4,8), x+y=1 (1,3,7), x+y=2 (2,5,6) -> Lines 16,17,18
        # m=2: x+2y=0 (0,5,7), x+2y=1 (1,6,8), x+2y=2 (2,3,4) -> Lines 19,20,21
        # Verticals (x=const): x=0 (0,3,6), x=1 (1,4,7), x=2 (2,5,8) -> Wait, these are same as y=const?
        # No, in affine plane, x=0 is distinct from y=0.
        # My mapping above:
        # 0:(0,0), 1:(1,0), 2:(2,0)
        # 3:(0,1), 4:(1,1), 5:(2,1)
        # 6:(0,2), 7:(1,2), 8:(2,2)
        
        # Line y=0: 0,1,2. Plus point at infinity for slope 0?
        # In PG(2,q), each line has q+1 points.
        # The line y=0 contains (0,0),(1,0),(2,0) and the point at infinity for horizontal lines.
        # Let's assign:
        # Pt 9: Inf_H (slope 0)
        # Pt 10: Inf_V (slope inf)
        # Pt 11: Inf_1 (slope 1)
        # Pt 12: Inf_2 (slope 2)
        
        # Lines:
        # L13 (y=0): 0,1,2,9
        # L14 (y=1): 3,4,5,9
        # L15 (y=2): 6,7,8,9
        # L16 (x=0): 0,3,6,10
        # L17 (x=1): 1,4,7,10
        # L18 (x=2): 2,5,8,10
        # L19 (x+y=0): 0,4,8,11  (Check: 0+0=0, 1+1=2!=0? Wait. 1+1=2. So (1,1) is not on x+y=0.
        # Let's re-verify x+y=0 mod 3.
        # (0,0)->0. (1,2)->3=0. (2,1)->3=0.
        # So points: 0, 7(1,2), 5(2,1). Plus Inf_2 (slope -1 = 2).
        # So L19: 0, 5, 7, 12.
        
        # Let's do this systematically.
        # Points 0-8: (x,y).
        # Points 9-12: Infinities for slopes 0, inf, 1, 2.
        
        # Line for slope m, intercept c:
        # Contains (x, mx+c) for x=0,1,2. And Inf_m.
        
        lines = {}
        # Slope 0 (Horizontal)
        for c in range(3):
            pts = [(x, c) for x in range(3)]
            pts_ids = [y*3+x for x,y in pts] # 0,1,2 for c=0
            pts_ids.append(9) # Inf_0
            lines[f"H{c}"] = pts_ids
            
        # Slope Inf (Vertical)
        for c in range(3):
            pts = [(c, y) for y in range(3)]
            pts_ids = [y*3+x for x,y in pts] # 0,3,6 for c=0
            pts_ids.append(10) # Inf_Inf
            lines[f"V{c}"] = pts_ids
            
        # Slope 1
        for c in range(3):
            pts = [(x, (x+c)%3) for x in range(3)]
            pts_ids = [y*3+x for x,y in pts]
            pts_ids.append(11) # Inf_1
            lines[f"S1_{c}"] = pts_ids
            
        # Slope 2
        for c in range(3):
            pts = [(x, (2*x+c)%3) for x in range(3)]
            pts_ids = [y*3+x for x,y in pts]
            pts_ids.append(12) # Inf_2
            lines[f"S2_{c}"] = pts_ids
            
        # Total 12 lines?
        # PG(2,3) has 13 lines.
        # The 12 lines above cover all 13 points?
        # Point 9 is on H0, H1, H2.
        # Point 10 is on V0, V1, V2.
        # Point 11 is on S1_0, S1_1, S1_2.
        # Point 12 is on S2_0, S2_1, S2_2.
        # Points 0-8 are on 4 lines each?
        # (0,0) is on H0, V0, S1_0, S2_0.
        # Yes.
        # But we are missing the "Line at Infinity" which connects 9,10,11,12?
        # In PG(2,3), the line at infinity contains the 4 points at infinity.
        # So Line 13: [9, 10, 11, 12].
        
        lines["Inf"] = [9, 10, 11, 12]
        
        # Now build adjacency for 26 vertices (0-12 points, 13-25 lines)
        adj = [0] * 26
        line_idx = 13
        for name, pts in lines.items():
            for p in pts:
                adj[p] |= (1 << line_idx)
                adj[line_idx] |= (1 << p)
            line_idx += 1
            
        return adj

    pg23_adj = generate_pg23()
    
    # Select best 20 vertices from 26
    # Remove 6 vertices.
    # Maximize edges in remaining subgraph.
    
    def get_subgraph_edges(adj_full, keep_mask):
        # keep_mask is 26-bit int
        edges = 0
        for i in range(26):
            if keep_mask & (1 << i):
                # Count neighbors j > i that are also kept
                neighbors = adj_full[i] & keep_mask & ~((1 << (i+1)) - 1)
                edges += neighbors.bit_count()
        return edges

    best_subgraph_mask = 0
    best_edges = 0
    
    # Iterate over all combinations of 6 vertices to remove
    # C(26,6) = 230230. Fast enough.
    for remove_combo in combinations(range(26), 6):
        remove_mask = 0
        for v in remove_combo:
            remove_mask |= (1 << v)
        keep_mask = ((1 << 26) - 1) ^ remove_mask
        e = get_subgraph_edges(pg23_adj, keep_mask)
        if e > best_edges:
            best_edges = e
            best_subgraph_mask = keep_mask

    # Map 26 vertices to 20 vertices
    keep_vertices = [i for i in range(26) if best_subgraph_mask & (1 << i)]
    # We need to relabel 0..19
    old_to_new = {old: new for new, old in enumerate(keep_vertices)}
    
    adj = [0] * 20
    for old_u in keep_vertices:
        new_u = old_to_new[old_u]
        for old_v in keep_vertices:
            if old_u < old_v and (pg23_adj[old_u] & (1 << old_v)):
                new_v = old_to_new[old_v]
                adj[new_u] |= (1 << new_v)
                adj[new_v] |= (1 << new_u)
                
    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
            
        # Check triples involving u or v
        # 1. {u, v, w}
        for w in range(n):
            if w == u or w == v:
                continue
            # Common neighbors of u, v, w
            # N'(u) = N(u) U {v}
            # N'(v) = N(v) U {u}
            # N(w) = N(w)
            # Intersection:
            # (N(u) U {v}) & (N(v) U {u}) & N(w)
            # = (N(u) & N(v) & N(w)) | (N(u) & {u} & N(w)) | (N(v) & {v} & N(w)) | ({v} & {u} & N(w))
            # Note: u not in N(u), v not in N(v).
            # {v} & {u} is empty.
            # N(u) & {u} is empty.
            # N(v) & {v} is empty.
            # Wait, u is in N'(v). Is u in N(u)? No.
            # So the only new elements are:
            # v is in N'(u). Is v in N(v)? No. Is v in N(w)? Maybe.
            # u is in N'(v). Is u in N(u)? No. Is u in N(w)? Maybe.
            
            common_base = (adj[u] & adj[v] & adj[w]).bit_count()
            add_v = 1 if (adj[w] & (1 << v)) else 0
            add_u = 1 if (adj[w] & (1 << u)) else 0
            
            if common_base + add_v + add_u > 2:
                return False

        # 2. {u, a, b} where a,b != v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                # N'(u) & N(a) & N(b)
                # = (N(u) U {v}) & N(a) & N(b)
                # = (N(u) & N(a) & N(b)) U ({v} & N(a) & N(b))
                common_base = (adj[u] & adj[a] & adj[b]).bit_count()
                add_v = 1 if (adj[a] & adj[b] & (1 << v)) else 0
                if common_base + add_v > 2:
                    return False
                
                # N'(v) is not involved here?
                # The triple is {u, a, b}.
                # We only changed N(u).
                # So we only check N'(u) & N(a) & N(b).
                # What about triples {v, a, b}?
                # N'(v) & N(a) & N(b) = (N(v) U {u}) & N(a) & N(b)
                common_base_v = (adj[v] & adj[a] & adj[b]).bit_count()
                add_u = 1 if (adj[a] & adj[b] & (1 << u)) else 0
                if common_base_v + add_u > 2:
                    return False

        return True

    def local_search(adj):
        current_adj = adj[:]
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try adding any valid edge
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if not (current_adj[u] & (1 << v)):
                        if is_valid_addition(current_adj, u, v):
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                            improved = True
        return current_adj

    adj = local_search(adj)
    
    # Random restarts with different subgraphs?
    # The PG(2,3) subgraph is a strong seed.
    # Let's just return this.
    
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
