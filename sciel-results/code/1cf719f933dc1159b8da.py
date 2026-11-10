"""C4-free edge sets of Q7: linear-algebraic construction plus local search."""
import time
import random
import numpy as np

# Deadline guard: evaluator kills at 150s; use 140s as our hard limit.
_DEADLINE = 140.0

def construct():
    start_time = time.time()
    
    d = 7
    n = 1 << d  # 128 vertices
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges: list of (u, v) where u < v, u and v differ in exactly one bit
    # Also precompute for each edge, the 6 other edges in the 4 faces that contain it
    # Actually, for C4 check: an edge (u,v) along dimension i creates C4s with any other dim j
    # The C4 is: u, v, u^j, v^j (where j is a different dimension)
    # So adding edge (u,v) along dim i creates a C4 iff for some j != i,
    # the other 3 edges of that 4-cycle are already present.
    
    # For efficient C4 detection, for each pair of vertices (u, v) that differ in 2 bits,
    # we track which of the 4 edges of the corresponding square are selected.
    # A square is identified by (u, v) where u < v and popcount(u ^ v) == 2.
    # The 4 edges of the square with corners u, v, u^a, u^b (where a,b are the two differing bits)
    # are: (u, u^a), (u, u^b), (u^a, v), (u^b, v) where v = u^a^b.
    
    # Let's precompute all squares (2-faces) of Q7.
    # A 2-face is determined by choosing 2 dimensions i < j and a vertex u with bits i,j = 0 (WLOG).
    # The 4 vertices are: u, u^(1<<i), u^(1<<j), u^(1<<(i|j))... wait, u ^ (1<<i) ^ (1<<j).
    # The 4 edges are:
    #   (u, u^(1<<i)), (u^(1<<i), u^(1<<i)^(1<<j)), 
    #   (u, u^(1<<j)), (u^(1<<j), u^(1<<i)^(1<<j))
    
    # Let me represent each square by its 4 edges.
    # For each edge, I want to know which squares it belongs to, and for each square, which edges it has.
    
    # Alternative approach: for each edge e = (u, v) along dimension i,
    # it belongs to (d-1) squares, one for each other dimension j.
    # The square for dims (i, j) containing edge (u, v) has vertices:
    #   u, v = u^(1<<i), u^(1<<j), v^(1<<j) = u^(1<<i)^(1<<j)
    # The 4 edges of this square are:
    #   (u, u^(1<<i)) = e
    #   (u, u^(1<<j))
    #   (u^(1<<i), u^(1<<i)^(1<<j)) = (v, v^(1<<j))
    #   (u^(1<<j), u^(1<<i)^(1<<j))
    
    # For fast C4 checking when adding edge e along dim i:
    # For each j != i, check if the other 3 edges of the (i,j)-square are all present.
    
    # Let's precompute: for each edge (u, v, i) [i is the dimension], 
    # store the list of (j, other3_edges) for j != i.
    
    # Edge representation: (min(u,v), max(u,v)) as a tuple, plus we know the dim.
    # Actually let's just use a dict: edge_key = (min, max) -> True if selected.
    
    # Precompute for each edge its conflict info
    edge_list = []  # list of (u, v, i) where u < v
    edge_to_idx = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edge_list)
                edge_list.append((u, v, i))
                edge_to_idx[(u, v)] = idx
    
    # For each edge index, precompute the squares it's in and the other edges
    # edge_squares[idx] = list of (j, [other_edge_idx1, other_edge_idx2, other_edge_idx3])
    edge_squares = [[] for _ in range(total_edges)]
    
    for idx, (u, v, i) in enumerate(edge_list):
        for j in range(d):
            if j == i:
                continue
            # The square with dims (i, j) containing this edge
            # Vertices: u, v=u^(1<<i), u^(1<<j), v^(1<<j)
            # Edges:
            #   e1 = (u, v) = current edge
            #   e2 = (u, u^(1<<j))
            #   e3 = (v, v^(1<<j)) = (u^(1<<i), u^(1<<i)^(1<<j))
            #   e4 = (u^(1<<j), u^(1<<i)^(1<<j))
            
            u_j = u ^ (1 << j)
            v_j = v ^ (1 << j)
            
            # e2: (u, u_j)
            a, b = min(u, u_j), max(u, u_j)
            e2_idx = edge_to_idx[(a, b)]
            
            # e3: (v, v_j)
            a, b = min(v, v_j), max(v, v_j)
            e3_idx = edge_to_idx[(a, b)]
            
            # e4: (u_j, v_j)
            a, b = min(u_j, v_j), max(u_j, v_j)
            e4_idx = edge_to_idx[(a, b)]
            
            edge_squares[idx].append((j, e2_idx, e3_idx, e4_idx))
    
    # Initial construction using the linear algebraic idea:
    # Map vertices to F2^3 via a linear map, select edges based on parity.
    # Actually, let's use a known good heuristic: 
    # Select edge (u, v) along dim i if some function of u, v, i is odd.
    
    # A simple approach: use a random linear map from F2^7 to F2^3.
    # For edge (u, v) along dim i, select it if the dot product of some hash is 0.
    
    # Let me try: select edge (u, v, i) if popcount(u & (1 << i)) == 0 or some condition.
    # Actually, a cleaner algebraic construction:
    # Assign to each vertex x a vector f(x) in F2^k for some k.
    # Select edge (x, y) along dim i if f(x) + f(y) has some property.
    
    # For C4-freeness: in any 4-cycle x, x^i, x^i^j, x^j (dims i,j),
    # the edges are (x, x^i), (x^i, x^i^j), (x, x^j), (x^j, x^i^j).
    # If we select edge (a, b) iff g(a, b) = 0 for some function,
    # we need to avoid all 4 being selected.
    
    # Let's try: select edge (u, v) along dim i iff (u >> i) % 2 == 0 
    # i.e., the bit above dimension i in u is 0. But u and v differ in bit i,
    # so (u >> i) and (v >> i) differ only in the LSB.
    # Hmm, this might not work well.
    
    # Alternative: use a fixed random seed for reproducibility, then local search.
    random.seed(42)
    np.random.seed(42)
    
    # Start with a greedy construction using random edge ordering
    selected = np.zeros(total_edges, dtype=bool)
    
    # Greedy with random order
    order = list(range(total_edges))
    random.shuffle(order)
    
    for idx in order:
        # Check if adding this edge creates a C4
        ok = True
        for (j, e2, e3, e4) in edge_squares[idx]:
            if selected[e2] and selected[e3] and selected[e4]:
                ok = False
                break
        if ok:
            selected[idx] = True
    
    best_count = int(np.sum(selected))
    best_selected = selected.copy()
    
    # Local search: try to add edges, and if stuck, try removing one and adding two.
    # Phase 1: try to add any edge that doesn't create a C4
    def can_add(selected, idx):
        for (j, e2, e3, e4) in edge_squares[idx]:
            if selected[e2] and selected[e3] and selected[e4]:
                return False
        return True
    
    def try_add_edges(selected):
        """Try to add any possible edge. Returns count added."""
        added = 0
        for idx in range(total_edges):
            if not selected[idx] and can_add(selected, idx):
                selected[idx] = True
                added += 1
        return added
    
    # Initial greedy already did this, but let's do another pass
    try_add_edges(selected)
    best_count = int(np.sum(selected))
    best_selected = selected.copy()
    
    # Phase 2: local search with edge removal and re-addition
    # Try removing one edge and see if we can add 2 or more back
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 20:
        # Not enough time for local search
        return [[edge_list[i][0], edge_list[i][1]] for i in range(total_edges) if best_selected[i]]
    
    # Local search: random restarts with kick
    best_overall = best_selected.copy()
    best_overall_count = best_count
    
    while time.time() - start_time < _DEADLINE - 5:
        # Start from current best
        cur = best_overall.copy()
        cur_count = best_overall_count
        
        # Do a few rounds of: remove k random edges, then greedily add back
        kick_size = random.randint(1, 5)
        for _ in range(kick_size):
            # Remove a random selected edge
            sel_indices = np.where(cur)[0]
            if len(sel_indices) == 0:
                break
            rem = random.choice(sel_indices)
            cur[rem] = False
            cur_count -= 1
        
        # Now greedily add edges
        added = try_add_edges(cur)
        cur_count += added
        
        # Try another pass
        added2 = try_add_edges(cur)
        cur_count += added2
        
        if cur_count > best_overall_count:
            best_overall = cur.copy()
            best_overall_count = cur_count
    
    # Final pass: try to add any remaining edges
    try_add_edges(best_overall)
    final_count = int(np.sum(best_overall))
    
    # Build output
    result = []
    for i in range(total_edges):
        if best_overall[i]:
            u, v, _ = edge_list[i]
            result.append([u, v])
    
    return result
