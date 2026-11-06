import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # safety margin before 60s kill
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Number of edges with first vertex < i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    total_edges = N * (N - 1) // 2
    
    # Precompute: for each edge, list of (other_edge1, other_edge2) pairs that form 4-cycles
    # A 4-cycle on vertices {a,b,c,d} (sorted) has 3 matchings:
    #   (ab,cd), (ac,bd), (ad,bc)
    # Edge (i,j) participates in cycles where i,j are two of the four vertices.
    # For edge (i,j) with i<j, a 4-cycle containing it uses two more vertices k,l.
    # The cycle edges are: (i,j) plus two of {(i,k),(i,l),(j,k),(j,l)} forming a 4-cycle.
    # Actually, the three 4-cycles on {i,j,k,l} (sorted a<b<c<d) are:
    #   abcd: edges (a,b),(b,c),(c,d),(d,a)
    #   acbd: edges (a,c),(c,b),(b,d),(d,a)  
    #   adbc: edges (a,d),(d,b),(b,c),(c,a)
    # Wait, let me think more carefully.
    # On 4 vertices {v1,v2,v3,v4}, the three 4-cycles (Hamiltonian cycles) are:
    #   v1-v2-v3-v4-v1: edges (v1,v2),(v2,v3),(v3,v4),(v4,v1)
    #   v1-v2-v4-v3-v1: edges (v1,v2),(v2,v4),(v4,v3),(v3,v1)
    #   v1-v3-v2-v4-v1: edges (v1,v3),(v3,v2),(v2,v4),(v4,v1)
    
    # For each edge e=(i,j), find all 4-cycles containing it.
    # A 4-cycle contains edge (i,j) if i,j are adjacent in the cycle.
    # The other two vertices are k,l (distinct from i,j).
    # For the three cycles on {i,j,k,l}:
    #   Cycle 1: i-j-k-l-i -> edges (i,j),(j,k),(k,l),(l,i)
    #   Cycle 2: i-j-l-k-i -> edges (i,j),(j,l),(l,k),(k,i)
    #   Cycle 3: i-k-j-l-i -> edges (i,k),(k,j),(j,l),(l,i)
    #   Cycle 4: i-l-j-k-i -> edges (i,l),(l,j),(j,k),(k,i)
    # Wait, I need to be more careful. Let me enumerate properly.
    
    # Actually, for 4 vertices {a,b,c,d} with a<b<c<d, the three 4-cycles are:
    # 1. a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
    # 2. a-b-d-c-a: edges (a,b),(b,d),(d,c),(c,a)
    # 3. a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)
    
    # So for edge (a,b): appears in cycles 1 and 2
    # For edge (b,c): appears in cycle 1
    # For edge (c,d): appears in cycle 1
    # For edge (d,a): appears in cycles 1 and 3
    # For edge (b,d): appears in cycles 2 and 3
    # For edge (c,a): appears in cycles 2 and 3
    
    # Let me just precompute all 4-cycles and their edges.
    
    # Build edge index lookup
    # For each 4-set {a,b,c,d} with a<b<c<d, the three 4-cycles are:
    # C1: (a,b),(b,c),(c,d),(d,a)
    # C2: (a,b),(b,d),(d,c),(c,a)
    # C3: (a,c),(c,b),(b,d),(d,a)
    
    # Precompute for each edge, the list of (e1, e2) pairs where {edge, e1, e2, e3} forms a 4-cycle
    # Actually, for a 4-cycle with edges (e0,e1,e2,e3), the constraint is that all 4 colors must not be distinct.
    # When we recolor edge e, we need to check all 4-cycles containing e.
    # For each such 4-cycle, the other 3 edges are fixed, so we just check if the new color makes 4 distinct.
    
    # Precompute: for each edge, list of tuples (idx1, idx2, idx3) where the 4-cycle has edges (this_edge, idx1, idx2, idx3)
    edge_cycles = [[] for _ in range(total_edges)]
    
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Three 4-cycles on {a,b,c,d}
                    # C1: (a,b),(b,c),(c,d),(d,a)
                    e_ab = edge_idx(a,b)
                    e_bc = edge_idx(b,c)
                    e_cd = edge_idx(c,d)
                    e_da = edge_idx(d,a)
                    edge_cycles[e_ab].append((e_bc, e_cd, e_da))
                    edge_cycles[e_bc].append((e_ab, e_cd, e_da))
                    edge_cycles[e_cd].append((e_ab, e_bc, e_da))
                    edge_cycles[e_da].append((e_ab, e_bc, e_cd))
                    
                    # C2: (a,b),(b,d),(d,c),(c,a)
                    e_bd = edge_idx(b,d)
                    e_dc = edge_idx(d,c)
                    e_ca = edge_idx(c,a)
                    edge_cycles[e_ab].append((e_bd, e_dc, e_ca))
                    edge_cycles[e_bd].append((e_ab, e_dc, e_ca))
                    edge_cycles[e_dc].append((e_ab, e_bd, e_ca))
                    edge_cycles[e_ca].append((e_ab, e_bd, e_dc))
                    
                    # C3: (a,c),(c,b),(b,d),(d,a)
                    e_ac = edge_idx(a,c)
                    e_cb = edge_idx(c,b)
                    e_bd2 = edge_idx(b,d)
                    e_da2 = edge_idx(d,a)
                    edge_cycles[e_ac].append((e_cb, e_bd2, e_da2))
                    edge_cycles[e_cb].append((e_ac, e_bd2, e_da2))
                    edge_cycles[e_bd2].append((e_ac, e_cb, e_da2))
                    edge_cycles[e_da2].append((e_ac, e_cb, e_bd2))
    
    # Initial coloring: transitive (color = min(i,j))
    colors = [0] * total_edges
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = i
    
    def is_valid():
        for e in range(total_edges):
            for (e1, e2, e3) in edge_cycles[e]:
                if len({colors[e], colors[e1], colors[e2], colors[e3]}) == 4:
                    return False
        return True
    
    def can_set_edge(e, new_color):
        old_color = colors[e]
        if new_color == old_color:
            return True
        for (e1, e2, e3) in edge_cycles[e]:
            if len({new_color, colors[e1], colors[e2], colors[e3]}) == 4:
                return False
        return True
    
    def count_colors():
        return len(set(colors))
    
    best_colors = colors[:]
    best_count = count_colors()
    
    # Local search: try to recolor edges to new colors
    max_color = 119
    
    # Strategy: iteratively try to introduce new colors
    # For each edge, try assigning a color not yet used, if valid
    # Then do hill climbing on number of distinct colors
    
    rng = random.Random(42)
    
    # First pass: greedy - try to assign new colors
    used_colors = set(colors)
    for _ in range(10000):
        if time.time() > deadline:
            break
        improved = False
        for e in range(total_edges):
            if time.time() > deadline:
                break
            # Try to find a new color for this edge
            available = [c for c in range(max_color + 1) if c not in used_colors]
            rng.shuffle(available)
            for c in available:
                if can_set_edge(e, c):
                    colors[e] = c
                    used_colors.add(c)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    current_count = count_colors()
    if current_count > best_count:
        best_count = current_count
        best_colors = colors[:]
    
    # Hill climbing: try random recolorings that maintain validity
    # Objective: maximize number of distinct colors
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try a random edge and random new color
        e = rng.randint(0, total_edges - 1)
        old_c = colors[e]
        
        # Try a random color (could be new or existing)
        new_c = rng.randint(0, max_color)
        
        if can_set_edge(e, new_c):
            colors[e] = new_c
            new_count = count_colors()
            if new_count >= current_count:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
            else:
                # Revert
                colors[e] = old_c
        else:
            # Try a different approach: pick from colors that are valid
            # Sample a few random colors
            for _ in range(5):
                new_c = rng.randint(0, max_color)
                if can_set_edge(e, new_c):
                    colors[e] = new_c
                    new_count = count_colors()
                    if new_count >= current_count:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = colors[:]
                    else:
                        colors[e] = old_c
                    break
    
    # Second phase: more aggressive search with simulated annealing-like approach
    # Try to introduce colors that are "almost" usable
    for iteration in range(30000):
        if time.time() > deadline:
            break
        
        e = rng.randint(0, total_edges - 1)
        old_c = colors[e]
        
        # Try colors in a range
        for _ in range(3):
            new_c = rng.randint(0, max_color)
            if can_set_edge(e, new_c):
                colors[e] = new_c
                new_count = count_colors()
                if new_count >= current_count - 1:  # allow slight decrease
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
                else:
                    colors[e] = old_c
                break
    
    # Final: try to fill in with any valid new colors
    used_colors = set(best_colors)
    for e in range(total_edges):
        if time.time() > deadline:
            break
        if best_colors[e] in used_colors:
            for c in range(max_color + 1):
                if c not in used_colors and can_set_edge(e, c):
                    best_colors[e] = c
                    used_colors.add(c)
                    break
    
    return best_colors
# EVOLVE-BLOCK-END

