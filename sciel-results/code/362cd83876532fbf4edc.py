import time
import random

N = 20
C = 20
DEADLINE = 148.0


def _build_conflict_matrix():
    """Precompute conflict[e1][e2] = 1 if edges e1,e2 are disjoint and in a common 4-cycle."""
    pairs = []
    for u in range(N):
        for v in range(u + 1, N):
            pairs.append((u, v))
    m = len(pairs)  # 190
    eidx = {(u, v): i for i, (u, v) in enumerate(pairs)}
    
    # For each pair of edges, check if they are disjoint and form part of a 4-cycle
    conflict = [[0] * m for _ in range(m)]
    
    # Precompute adjacency for K_n: all vertices adjacent to all others
    # Two edges (a,b) and (c,d) with all distinct vertices form a 4-cycle if
    # they are "opposite" in some 4-cycle, i.e., the four vertices form a C4
    # In K_n, any 4 distinct vertices form a C4 (multiple ways)
    # Edges (a,b) and (c,d) with {a,b,c,d} all distinct: they are in a 4-cycle
    # specifically, a-b-c-d-a uses edges (a,b),(b,c),(c,d),(d,a)
    # or a-b-d-c-a uses (a,b),(b,d),(d,c),(c,a)
    # So (a,b) and (c,d) share a 4-cycle iff all four vertices are distinct
    
    for i in range(m):
        a, b = pairs[i]
        for j in range(i + 1, m):
            c, d = pairs[j]
            if len({a, b, c, d}) == 4:
                # Check: do they share a 4-cycle?
                # In K_n, yes - any 4 distinct vertices give a C4
                # But we need edges (a,b) and (c,d) to be in the SAME 4-cycle
                # A 4-cycle is a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
                # So (a,b) and (c,d) are in this C4 if they are opposite edges
                # For (a,b) and (c,d) to be opposite in a 4-cycle, we need
                # the 4-cycle to be a-?-c-?-b-? ... actually:
                # C4: a, x, c, y where edges are (a,x),(x,c),(c,y),(y,a)
                # OR a, y, c, x where edges are (a,y),(y,c),(c,x),(x,a)
                # In K_n both x and y can be any of the remaining 2 vertices
                # So (a,b) and (c,d) with 4 distinct vertices: 
                #   C4 a-b-c-d-a needs edges (a,b),(b,c),(c,d),(d,a) - YES (a,b) and (c,d) are opposite
                #   So they share a 4-cycle
                conflict[i][j] = 1
                conflict[j][i] = 1
    
    return pairs, eidx, conflict


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def construct():
    start = time.time()
    rng = random.Random(12345)
    
    # Build conflict structure
    pairs, eidx, conflict = _build_conflict_matrix()
    m = len(pairs)  # 190
    
    # Precompute for each edge, the list of conflicting edges
    conflict_lists = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and conflict[i][j]:
                conflict_lists[i].append(j)
    
    # Initialize: color each edge randomly
    color = [rng.randrange(C) for _ in range(m)]
    
    # Build color-to-edges mapping
    color_edges = {c: set() for c in range(C)}
    for i in range(m):
        color_edges[color[i]].add(i)
    
    # Compute initial cost: number of conflicting pairs in same color
    # cost = sum over colors c of sum over pairs (e1,e2) in E_c of conflict[e1][e2]
    def compute_cost():
        cost = 0
        for c in range(C):
            elist = list(color_edges[c])
            for idx_i in range(len(elist)):
                e1 = elist[idx_i]
                for idx_j in range(idx_i + 1, len(elist)):
                    e2 = elist[idx_j]
                    cost += conflict[e1][e2]
        return cost
    
    current_cost = compute_cost()
    best_cost = current_cost
    best_color = color[:]
    
    if current_cost == 0:
        # Already valid!
        result = [[pairs[i][0], pairs[i][1], color[i]] for i in range(m)]
        return result
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.1
    steps = 50000
    
    # Precompute conflict sums for fast delta
    # For edge i in color c, its contribution to cost from color c is:
    #   sum_{j in E_c, j != i} conflict[i][j]
    # When we move edge i from c_old to c_new:
    #   delta = (sum_{j in E_new} conflict[i][j]) - (sum_{j in E_old, j != i} conflict[i][j])
    
    # Precompute for speed: for each edge i, an array conf_sum[i][c] = sum_{j in E_c} conflict[i][j]
    # This changes as colors change, so we update incrementally
    
    # Actually, let's just compute delta on the fly using the conflict_lists
    # For edge i, delta when moving from c_old to c_new:
    #   cost_change = sum_{j in color_edges[c_new]} conflict[i][j] - sum_{j in color_edges[c_old], j != i} conflict[i][j]
    
    for step in range(steps):
        if time.time() > DEADLINE - 2:
            break
        
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Pick random edge
        i = rng.randrange(m)
        c_old = color[i]
        
        # Pick new color (different)
        if C <= 2:
            c_new = 1 - c_old
        else:
            c_new = rng.randrange(C)
            while c_new == c_old:
                c_new = rng.randrange(C)
        
        # Compute delta cost
        # Remove i from c_old: lose conflicts with other edges in c_old
        lost = 0
        for j in color_edges[c_old]:
            if j != i:
                lost += conflict[i][j]
        
        # Add i to c_new: gain conflicts with edges in c_new
        gained = 0
        for j in color_edges[c_new]:
            gained += conflict[i][j]
        
        delta = gained - lost
        
        # Accept if delta < 0 or with probability exp(-delta/T)
        if delta < 0:
            accept = True
        else:
            accept = (rng.random() < (2.718281828 ** (-delta / T)))
        
        if accept:
            # Update
            color_edges[c_old].discard(i)
            color_edges[c_new].add(i)
            color[i] = c_new
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_color = color[:]
                
                if best_cost == 0:
                    # Found valid coloring!
                    result = [[pairs[i2][0], pairs[i2][1], best_color[i2]] for i2 in range(m)]
                    return result
    
    # Return best found
    result = [[pairs[i][0], pairs[i][1], best_color[i]] for i in range(m)]
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print("edges:", len(res))
    print("time: %.2f" % (t1 - t0))
    print("valid:", _validate(res))
