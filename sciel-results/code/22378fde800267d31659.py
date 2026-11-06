import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit
    
    def check_c4_fast(adj_bits, n, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks.
        A C4 exists if there exist x in N(u), y in N(v), x!=y, x~y."""
        nu = adj_bits[u] & ~(1 << u)
        nv = adj_bits[v] & ~(1 << v)
        # For each x in N(u), check if any neighbor of x is in N(v) and != x
        # More efficient: check common neighbors of x and v for x in N(u)
        # C4: u-x-y-v where x~u, x~y, y~v
        # So for each x in N(u), we need y in N(v) \ {x} such that x~y
        nu_iter = nu
        while nu_iter:
            lsb = nu_iter & (-nu_iter)
            x = lsb.bit_length() - 1
            nu_iter ^= lsb
            # Check if x has any neighbor in N(v) other than itself
            # adj_bits[x] & nv gives common neighbors of x and v
            # But we need x~y where y in N(v), y != x
            # adj_bits[x] & nv: if x is in nv, we need to exclude it, but x shouldn't be in nv typically
            common = adj_bits[x] & nv
            if common:
                # There exists y in N(v) such that x~y
                # But we need x != y, and since y is in N(v) and x is in N(u), 
                # if x is also in N(v), we need to check if there's another y
                # Actually if common != 0, there's a C4 (x~u, x~y, y~v, and we're adding u~v)
                # Wait, we need the cycle u-x-y-v-u. So x~u (given), x~y (checked), y~v (given), v~u (being added)
                # The issue: x could equal v? No, x is in N(u) and we're checking if adding u-v creates C4
                # x could equal... let's just check: if common has any bit, there's a C4
                # But we should exclude the case where the only common neighbor is problematic
                # Actually for C4: u-x-y-v, we need x != y, x != v, y != u
                # x is in N(u) so x != u. y is in N(v) so y != v.
                # We need x != y. If common = adj_bits[x] & nv, and if x is in nv, then x is in common.
                # But if x is the only element in common, then x~x which is impossible, so x can't be in common
                # Wait, adj_bits[x] doesn't include x (no self-loops), so x is not in adj_bits[x].
                # So common = adj_bits[x] & nv never includes x. Good.
                return True
        return False
    
    def count_c4s_bitmask(adj_bits, n):
        """Count C4s using bitmask approach."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] & (1 << j)):
                    # Not adjacent, check common neighbors
                    common = adj_bits[i] & adj_bits[j]
                    c = common.bit_count()
                    if c >= 2:
                        count += 1
        return count
    
    def min_degree(adj_bits, n):
        return min(adj_bits[i].bit_count() for i in range(n))
    
    def get_low_degree_vertices(adj_bits, n, min_deg_req):
        """Get vertices with degree below required."""
        return [i for i in range(n) if adj_bits[i].bit_count() < min_deg_req]
    
    def greedy_build(n, min_deg_req, time_limit, seed):
        """Build a C4-free graph with min degree >= min_deg_req using greedy + global repair."""
        t0 = time.time()
        rng = random.Random(seed)
        
        adj_bits = [0] * n
        edges_set = set()
        
        def add_edge(u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.add((a, b))
        
        def remove_edge(u, v):
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            a, b = min(u, v), max(u, v)
            edges_set.discard((a, b))
        
        def min_deg():
            return min(adj_bits[i].bit_count() for i in range(n))
        
        def can_add_edge(u, v):
            """Check if edge (u,v) can be added without creating C4 and without violating min degree."""
            if u == v:
                return False
            if adj_bits[u] & (1 << v):
                return False
            if check_c4_fast(adj_bits, n, u, v):
                return False
            # Check min degree won't be violated (adding edge only increases degrees)
            return True
        
        # Initialize with a sparse random graph
        p = min_deg_req / max(1, n - 1) * 0.8
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    if not check_c4_fast(adj_bits, n, i, j):
                        add_edge(i, j)
        
        stuck_count = 0
        repair_budget = 500
        global_repair_count = 0
        
        while time.time() - t0 < time_limit:
            md = min_deg()
            if md >= min_deg_req:
                edge_list = []
                for a, b in edges_set:
                    edge_list.append([a, b])
                return {'n': n, 'edges': edge_list}
            
            # Find vertices with minimum degree
            low_v = min(range(n), key=lambda i: adj_bits[i].bit_count())
            low_deg = adj_bits[low_v].bit_count()
            
            if low_deg < min_deg_req:
                # Try to add edges to low_v
                candidates = [v for v in range(n) if v != low_v and not (adj_bits[low_v] & (1 << v))]
                rng.shuffle(candidates)
                
                added = False
                for v in candidates:
                    if can_add_edge(low_v, v):
                        add_edge(low_v, v)
                        added = True
                        stuck_count = 0
                        break
                
                if not added:
                    stuck_count += 1
                    # Global repair: use min-cut inspired approach
                    # Identify all low-degree vertices and find edges whose removal
                    # would unblock the most of them
                    if stuck_count >= 5 and repair_budget > 0:
                        repair_budget -= 1
                        global_repair_count += 1
                        
                        low_verts = get_low_degree_vertices(adj_bits, n, min_deg_req)
                        if len(low_verts) >= 2 and edges_set:
                            # For each low-degree vertex, find which existing edges block it
                            # An edge (a,b) blocks vertex v if removing it would allow v to connect to a or b
                            # More precisely: for low vertex v, it can't connect to non-neighbors because of C4 constraints
                            # The blocking edges are those that create C4s when we try to add v's desired edges
                            
                            # Simplified: find edges incident to neighbors of low vertices that, if removed,
                            # would reduce C4 constraints
                            # For each low vertex v, compute which edges are "critical" for blocking
                            
                            block_scores = defaultdict(int)
                            for v in low_verts:
                                non_neigh = [x for x in range(n) if x != v and not (adj_bits[v] & (1 << x))]
                                for x in non_neigh:
                                    # Check why v can't connect to x
                                    if not check_c4_fast(adj_bits, n, v, x):
                                        continue  # Can actually add, shouldn't be here
                                    # Find the C4 that blocks this
                                    # C4: v-a-b-x where a in N(v), b in N(x), a~b
                                    # The blocking edges are (a,b), (a,v), (b,x), (v,x being added)
                                    # Removing (a,b) would unblock v-x
                                    nu = adj_bits[v]
                                    nx = adj_bits[x]
                                    nu_iter = nu
                                    while nu_iter:
                                        lsb = nu_iter & (-nu_iter)
                                        a = lsb.bit_length() - 1
                                        nu_iter ^= lsb
                                        # Check if a has neighbor in N(x)
                                        common = adj_bits[a] & nx
                                        if common:
                                            # Edge (a, b) for b in common is blocking
                                            common_iter = common
                                            while common_iter:
                                                lsb2 = common_iter & (-common_iter)
                                                b = lsb2.bit_length() - 1
                                                common_iter ^= lsb2
                                                if a != b and (adj_bits[a] & (1 << b)):
                                                    ea, eb = min(a, b), max(a, b)
                                                    block_scores[(ea, eb)] += 1
                            
                            if block_scores:
                                # Remove the most blocking edge
                                worst_edge = max(block_scores, key=block_scores.get)
                                u, w = worst_edge
                                if adj_bits[u] & (1 << w):
                                    # Check if removing this edge violates min degree
                                    du = adj_bits[u].bit_count() - 1
                                    dw = adj_bits[w].bit_count() - 1
                                    if du >= min_deg_req and dw >= min_deg_req:
                                        remove_edge(u, w)
                                        stuck_count = 0
                                    else:
                                        # Try second worst
                                        sorted_edges = sorted(block_scores.items(), key=lambda x: -x[1])
                                        for edge_candidate, score in sorted_edges[1:5]:
                                            u2, w2 = edge_candidate
                                            if adj_bits[u2] & (1 << w2):
                                                du2 = adj_bits[u2].bit_count() - 1
                                                dw2 = adj_bits[w2].bit_count() - 1
                                                if du2 >= min_deg_req and dw2 >= min_deg_req:
                                                    remove_edge(u2, w2)
                                                    stuck_count = 0
                                                    break
                                else:
                                    stuck_count = 0
                            else:
                                # No specific blocking edges found, remove random edge not incident to low vertices
                                safe_edges = [(a, b) for a, b in edges_set 
                                             if a not in low_verts and b not in low_verts]
                                if safe_edges:
                                    u, w = rng.choice(safe_edges)
                                    du = adj_bits[u].bit_count() - 1
                                    dw = adj_bits[w].bit_count() - 1
                                    if du >= min_deg_req and dw >= min_deg_req:
                                        remove_edge(u, w)
                                        stuck_count = 0
                        else:
                            # Single low vertex or no edges, try random removal
                            if edges_set:
                                safe_edges = [(a, b) for a, b in edges_set
                                             if adj_bits[a].bit_count() > min_deg_req and adj_bits[b].bit_count() > min_deg_req]
                                if safe_edges:
                                    u, w = rng.choice(safe_edges)
                                    remove_edge(u, w)
                                    stuck_count = 0
                    elif stuck_count > 20:
                        # Give up on this attempt
                        break
                else:
                    continue
            else:
                # All vertices meet minimum degree, but we're in the loop
                # This shouldn't happen if we check at top, but just in case
                break
        
        return None
    
    best_result = None
    best_n = 0
    
    # Time budgets for each n
    time_budgets = {
        47: 28.0,
        46: 35.0,
        45: 20.0,
        44: 12.0,
        43: 8.0,
        42: 6.0,
        41: 5.0,
        40: 4.0,
        39: 3.0,
    }
    
    seed_counter = 0
    for n in range(47, 39, -1):
        if time.time() - start_time > deadline - 2:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        min_deg_req = max(1, n - 39)
        t_limit = min(time_budgets.get(n, 5.0), remaining - 2)
        if t_limit < 1:
            continue
        
        # Multiple restarts with different seeds
        # Allocate time across attempts
        num_attempts = max(1, int(t_limit / 4.0))
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline - 2:
                break
            
            time_per_attempt = t_limit / max(1, num_attempts - attempt) if attempt < num_attempts - 1 else 2.0
            seed = 12345 + seed_counter * 7919 + attempt * 104729
            seed_counter += 1
            
            result = greedy_build(n, min_deg_req, time_per_attempt, seed)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                # Found valid graph for this n, return it
                return result
    
    # Fallback
    if best_result is not None:
        return best_result
    
    # Trivial valid graph: K3 (n=3, min_deg_req=1, no C4)
    return {'n': 3, 'edges': [[0, 1], [1, 2], [0, 2]]}
