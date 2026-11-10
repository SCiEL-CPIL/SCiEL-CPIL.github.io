import time
import random
from itertools import permutations, combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_bitmask(adj, n=16):
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def greedy_3_colorable(adj, n=16, max_trials=10):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    if greedy_3_colorable(adj, n, max_trials=5):
        return False
    return is_not_3_colorable_bitmask(adj, n)

def generate_compositions(n, k):
    if k == 1:
        yield [n]
    else:
        for i in range(1, n - k + 2):
            for tail in generate_compositions(n - i, k - 1):
                yield [i] + tail

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    # Grötzsch graph structure
    # Vertices 0-4: Cycle
    # Vertices 5-9: Copies
    # Vertex 10: Root
    # Edges in original graph (indices 0-10)
    g_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    best_edges = []
    best_count = 0
    
    # Pre-calculate neighbor lists for the original graph to speed up edge count calculation
    # neighbors[i] is list of j such that (i,j) in g_edges
    neighbors = [[] for _ in range(11)]
    for u, v in g_edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
        
    # Iterate over all size distributions
    for comp in generate_compositions(n, num_original):
        if time.time() - START_TIME > TIME_LIMIT - 20: # Leave 20s for local search
            break
            
        comp_sorted = sorted(comp)
        s_root = comp_sorted[-1]
        cycle_sizes = comp_sorted[:5]
        copy_sizes = comp_sorted[5:10]
        
        best_comp_edges = 0
        best_cycle_perm = None
        best_copy_perm = None
        
        # Try all permutations of cycle sizes (120)
        for c_perm in permutations(cycle_sizes):
            # Map vertex 0-4 to sizes in c_perm
            s_cycle = list(c_perm)
            
            # Calculate "weights" for copy vertices
            # Copy 5 connects to Cycle 1, 4
            # Copy 6 connects to Cycle 0, 2
            # Copy 7 connects to Cycle 1, 3
            # Copy 8 connects to Cycle 2, 4
            # Copy 9 connects to Cycle 3, 0
            weights = [0] * 5
            weights[0] = s_cycle[1] + s_cycle[4] # Copy 5
            weights[1] = s_cycle[0] + s_cycle[2] # Copy 6
            weights[2] = s_cycle[1] + s_cycle[3] # Copy 7
            weights[3] = s_cycle[2] + s_cycle[4] # Copy 8
            weights[4] = s_cycle[3] + s_cycle[0] # Copy 9
            
            # To maximize sum s_copy[i] * weights[i], we should sort copy_sizes and weights in the same order (rearrangement inequality)
            # So we don't need to iterate 120 permutations for copies, just the optimal sorted assignment!
            sorted_weights = sorted(weights, reverse=True)
            sorted_copies = sorted(copy_sizes, reverse=True)
            
            # Reconstruct the assignment of copy sizes to vertices 5-9
            # We need to know which vertex gets which size.
            # We have weights for vertices 5,6,7,8,9.
            # We want to assign largest copy size to largest weight, etc.
            # Create a list of (weight, vertex_index)
            w_v = [(weights[i], i) for i in range(5)]
            w_v.sort(reverse=True)
            
            s_copy = [0] * 5
            for i in range(5):
                vertex_idx = w_v[i][1]
                s_copy[vertex_idx] = sorted_copies[i]
            
            # Calculate total edges for this configuration
            edge_count = 0
            # Cycle edges
            for i in range(5):
                j = (i + 1) % 5
                edge_count += s_cycle[i] * s_cycle[j]
            
            # Copy-Cycle edges
            # 5-1, 5-4
            edge_count += s_copy[0] * (s_cycle[1] + s_cycle[4])
            # 6-0, 6-2
            edge_count += s_copy[1] * (s_cycle[0] + s_cycle[2])
            # 7-1, 7-3
            edge_count += s_copy[2] * (s_cycle[1] + s_cycle[3])
            # 8-2, 8-4
            edge_count += s_copy[3] * (s_cycle[2] + s_cycle[4])
            # 9-3, 9-0
            edge_count += s_copy[4] * (s_cycle[3] + s_cycle[0])
            
            # Root-Copy edges
            edge_count += s_root * sum(s_copy)
            
            if edge_count > best_comp_edges:
                best_comp_edges = edge_count
                best_cycle_perm = s_cycle
                best_copy_perm = s_copy
                best_root = s_root
                
        # After finding best perms for this composition, construct the graph
        if best_cycle_perm is not None:
            sizes = [0] * 11
            for i in range(5):
                sizes[i] = best_cycle_perm[i]
            for i in range(5):
                sizes[5+i] = best_copy_perm[i]
            sizes[10] = best_root
            
            edges = []
            adj = [0] * n
            
            # Map original vertex to range of new vertices
            offsets = [0] * 12
            for i in range(11):
                offsets[i+1] = offsets[i] + sizes[i]
                
            for u, v in g_edges:
                for i in range(sizes[u]):
                    u_node = offsets[u] + i
                    for j in range(sizes[v]):
                        v_node = offsets[v] + j
                        edges.append([u_node, v_node])
                        adj[u_node] |= 1 << v_node
                        adj[v_node] |= 1 << u_node
                        
            # Verify triangle-free (should be true by construction)
            if is_triangle_free_bitmask(adj, n):
                # Verify 4-chromatic
                if is_not_3_colorable_fast(adj, n):
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = list(edges)
                        
    # Local search phase
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        deadline = START_TIME + TIME_LIMIT
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            candidates.append((u, v))
            
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > deadline:
                    break
                
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                current_edges.append([u, v])
                
                if is_not_3_colorable_fast(adj, n):
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                    improved = True
                else:
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    current_edges.pop()
                    
            if not improved:
                break
                
    if not best_edges:
        # Fallback to a known valid graph if search fails
        # Simple Grötzsch blow-up with sizes [1,1,1,1,1, 1,1,1,1,1, 6]
        sizes = [1,1,1,1,1, 1,1,1,1,1, 6]
        edges = []
        offsets = [0] * 12
        for i in range(11):
            offsets[i+1] = offsets[i] + sizes[i]
        for u, v in g_edges:
            for i in range(sizes[u]):
                u_node = offsets[u] + i
                for j in range(sizes[v]):
                    v_node = offsets[v] + j
                    edges.append([u_node, v_node])
        best_edges = edges

    return best_edges
