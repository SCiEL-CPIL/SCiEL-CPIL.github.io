import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)
    
    # Map edge to index
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
            
    # Precompute 8-subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute local edge mappings for each subset
    # local_map[S] is a list of 28 global edge indices
    local_map = {}
    for S in all_subsets:
        S_set = set(S)
        mapping = []
        # Local vertices are 0..7 corresponding to S[0]..S[7]
        for i in range(8):
            for j in range(i + 1, 8):
                u_glob = S[i]
                v_glob = S[j]
                mapping.append(edge_index[u_glob][v_glob])
        local_map[S] = mapping
        
    # Precompute subsets containing each edge
    subsets_by_edge = [[] for _ in range(total_edges)]
    # Also need local index of the edge in the subset
    local_idx_in_subset = {} # (S, e_idx) -> local_idx
    for S in all_subsets:
        S_set = set(S)
        # Map global vertex to local index
        v_to_l = {v: i for i, v in enumerate(S)}
        for i in range(8):
            for j in range(i + 1, 8):
                u_glob = S[i]
                v_glob = S[j]
                e_idx = edge_index[u_glob][v_glob]
                subsets_by_edge[e_idx].append(S)
                # Local index is the position in the 28-edge list
                # We can compute it or store it.
                # The list local_map[S] has the same order.
                # So local_idx is the index in local_map[S]
                # It's easier to just find it or store a parallel list.
                # Let's store a dict for (S, e_idx) -> local_idx
                local_idx = len(local_map[S]) # Wrong, need to find position
                # Actually, we can just iterate local_map[S] to find the index
                # But we are precomputing.
                # Let's just store the local index in a separate structure or recompute.
                # Recalculating is fast enough if we just check the bit.
                pass
                
    # Let's create a faster lookup: for each S, a dict from global edge idx to local idx
    S_edge_to_local = {}
    for S in all_subsets:
        mapping = local_map[S]
        d = {}
        for l_idx, g_idx in enumerate(mapping):
            d[g_idx] = l_idx
        S_edge_to_local[S] = d

    # Generate 1680 Q3 patterns (28-bit masks)
    # Bipartition A=0..3, B=4..7
    # Patterns are 3-regular bipartite graphs.
    # Defined by permutation p of 0..3 (missing edges (i, 4+p[i]))
    patterns = []
    for p in itertools.permutations(range(4)):
        mask = 0
        # Local edges: (i, j) where 0<=i<4, 4<=j<8
        # We need to map (i, j) to local edge index 0..27
        # Local edge index for (u, v) with u<v in 0..7:
        # Count edges (a, b) with a < u, plus edges (u, b) with b < v.
        def get_local_edge_idx(u, v):
            if u > v: u, v = v, u
            count = 0
            for a in range(u):
                for b in range(a + 1, 8):
                    count += 1
            for b in range(u + 1, v):
                count += 1
            return count
            
        for i in range(4):
            for j in range(4):
                if j != p[i]:
                    u = i
                    v = 4 + j
                    l_idx = get_local_edge_idx(u, v)
                    mask |= (1 << l_idx)
        patterns.append(mask)
        
    patterns = list(set(patterns)) # Should be 1680, but let's be safe

    def get_local_mask(G, S):
        mapping = local_map[S]
        m = 0
        for i, g_idx in enumerate(mapping):
            if (G >> g_idx) & 1:
                m |= (1 << i)
        return m

    def has_q3(G, S):
        m = get_local_mask(G, S)
        # Quick check: if popcount < 12, return False
        if bin(m).count('1') < 12:
            return False
        for p in patterns:
            if (m & p) == p:
                return True
        return False

    def find_q3(G):
        for S in all_subsets:
            if has_q3(G, S):
                return S
        return None

    def is_safe_add(G, e_idx):
        u, v = edge_list[e_idx]
        for S in subsets_by_edge[e_idx]:
            m = get_local_mask(G, S)
            # Get local index of e_idx in S
            l_idx = S_edge_to_local[S][e_idx]
            m |= (1 << l_idx)
            if bin(m).count('1') < 12:
                continue
            for p in patterns:
                if (m & p) == p:
                    return False
        return True

    best_graph = 0
    best_count = 0

    random.seed(42)

    while time.time() < deadline - 5.0:
        # Start with K16
        G = (1 << total_edges) - 1
        
        # Remove edges to make it Q3-free
        # Randomize removal order
        # We can just pick a random Q3 and remove a random edge from it
        while time.time() < deadline - 5.0:
            S = find_q3(G)
            if S is None:
                break
            
            # Get the local mask to find which edges are in the Q3
            m = get_local_mask(G, S)
            # Find a pattern that is a subset
            valid_pattern = None
            for p in patterns:
                if (m & p) == p:
                    valid_pattern = p
                    break
            
            if valid_pattern is None:
                break # Should not happen
            
            # Pick a random edge from the pattern to remove
            # Get indices of bits set in valid_pattern
            bits = []
            pm = valid_pattern
            while pm:
                lsb = pm & -pm
                l_idx = lsb.bit_length() - 1
                bits.append(l_idx)
                pm ^= lsb
            
            l_idx_to_remove = random.choice(bits)
            g_idx_to_remove = local_map[S][l_idx_to_remove]
            G &= ~(1 << g_idx_to_remove)
            
        # Now G is Q3-free
        current_count = bin(G).count('1')
        
        # Try to add edges back
        missing_edges = [i for i in range(total_edges) if not (G >> i) & 1]
        random.shuffle(missing_edges)
        
        added_any = True
        while added_any and time.time() < deadline - 5.0:
            added_any = False
            for e_idx in missing_edges:
                if is_safe_add(G, e_idx):
                    G |= (1 << e_idx)
                    missing_edges.remove(e_idx)
                    current_count += 1
                    added_any = True
        
        if current_count > best_count:
            best_count = current_count
            best_graph = G

    # Final pass
    missing_edges = [i for i in range(total_edges) if not (best_graph >> i) & 1]
    for e_idx in missing_edges:
        if is_safe_add(best_graph, e_idx):
            best_graph |= (1 << e_idx)

    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if (best_graph >> e_idx) & 1:
            result.append([u, v])

    return result
