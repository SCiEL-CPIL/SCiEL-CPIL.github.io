import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 16
    random.seed(42)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)  # 120

    # Map global edges to indices
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1

    # Precompute the 24 Q3 patterns on vertices 0..7 with A={0,1,2,3}, B={4,5,6,7}
    # Each pattern is a 3-regular bipartite graph, defined by which edge is missing per row/col
    # Missing edges form a permutation: (i, 4+p[i]) for i=0..3
    patterns = []
    for p in itertools.permutations(range(4)):
        edges = set()
        for i in range(4):
            for j in range(4):
                if j != p[i]:
                    edges.add((i, 4 + j))
        patterns.append(frozenset(edges))

    # For each 8-subset of vertices, generate all Q3 edge masks
    all_q3_masks = set()

    all_subsets = list(itertools.combinations(range(n), 8))

    for S in all_subsets:
        S_set = set(S)
        # Generate all bipartitions of S into A, B of size 4
        for A in itertools.combinations(S, 4):
            A_set = set(A)
            B = tuple(v for v in S if v not in A_set)

            for pat in patterns:
                mask = 0
                for u_local, v_local in pat:
                    u_glob = A[u_local]
                    v_glob = B[v_local - 4]
                    g_idx = edge_index[u_glob][v_glob]
                    mask |= (1 << g_idx)
                all_q3_masks.add(mask)

    q3_masks = list(all_q3_masks)
    num_q3s = len(q3_masks)

    # Precompute for each edge, which Q3s contain it
    q3_by_edge = [[] for _ in range(total_edges)]
    for q_mask in q3_masks:
        m = q_mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3_by_edge[e_idx].append(q_mask)
            m ^= lsb

    def is_q3_free_addition(G, e_idx):
        for q_mask in q3_by_edge[e_idx]:
            other_edges = q_mask ^ (1 << e_idx)
            if (G & other_edges) == other_edges:
                return False
        return True

    def count_edges(G):
        return bin(G).count('1')

    def add_back_pass(G):
        """Try to add all possible missing edges repeatedly until no more can be added."""
        improved = True
        while improved:
            improved = False
            for e_idx in range(total_edges):
                if not (G >> e_idx) & 1:
                    if is_q3_free_addition(G, e_idx):
                        G |= (1 << e_idx)
                        improved = True
        return G

    def local_search(G):
        """Hill climbing: try to add missing edges, then try swaps to escape local optima."""
        current_count = count_edges(G)
        
        # First, add all possible edges
        G = add_back_pass(G)
        current_count = count_edges(G)
        
        # Try swaps to improve
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            missing_edges = [i for i in range(total_edges) if not (G >> i) & 1]
            if not missing_edges:
                break
            random.shuffle(missing_edges)
            
            for e_add in missing_edges:
                if is_q3_free_addition(G, e_add):
                    G |= (1 << e_idx)
                    current_count += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove an edge from a blocking Q3, add the missing edge
                e_add = random.choice(missing_edges)
                blocker = None
                for q_mask in q3_by_edge[e_add]:
                    other_edges = q_mask ^ (1 << e_add)
                    if (G & other_edges) == other_edges:
                        blocker = q_mask
                        break
                
                if blocker:
                    candidates = []
                    m = blocker ^ (1 << e_add)
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        candidates.append(e_idx)
                        m ^= lsb
                    
                    if candidates:
                        e_remove = random.choice(candidates)
                        G_temp = G & ~(1 << e_remove)
                        if is_q3_free_addition(G_temp, e_add):
                            G = G_temp | (1 << e_add)
                            improved = True
                            # After swap, try to add back more edges
                            G = add_back_pass(G)
                            current_count = count_edges(G)
                            break
        
        return G

    best_graph = 0
    best_count = 0

    # Multiple restarts with greedy + local search
    while time.time() < deadline - 15.0:
        # Greedy construction with random edge order
        G = 0
        edge_order = list(range(total_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if is_q3_free_addition(G, e_idx):
                G |= (1 << e_idx)
        
        # Local search refinement
        G = local_search(G)
        
        current_count = count_edges(G)
        if current_count > best_count:
            best_count = current_count
            best_graph = G

    # Final add-back pass on best graph
    best_graph = add_back_pass(best_graph)

    # Convert to edge list
    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if (best_graph >> e_idx) & 1:
            result.append([u, v])

    return result
