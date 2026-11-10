import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = 120

    # ---------- Precomputation (once) ----------
    # Global edge index
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1

    # All 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)

    # Local K8 edge indices (0..27)
    local_edge_idx = [[-1] * 8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1

    # Generate Q3 masks for K8 as 28-bit integers
    q3_masks_for_k8 = []
    seen_masks = set()
    for A in itertools.combinations(range(8), 4):
        B = tuple(v for v in range(8) if v not in A)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                m = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != m:
                        mask |= (1 << local_edge_idx[a][b])
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_k8.append(mask)
    q3_masks_for_k8 = tuple(q3_masks_for_k8)

    # Map global edges -> (subset, local_pos)
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1] * total_edges for _ in range(num_subsets)]
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i + 1, 8):
                g_idx = edge_index[s[i]][s[j]]
                l_pos = local_edge_idx[i][j]
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Reverse local_pos -> global edge, per subset (for swap candidate removal)
    local_to_global = [[-1] * 28 for _ in range(num_subsets)]
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i + 1, 8):
                g_idx = edge_index[s[i]][s[j]]
                local_to_global[s_idx][local_edge_idx[i][j]] = g_idx

    # ---------- Fast Q3 membership ----------
    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            new_mask = subset_masks[s_idx] | (1 << local_idx[s_idx][e_idx])
            if contains_q3(new_mask):
                return False
        return True

    # ---------- Tabu Search with Randomized Greedy Initialization ----------
    def run_tabu_search(seed):
        rng = random.Random(seed)
        
        # Initialize with a randomized greedy solution
        G = 0
        subset_masks = [0] * num_subsets
        missing_edges = set(range(total_edges))
        
        def add(e_idx):
            nonlocal G
            G |= (1 << e_idx)
            missing_edges.discard(e_idx)
            for s_idx in edge_to_subsets[e_idx]:
                subset_masks[s_idx] |= (1 << local_idx[s_idx][e_idx])
        
        def remove(e_idx):
            nonlocal G
            G &= ~(1 << e_idx)
            missing_edges.add(e_idx)
            for s_idx in edge_to_subsets[e_idx]:
                subset_masks[s_idx] &= ~(1 << local_idx[s_idx][e_idx])
        
        # Randomized greedy initialization
        edges = list(range(total_edges))
        rng.shuffle(edges)
        for e_idx in edges:
            if time.time() > deadline:
                break
            if can_add_edge(G, e_idx, subset_masks):
                add(e_idx)
        
        # Tabu search
        current_count = G.bit_count()
        best_G = G
        best_count = current_count
        best_subset_masks = subset_masks[:]
        
        tabu_list = {}  # edge_idx -> iteration when it expires
        tabu_tenure = 20
        max_iterations = 2000
        no_improve = 0
        max_no_improve = 500
        aspiration_threshold = best_count
        
        iteration = 0
        while iteration < max_iterations and no_improve < max_no_improve and time.time() < deadline - 5.0:
            # Generate neighbors: try adding each missing edge, removing each present edge
            best_neighbor_G = None
            best_neighbor_count = current_count - 1  # require improvement over current
            best_neighbor_move = None
            best_neighbor_type = None  # 'add' or 'remove'
            best_neighbor_e_idx = None
            best_neighbor_f_idx = None
            
            present_edges = [i for i in range(total_edges) if (G >> i) & 1]
            
            # Try adding edges
            for e_idx in list(missing_edges):
                if can_add_edge(G, e_idx, subset_masks):
                    # Check if tabu
                    if e_idx in tabu_list and tabu_list[e_idx] > iteration:
                        # Aspiration: allow if it improves best
                        if current_count + 1 >= aspiration_threshold:
                            pass  # allow despite tabu
                        else:
                            continue
                    
                    # Evaluate
                    add(e_idx)
                    new_count = G.bit_count()
                    if new_count > best_neighbor_count:
                        best_neighbor_count = new_count
                        best_neighbor_G = G
                        best_neighbor_move = 'add'
                        best_neighbor_e_idx = e_idx
                        best_neighbor_f_idx = -1
                    remove(e_idx)
            
            # Try removing edges
            for f_idx in present_edges:
                if f_idx in tabu_list and tabu_list[f_idx] > iteration:
                    # Check aspiration
                    if current_count - 1 >= aspiration_threshold:
                        pass
                    else:
                        continue
                
                # Evaluate
                remove(f_idx)
                new_count = G.bit_count()
                if new_count > best_neighbor_count:
                    best_neighbor_count = new_count
                    best_neighbor_G = G
                    best_neighbor_move = 'remove'
                    best_neighbor_e_idx = -1
                    best_neighbor_f_idx = f_idx
                add(f_idx)
            
            # If no improving neighbor found, accept best neighbor (could be worsening)
            if best_neighbor_G is None:
                # Try random move to escape
                if missing_edges and present_edges:
                    # Random swap
                    e_idx = rng.choice(list(missing_edges))
                    f_idx = rng.choice(present_edges)
                    
                    # Remove f, add e
                    remove(f_idx)
                    if can_add_edge(G, e_idx, subset_masks):
                        add(e_idx)
                        current_count = G.bit_count()
                        tabu_list[f_idx] = iteration + tabu_tenure
                        if e_idx != -1:
                            tabu_list[e_idx] = iteration + tabu_tenure
                    else:
                        add(f_idx)
                elif present_edges:
                    # Random remove
                    f_idx = rng.choice(present_edges)
                    remove(f_idx)
                    current_count = G.bit_count()
                    tabu_list[f_idx] = iteration + tabu_tenure
                elif missing_edges:
                    # Random add
                    e_idx = rng.choice(list(missing_edges))
                    if can_add_edge(G, e_idx, subset_masks):
                        add(e_idx)
                        current_count = G.bit_count()
                        tabu_list[e_idx] = iteration + tabu_tenure
                else:
                    break
            else:
                # Apply best move
                if best_neighbor_move == 'add':
                    add(best_neighbor_e_idx)
                    tabu_list[best_neighbor_e_idx] = iteration + tabu_tenure
                else:
                    remove(best_neighbor_f_idx)
                    tabu_list[best_neighbor_f_idx] = iteration + tabu_tenure
                
                current_count = G.bit_count()
            
            # Update best
            if current_count > best_count:
                best_count = current_count
                best_G = G
                best_subset_masks = subset_masks[:]
                aspiration_threshold = best_count
                no_improve = 0
            else:
                no_improve += 1
            
            # Clean up expired tabu entries
            if iteration % 10 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
            
            iteration += 1
        
        return best_G, best_count

    # ---------- Multi-restart with tabu search ----------
    best_G = 0
    best_count = 0
    seed = 98765
    while time.time() < deadline - 3.0:
        G, cnt = run_tabu_search(seed)
        seed += 1
        if cnt > best_count:
            best_count = cnt
            best_G = G

    # ---------- Guaranteed-valid fallback (in case nothing found) ----------
    if best_count == 0:
        # 3x3 complete bipartite K3,3,3,3 is Q3-free
        A = [0, 1, 2]
        B = [3, 4, 5]
        C = [6, 7, 8]
        D = [9, 10, 11, 12, 13, 14, 15]
        fb = []
        for a in A:
            for b in B + C + D:
                fb.append([a, b])
        for b in B:
            for c in C:
                fb.append([b, c])
        for b in B:
            for d in D:
                fb.append([b, d])
        for c in C:
            for d in D:
                fb.append([c, d])
        return fb

    return [[u, v] for (u, v) in edge_list if (best_G >> edge_index[u][v]) & 1]
