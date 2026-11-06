import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    # We want to find a clique in the complement graph (where edge exists if lcm > 256)
    # So conflict_mask[i] has bits set for nodes that CONFLICT with i (lcm <= 256)
    # The "compatibility" mask is the complement
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Compatibility: node i is compatible with node j if they do NOT conflict
    # compat_mask[i] has bits set for nodes compatible with i (including i itself for convenience)
    compat_mask = [0] * n_total
    for i in range(n_total):
        # Start with all bits set, then clear conflict bits
        compat_mask[i] = ((1 << n_total) - 1) & ~conflict_mask[i]
    
    conflict_count = [bin(conflict_mask[i]).count('1') for i in range(n_total)]
    
    # Multiple greedy orderings to get good initial solutions
    scores = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    order1 = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    order2 = sorted(range(n_total), key=lambda i: 1.0 / all_nodes[i], reverse=True)
    order3 = sorted(range(n_total), key=lambda i: conflict_count[i])
    scores4 = [(1.0 / all_nodes[i]) / (1 + conflict_count[i])**0.5 for i in range(n_total)]
    order4 = sorted(range(n_total), key=lambda i: scores4[i], reverse=True)
    scores5 = [(1.0 / all_nodes[i]) / max(1, conflict_count[i]) for i in range(n_total)]
    order5 = sorted(range(n_total), key=lambda i: scores5[i], reverse=True)
    
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    for order in [order1, order2, order3, order4, order5]:
        sel = []
        used = 0
        for idx in order:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Enhanced local search: add, swap, 2-swap
    def local_search(selected, deadline):
        selected = selected[:]
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # 1. Try adding nodes
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                used |= (1 << i)
                improved = True
            
            # 2. Try 1-swap
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                s_i = node_to_idx[s]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break
            
            # 3. Try 2-swap: remove two, add two
            if not improved and len(selected) >= 2:
                n_sel = len(selected)
                for i in range(n_sel):
                    if (time.time() - start_time) > deadline:
                        break
                    for j in range(i + 1, n_sel):
                        if (time.time() - start_time) > deadline:
                            break
                        si = node_to_idx[selected[i]]
                        sj = node_to_idx[selected[j]]
                        used_without = used & ~(1 << si) & ~(1 << sj)
                        removed_sum = 1.0 / selected[i] + 1.0 / selected[j]
                        
                        best_pair = None
                        best_gain = 0
                        non_sel_list = [k for k in range(n_total) if not ((used_without >> k) & 1)]
                        
                        for a_idx in range(len(non_sel_list)):
                            if (time.time() - start_time) > deadline:
                                break
                            na = non_sel_list[a_idx]
                            na_val = all_nodes[na]
                            if used_without & conflict_mask[na]:
                                continue
                            for b_idx in range(a_idx + 1, len(non_sel_list)):
                                nb = non_sel_list[b_idx]
                                nb_val = all_nodes[nb]
                                if (used_without | (1 << na)) & conflict_mask[nb]:
                                    continue
                                gain = (1.0 / na_val + 1.0 / nb_val) - removed_sum
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (na, nb)
                        
                        if best_pair is not None:
                            na, nb = best_pair
                            selected[i] = all_nodes[na]
                            selected[j] = all_nodes[nb]
                            used = used_without | (1 << na) | (1 << nb)
                            improved = True
                            break
                    if improved:
                        break
        
        return selected
    
    # Run local search from heuristic
    best_subset = local_search(best_heuristic_subset, deadline - 25)
    best_value = sum(1.0 / v for v in best_subset)
    
    # Maximum weight clique via branch-and-bound with bitset filtering
    # The graph: vertices are nodes 2..256, edge between i,j if lcm(i,j) > 256
    # We want max weight clique where weight[i] = 1/all_nodes[i]
    
    def max_weight_clique(weights, compat, n, deadline, init_clique=None):
        """
        weights: list of weights for each vertex
        compat: list of bitmasks, compat[i] has bit j set if i and j are compatible (edge exists)
        n: number of vertices
        deadline: time limit
        init_clique: initial clique to start from (list of vertex indices)
        Returns: (best_clique_list, best_weight)
        """
        best_clique = init_clique[:] if init_clique else []
        best_weight = sum(weights[i] for i in best_clique)
        
        # Greedy coloring for upper bound
        def greedy_coloring(candidates, weights):
            """
            Color the candidate set. Returns (coloring, max_color)
            coloring[i] = color of vertex i (0-indexed among candidates)
            max_color = number of colors used
            Upper bound on clique size = max_color
            For weighted case, we use a weighted coloring bound
            """
            if not candidates:
                return {}, 0
            
            # Sort candidates by weight descending for better bounds
            sorted_cands = sorted(candidates, key=lambda x: weights[x], reverse=True)
            
            coloring = {}
            color_classes = []  # list of lists of vertices
            
            for v in sorted_cands:
                placed = False
                for c, cls in enumerate(color_classes):
                    # Check if v is compatible with all vertices in this color class
                    can_place = True
                    for u in cls:
                        if not (compat[v] >> u) & 1:
                            can_place = False
                            break
                    if can_place:
                        color_classes[c].append(v)
                        coloring[v] = c
                        placed = True
                        break
                if not placed:
                    color_classes.append([v])
                    coloring[v] = len(color_classes) - 1
            
            return coloring, len(color_classes)
        
        def weighted_color_bound(candidates, weights):
            """
            Compute an upper bound on the total weight of any clique in candidates.
            Uses greedy coloring: sum of max weight in each color class.
            """
            if not candidates:
                return 0.0
            
            # Sort by weight descending
            sorted_cands = sorted(candidates, key=lambda x: weights[x], reverse=True)
            
            color_classes = []  # list of (max_weight_in_class, list_of_vertices)
            
            for v in sorted_cands:
                placed = False
                for c, (max_w, cls) in enumerate(color_classes):
                    # Check if v is compatible with all vertices in this color class
                    can_place = True
                    for u in cls:
                        if not (compat[v] >> u) & 1:
                            can_place = False
                            break
                    if can_place:
                        new_max_w = max(max_w, weights[v])
                        color_classes[c] = (new_max_w, cls + [v])
                        placed = True
                        break
                if not placed:
                    color_classes.append((weights[v], [v]))
            
            # Upper bound: sum of max weights in each color class
            return sum(max_w for max_w, _ in color_classes)
        
        # Branch and bound
        # State: (candidates_bitmask, current_clique, current_weight)
        # We use a stack-based DFS
        
        # Initialize candidates
        if init_clique:
            # Start with vertices compatible with all in init_clique
            cand_mask = (1 << n) - 1
            for v in init_clique:
                cand_mask &= compat[v]
            # Remove vertices already in clique
            for v in init_clique:
                cand_mask &= ~(1 << v)
        else:
            cand_mask = (1 << n) - 1
        
        # Stack: (candidates_mask, clique_list, clique_weight)
        stack = [(cand_mask, best_clique[:], best_weight)]
        
        # Order candidates by weight for better pruning
        # Precompute weight-sorted order
        weight_order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        
        while stack:
            if (time.time() - start_time) > deadline:
                break
            
            cand_mask, clique, cl_weight = stack.pop()
            
            if cand_mask == 0:
                continue
            
            # Get list of candidate vertices
            candidates = []
            for i in weight_order:
                if (cand_mask >> i) & 1:
                    candidates.append(i)
            
            if not candidates:
                continue
            
            # Compute upper bound
            ub = cl_weight + weighted_color_bound(candidates, weights)
            if ub <= best_weight:
                continue
            
            # Try adding each candidate
            for v in candidates:
                if (time.time() - start_time) > deadline:
                    break
                
                new_clique = clique + [v]
                new_weight = cl_weight + weights[v]
                
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_clique = new_clique[:]
                
                # New candidates: those compatible with v and still in candidates
                new_cand_mask = cand_mask & compat[v] & ~(1 << v)
                
                # Only consider candidates that come after v in weight_order for symmetry breaking
                # Actually, for weighted clique, we need to be more careful
                # Let's just add to stack and let the bound handle it
                if new_cand_mask != 0:
                    stack.append((new_cand_mask, new_clique, new_weight))
        
        return best_clique, best_weight
    
    # Prepare for clique solver
    weights = [1.0 / all_nodes[i] for i in range(n_total)]
    
    # Convert best_subset to indices
    init_clique_indices = [node_to_idx[v] for v in best_subset]
    
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 1:
        try:
            clique_result, clique_weight = max_weight_clique(
                weights, compat_mask, n_total, deadline - 3, init_clique_indices
            )
            
            clique_subset = [all_nodes[i] for i in clique_result]
            if clique_subset:
                if clique_weight > best_value:
                    best_subset = clique_subset
                    best_value = clique_weight
        except Exception:
            pass
    
    # Final polish: local search on the best solution
    if (time.time() - start_time) < deadline - 3:
        polished = local_search(best_subset, deadline - 2)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            best_subset = polished
            best_value = val
    
    if not best_subset:
        return [256]
    
    return best_subset
