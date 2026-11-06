import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 11
    best_colors = None
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Try to find larger n using Local Search with Efficient Delta Evaluation
    # Key insight: Compute delta in O(n) instead of O(n^2) by maintaining adjacency masks
    
    def solve_n(n, time_limit):
        """
        Attempts to find a valid 5-coloring of K_n.
        Returns (colors, violations) or (None, None).
        """
        if time.time() > time_limit:
            return None, None
            
        num_edges = n * (n - 1) // 2
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Precompute edge index lookup
        edge_idx = {}
        for k, (i, j) in enumerate(edge_list):
            edge_idx[(i, j)] = k
            edge_idx[(j, i)] = k
            
        # Precompute neighbors for each vertex
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                neighbors[i].append(j)
                neighbors[j].append(i)
        
        def get_violations_fast(colors):
            """Compute total violations using adjacency masks."""
            adj_mask = [[0] * n for _ in range(5)]
            for k, (i, j) in enumerate(edge_list):
                c = colors[k]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                
            violations = 0
            for c in range(5):
                for u in range(n):
                    mask_u = adj_mask[c][u]
                    if mask_u == 0:
                        continue
                    for v in range(u + 1, n):
                        mask_v = adj_mask[c][v]
                        if mask_v == 0:
                            continue
                        common = mask_u & mask_v
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
            return violations, adj_mask
        
        def compute_delta(old_color, new_color, u, v, colors, adj_mask):
            """
            Compute change in violations when changing edge (u,v) from old_color to new_color.
            Returns delta (new_violations - old_violations).
            """
            if old_color == new_color:
                return 0
                
            delta = 0
            
            # Remove contributions of edge (u,v) with old_color
            # For each w, if (u,w) is old_color and (v,w) is old_color, 
            # then u and v share w as common neighbor in old_color
            mask_u_old = adj_mask[old_color][u]
            mask_v_old = adj_mask[old_color][v]
            common_old = mask_u_old & mask_v_old
            cnt_old = common_old.bit_count()
            
            # When we remove edge (u,v) from old_color:
            # u and v no longer share each other, but they might share other neighbors
            # Actually, the violation count for pair (u,v) in old_color was max(0, cnt_old - 1)
            # After removal, cnt_old decreases by 1 (since v was in mask_u_old and u was in mask_v_old)
            # Wait, no. adj_mask[old_color][u] contains v, and adj_mask[old_color][v] contains u.
            # common_old includes both u and v themselves? No, common neighbors are w != u, v.
            # Let's think carefully.
            
            # adj_mask[c][u] is the set of neighbors of u in color c.
            # So adj_mask[old_color][u] contains v (since (u,v) is old_color).
            # adj_mask[old_color][v] contains u.
            # common_old = adj_mask[old_color][u] & adj_mask[old_color][v]
            # This includes w such that (u,w) is old_color AND (v,w) is old_color.
            # Note: u is in adj_mask[old_color][v], and v is in adj_mask[old_color][u].
            # But u is NOT in adj_mask[old_color][u] (no self-loops), so u is not in common_old.
            # Similarly v is not in common_old.
            # So common_old is exactly the set of common neighbors w != u, v.
            
            # Before change: violation contribution from pair (u,v) in old_color = max(0, cnt_old - 1)
            # After removing (u,v) from old_color:
            #   adj_mask[old_color][u] loses v, adj_mask[old_color][v] loses u
            #   common_new_old = common_old (unchanged, since u,v not in common_old)
            #   Wait, that's wrong. The common neighbors don't change when we remove (u,v).
            #   The violation is about pairs of vertices having >= 2 common neighbors.
            #   For pair (u,v), the common neighbors are w such that (u,w)=c and (v,w)=c.
            #   Removing (u,v) doesn't change which w are common neighbors of u and v.
            #   So the violation count for pair (u,v) doesn't change when we remove (u,v)?
            #   
            #   Hmm, but the violation is counted per pair (u,v) as max(0, common_count - 1).
            #   Changing the color of (u,v) affects:
            #   1. The color class of (u,v) itself
            #   2. The common neighbor counts for pairs involving u or v
            #
            #   Let's reconsider. The total violation is sum over all pairs (a,b) and colors c of max(0, common_count_c(a,b) - 1).
            #   When we change (u,v) from old_color to new_color:
            #   - In old_color: (u,v) is removed. This affects common_count_old(a,b) for pairs (a,b) where a or b is u or v.
            #     Specifically, for any w, if (u,w) is old_color, then u and w shared v as common neighbor.
            #     So common_count_old(u,w) decreases by 1.
            #     Similarly common_count_old(v,w) decreases by 1 if (v,w) is old_color.
            #   - In new_color: (u,v) is added. Similarly, common_count_new(u,w) increases by 1 if (u,w) is new_color.
            #
            #   This is getting complex. Let's just recompute violations from scratch for simplicity,
            #   but use the fast bitmask method. For n=26, num_edges=325, this should be fast enough.
            
            # Actually, let's just recompute. The key is that get_violations_fast is O(n^2 * 5) which is fine.
            return None  # We'll just recompute
            
        def local_search_fast(colors, max_iters, time_limit):
            """Local search with fast violation recomputation."""
            current_violations, adj_mask = get_violations_fast(colors)
            if current_violations == 0:
                return colors, 0
                
            best_colors = colors[:]
            best_violations = current_violations
            temp = 1.0
            cooling = 0.9995
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                    
                if current_violations == 0:
                    break
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                new_violations, _ = get_violations_fast(colors)
                delta = new_violations - current_violations
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                else:
                    colors[e_idx] = old_color
                    
                temp *= cooling
                
            return best_colors, best_violations
        
        # ILS Loop
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
                
            # Random Init
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Local Search
            colors, viol = local_search_fast(colors, 3000, time_limit)
            
            if viol == 0:
                return colors, 0
            else:
                # Perturb
                num_changes = max(1, num_edges // 20)
                for _ in range(num_changes):
                    e_idx = random.randint(0, num_edges - 1)
                    colors[e_idx] = random.randint(0, 4)
                    
                # Local Search on perturbed
                colors, viol = local_search_fast(colors, 3000, time_limit)
                if viol == 0:
                    return colors, 0

        return None, None

    # Try to find valid colorings for increasing n
    for n in range(12, 27):  # Try up to 26
        if time.time() > deadline - 5.0:
            break
            
        colors, viol = solve_n(n, deadline - 2.0)
        
        if colors is not None and viol == 0:
            best_n = n
            best_colors = colors
            # Continue to try larger n
        else:
            # Failed to find valid coloring for this n
            # Try one more time with more restarts if time allows
            if time.time() < deadline - 10.0:
                colors, viol = solve_n(n, deadline - 2.0)
                if colors is not None and viol == 0:
                    best_n = n
                    best_colors = colors
                else:
                    break
            else:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
