import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Focus on n from 9 to 13
    # We can extend to 14 if time allows, but 9-13 is usually sufficient for these problems
    for n in range(9, 14):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        total_colorings = 1 << n
        all_ones_mask = (1 << total_colorings) - 1
        
        # Precompute M0 and M1 (vertex color masks)
        # M1[v] has bit c set if vertex v is 1 in coloring c
        # M0[v] has bit c set if vertex v is 0 in coloring c
        M1 = [0] * n
        M0 = [0] * n
        
        for v in range(n):
            half = 1 << v
            period = 1 << (v + 1)
            pattern = ((1 << half) - 1) << half
            num_periods = total_colorings // period
            m1 = 0
            for p in range(num_periods):
                m1 |= (pattern << (p * period))
            M1[v] = m1
            M0[v] = all_ones_mask ^ m1
        
        # Compute edge masks
        # Edge e is monochromatic if all vertices are 0 OR all vertices are 1
        # Mask_e = (Intersection of M0[v] for v in e) OR (Intersection of M1[v] for v in e)
        edge_masks = [0] * num_edges
        for i, e in enumerate(all_edges):
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks[i] = m0 | m1
            
        best_local_count = float('inf')
        best_local_edges = None
        
        # ILS Parameters
        max_iter = 5000
        restarts = 0
        
        while time.time() < deadline:
            restarts += 1
            # Greedy Initialization
            covered = 0
            selected_indices = []
            remaining = list(range(num_edges))
            random.shuffle(remaining)
            
            while covered != all_ones_mask:
                best_gain = -1
                best_idx = -1
                
                # To speed up, we can sample if num_edges is large, but 1287 is small enough
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if best_gain == 0:
                            break
                            
                if best_gain <= 0:
                    break
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered != all_ones_mask:
                continue

            # Local Search: Reduce and Swap
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # 1. Remove redundant edges
                random.shuffle(selected_indices)
                for i in range(len(selected_indices) - 1, -1, -1):
                    e_idx = selected_indices[i]
                    # Check if e_idx is redundant
                    # Calculate union of all other edges
                    new_covered = 0
                    for j in range(len(selected_indices)):
                        if j != i:
                            new_covered |= edge_masks[selected_indices[j]]
                    if new_covered == all_ones_mask:
                        selected_indices.pop(i)
                        improved = True
                        break
                
                # 2. Try 1-for-1 swaps that enable removal of another edge
                if not improved:
                    random.shuffle(selected_indices)
                    for i in range(len(selected_indices)):
                        if time.time() > deadline: break
                        e_out_idx = selected_indices[i]
                        
                        # Calculate unique coverage of e_out
                        # Union of all others
                        union_others = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                union_others |= edge_masks[selected_indices[j]]
                        
                        unique_cov = edge_masks[e_out_idx] & ~union_others
                        if unique_cov == 0:
                            # Should have been caught in step 1, but just in case
                            selected_indices.pop(i)
                            improved = True
                            break
                        
                        # Find e_in that covers unique_cov
                        # We want e_in such that edge_masks[e_in] & unique_cov == unique_cov
                        # And ideally, adding e_in and removing e_out makes some other edge redundant
                        found_swap = False
                        # Sample some candidates to speed up? Or check all?
                        # num_edges is ~1000. Checking all is 1000 ops.
                        # Let's check all unselected edges
                        for e_in_idx in range(num_edges):
                            if e_in_idx in selected_indices:
                                continue
                            if (edge_masks[e_in_idx] & unique_cov) != unique_cov:
                                continue
                            
                            # Potential swap: remove e_out, add e_in
                            # New set: (selected \ {e_out}) U {e_in}
                            # Check if this new set has a redundant edge
                            # If yes, we can remove it, reducing count by 1
                            
                            # Optimization: Only check if e_in covers unique_cov.
                            # Now, does adding e_in make any existing edge redundant?
                            # Or does removing e_out make some edge redundant?
                            # Actually, if we swap e_out with e_in, the coverage is the same (assuming e_in covers unique_cov and e_out is removed).
                            # Wait, if e_in covers unique_cov, and e_out is removed, the union remains all_ones_mask ONLY IF e_in covers everything e_out covered that wasn't covered by others.
                            # Yes, that's what unique_cov is.
                            # So the union remains full.
                            # Now, is any edge in the NEW set redundant?
                            # The new set is S' = (S \ {e_out}) U {e_in}.
                            # We know S was minimal (no redundant edges).
                            # In S', e_in might be redundant if its coverage is subset of others in S'.
                            # Or some e_k in S \ {e_out} might be redundant if its unique coverage in S is now covered by e_in.
                            
                            # Let's just perform the swap and run the "Remove redundant" pass.
                            # But doing a full pass for every candidate is expensive.
                            # Heuristic: Check if e_in is redundant in S'.
                            # e_in is redundant if (Union of S') \ {e_in} == all_ones_mask.
                            # Union of S' \ {e_in} = Union of S \ {e_out}.
                            # We already computed union_others = Union of S \ {e_out}.
                            # So if union_others == all_ones_mask, then e_out was redundant (contradiction).
                            # So e_in is NOT redundant in S' unless union_others == all_ones_mask.
                            
                            # What about other edges?
                            # An edge e_k in S \ {e_out} becomes redundant if its unique coverage in S is covered by e_in.
                            # Unique coverage of e_k in S: U_k = M_{e_k} \ Union(S \ {e_k}).
                            # This is hard to compute for all k.
                            
                            # Simpler: Just do the swap, and then do a quick redundancy check on the new set.
                            # To avoid O(N^2) for every candidate, we can limit candidates.
                            
                            # Let's just pick the first valid e_in and swap, then let the next loop iteration clean up.
                            # This is a "blind" swap. It maintains the count.
                            # To REDUCE count, we need a swap that enables a removal.
                            
                            # Strategy: Swap e_out with e_in. Then try to remove any redundant edge.
                            # If we can remove one, we reduce count by 1.
                            
                            # Let's try this for a few candidates.
                            # We'll just pick the first one that covers unique_cov.
                            selected_indices[i] = e_in_idx
                            found_swap = True
                            improved = True # Force another pass to check for redundancy
                            break
                        
                        if found_swap:
                            break

            if len(selected_indices) < best_local_count:
                best_local_count = len(selected_indices)
                best_local_edges = [all_edges[i] for i in selected_indices]
                
                if best_local_count < best_count:
                    best_count = best_local_count
                    best_n = n
                    best_edges = [list(e) for e in best_local_edges]
                    print(f"Found better: n={n}, edges={best_count}")

            # Perturbation for ILS
            if best_local_edges is not None:
                # Sometimes start from best known to refine?
                # Or just random restarts.
                pass

    if best_edges is None:
        # Fallback: return a valid (though not minimal) hypergraph
        # e.g. all edges for n=9
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
