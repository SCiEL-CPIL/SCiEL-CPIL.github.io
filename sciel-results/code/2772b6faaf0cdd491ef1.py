import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    vertices = list(range(n))
    subsets_4 = list(itertools.combinations(vertices, 4))
    num_subsets = len(subsets_4)
    
    edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    subset_edges = []
    edge_to_subsets = [[] for _ in range(num_edges)]
    
    for s_idx, s in enumerate(subsets_4):
        a,b,c,d = s
        e_indices = [
            edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
            edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]
        ]
        subset_edges.append(e_indices)
        for e_idx in e_indices:
            edge_to_subsets[e_idx].append(s_idx)
            
    def count_colors(colours, e_indices):
        # Returns number of distinct colors in the 6 edges
        c1 = colours[e_indices[0]]
        c2 = colours[e_indices[1]]
        c3 = colours[e_indices[2]]
        c4 = colours[e_indices[3]]
        c5 = colours[e_indices[4]]
        c6 = colours[e_indices[5]]
        return len({c1,c2,c3,c4,c5,c6})

    def is_valid(colours):
        for s_idx in range(num_subsets):
            if count_colors(colours, subset_edges[s_idx]) < 5:
                return False
        return True

    # Try additive structures for m in 13..16
    for m in range(13, 17):
        if time.time() > deadline:
            break
        colours = [(i+j) % m for i in range(n) for j in range(i+1, n)]
        # Note: The edge index in 'edges' corresponds to (i,j) with i<j.
        # The list comprehension above generates in the same order.
        if is_valid(colours):
            return colours

    # Local search for m=15
    def local_search(m, max_time=5.0):
        start_ls = time.time()
        best_viol = float('inf')
        best_colours = None
        
        for restart in range(50):
            if time.time() - start_ls > max_time:
                break
            if time.time() > deadline:
                break
                
            colours = [random.randint(0, m-1) for _ in range(num_edges)]
            
            # Calculate initial violations
            viol = [0] * num_subsets
            total_viol = 0
            for s_idx in range(num_subsets):
                e_idx_list = subset_edges[s_idx]
                if count_colors(colours, e_idx_list) < 5:
                    viol[s_idx] = 1
                    total_viol += 1
                    
            if total_viol == 0:
                return colours
                
            if total_viol < best_viol:
                best_viol = total_viol
                best_colours = colours[:]
                
            improved = True
            stall_counter = 0
            while improved and time.time() - start_ls < max_time:
                improved = False
                stall_counter += 1
                if stall_counter > 2000:
                    break
                    
                idx = random.randint(0, num_edges-1)
                old_c = colours[idx]
                
                # Try all m colors
                best_delta = 0
                best_new_c = old_c
                best_new_viol = viol[:] # Copy for comparison? No, we need to check delta
                
                # To be fast, we just check the affected subsets
                affected_subsets = edge_to_subsets[idx]
                
                # We need to find a new color that minimizes violations
                # We can just iterate through colors and compute delta
                # Since m is small (15), this is fast.
                
                current_viol_sum = 0
                # Pre-compute current violation state for affected subsets?
                # We have viol[s_idx].
                
                min_new_total = total_viol
                found_better = False
                
                for new_c in range(m):
                    if new_c == old_c:
                        continue
                    
                    colours[idx] = new_c
                    
                    # Calculate new total violation for affected subsets
                    # Only these subsets change status
                    new_total = total_viol
                    
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        new_total += (is_viol - was_viol)
                        
                    if new_total < min_new_total:
                        min_new_total = new_total
                        best_new_c = new_c
                        # We don't update viol yet, we revert if not best
                        # But we need to keep track of the new viol states for the best move
                        # This is getting complicated to store.
                        # Simpler: Just pick the best color, then apply it and update viol properly.
                        
                # Revert colouring to old state for now
                colours[idx] = old_c
                
                if min_new_total < total_viol:
                    # Apply the best move
                    colours[idx] = best_new_c
                    # Update viol and total_viol
                    for s_idx in affected_subsets:
                        was_viol = viol[s_idx]
                        e_idx_list = subset_edges[s_idx]
                        is_viol = 1 if count_colors(colours, e_idx_list) < 5 else 0
                        viol[s_idx] = is_viol
                        total_viol += (is_viol - was_viol)
                    
                    improved = True
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colours = colours[:]
                    if total_viol == 0:
                        return colours
                else:
                    # No improvement, try random move or restart?
                    # For now, just break or try another edge
                    # To avoid getting stuck, we can accept a worse move with small prob (Simulated Annealing)
                    # But simple hill climbing is safer for validity.
                    # If we stall, we break and restart.
                    if stall_counter % 100 == 0:
                         break
                    # Otherwise, the while loop will continue and pick another random edge
                    # But 'improved' is False, so it will exit the while loop.
                    # So we should restart if we don't improve.
                    break

        return best_colours if best_viol == 0 else None

    # Try 15
    res15 = local_search(15, max_time=5.0)
    if res15 is not None:
        return res15
        
    # Try 16
    res16 = local_search(16, max_time=5.0)
    if res16 is not None:
        return res16
        
    # Fallback to 17
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
