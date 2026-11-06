import numpy as np
import time
import random

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    n = 16
    num_edges = 120
    num_k4 = 1820
    
    # Precompute edge indices for each K4
    # Edge (i, j) index: i*n - i*(i+1)//2 + (j-i-1) ? 
    # Simpler: map (i, j) to index
    edge_idx = {}
    idx = 0
    k4_edges = []
    k4_vertices = []
    for v1 in range(n):
        for v2 in range(v1+1, n):
            edge_idx[(v1, v2)] = idx
            idx += 1
            
    # Generate all K4s and their edge indices
    import itertools
    vertices = list(range(n))
    k4_list = list(itertools.combinations(vertices, 4))
    
    for k4 in k4_list:
        v1, v2, v3, v4 = k4
        e1 = edge_idx[(v1, v2)]
        e2 = edge_idx[(v1, v3)]
        e3 = edge_idx[(v1, v4)]
        e4 = edge_idx[(v2, v3)]
        e5 = edge_idx[(v2, v4)]
        e6 = edge_idx[(v3, v4)]
        k4_edges.append([e1, e2, e3, e4, e5, e6])
        k4_vertices.append(k4)
        
    k4_edges = np.array(k4_edges) # shape (1820, 6)
    
    def count_bad(coloring):
        # coloring: array of shape (120,)
        # Get colors for all edges in all K4s
        # colors_for_k4: shape (1820, 6)
        colors_for_k4 = coloring[k4_edges]
        # Check if number of unique colors in each row is < 5
        # Efficient way: 
        # Sort rows? No.
        # Just count uniques. 
        # Numpy doesn't have native "count unique per row" easily without sorting or loops.
        # Looping 1820 times is fast enough.
        bad_count = 0
        for i in range(num_k4):
            c = colors_for_k4[i]
            if len(set(c)) < 5:
                bad_count += 1
        return bad_count

    def run_sa(k, time_limit):
        # Initialize random coloring
        coloring = np.random.randint(0, k, size=num_edges)
        current_bad = count_bad(coloring)
        best_coloring = coloring.copy()
        best_bad = current_bad
        
        if current_bad == 0:
            return coloring, 0
        
        temp = 1.0
        cooling = 0.9995
        best_temp = temp
        best_bad_for_temp = current_bad
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Pick random edge
            e_idx = np.random.randint(num_edges)
            old_color = coloring[e_idx]
            
            # Pick new color
            if k == 1:
                new_color = old_color
            else:
                new_color = np.random.randint(k)
                if new_color == old_color:
                    continue
            
            coloring[e_idx] = new_color
            
            # Calculate delta
            # Only K4s containing e_idx are affected
            # Find which K4s contain e_idx
            # Precomputation: for each edge, list of K4 indices
            # Let's do this precomputation outside to be fast
            
            # Re-evaluating only affected K4s is complex to implement quickly here.
            # Given 1820 K4s, full re-evaluation is ~1820 * 6 ops = ~10k ops.
            # In Python, 10k ops is ~1ms.
            # So 1000 steps/sec is feasible. 
            # We want more.
            
            # Let's stick to full re-evaluation for simplicity and correctness, 
            # but optimize the count_bad function.
            
            new_bad = count_bad(coloring)
            
            delta = new_bad - current_bad
            
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                current_bad = new_bad
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_coloring = coloring.copy()
                    if best_bad == 0:
                        return best_coloring, 0
            else:
                coloring[e_idx] = old_color # Revert
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
                # Restart with higher temp?
                # Simple restart
                temp = 1.0
        
        return best_coloring, best_bad

    # Precompute affected K4s for each edge to speed up delta calculation
    # This allows O(1) update of bad count.
    # edge_to_k4s[e] = list of k4 indices
    edge_to_k4s = [[] for _ in range(num_edges)]
    for i in range(num_k4):
        edges_in_k4 = k4_edges[i]
        for e in edges_in_k4:
            edge_to_k4s[e].append(i)
    
    def count_bad_fast(coloring):
        bad = np.zeros(num_k4, dtype=bool)
        for i in range(num_k4):
            c = coloring[k4_edges[i]]
            if len(set(c)) < 5:
                bad[i] = True
        return np.sum(bad)
        
    # Actually, maintaining a set of bad K4 indices is faster.
    # But for simplicity, let's use the full re-eval but optimized.
    # 1820 K4s * 6 edges = 10920 edge lookups.
    # Using NumPy: 
    # colors = coloring[k4_edges] -> (1820, 6)
    # uniques = len(set(...)) is the bottleneck.
    
    # Let's use a different heuristic: Local Search with delta calculation.
    
    def run_sa_delta(k, time_limit):
        coloring = np.random.randint(0, k, size=num_edges)
        # Initial bad mask
        bad_mask = np.zeros(num_k4, dtype=bool)
        for i in range(num_k4):
            if len(set(coloring[k4_edges[i]])) < 5:
                bad_mask[i] = True
        current_bad = np.sum(bad_mask)
        best_coloring = coloring.copy()
        best_bad = current_bad
        
        if current_bad == 0:
            return coloring, 0

        temp = 1.0
        cooling = 0.999
        end_time = time.time() + time_limit
        steps = 0
        
        while time.time() < end_time:
            e_idx = np.random.randint(num_edges)
            old_color = coloring[e_idx]
            new_color = np.random.randint(k)
            if new_color == old_color:
                continue
            
            # Identify affected K4s
            affected = edge_to_k4s[e_idx]
            
            # Calculate delta
            delta = 0
            valid_move = True
            # We need to check if move improves/worsens
            # For each affected K4, check status before and after
            
            # Optimization: 
            # Before: bad_mask[i]
            # After: need to check distinct colors
            
            # Let's just calculate new status for affected
            # If number of affected is small (~91), this is fast.
            
            # Store old colors to revert if needed
            # We modify coloring[e_idx] = new_color temporarily?
            # Or just simulate.
            
            # Simulate:
            # For each k4 in affected:
            #   old_c = coloring[k4_edges]
            #   new_c = old_c.copy(); new_c[k4_edges.index(e_idx)] = new_color
            #   This is slow.
            
            # Better:
            # Just update coloring and re-evaluate affected, then revert if rejected.
            
            coloring[e_idx] = new_color
            
            new_bad_count_in_affected = 0
            old_bad_count_in_affected = 0
            
            for i in affected:
                if bad_mask[i]:
                    old_bad_count_in_affected += 1
                # Check if currently bad
                if len(set(coloring[k4_edges[i]])) < 5:
                    bad_mask[i] = True
                    new_bad_count_in_affected += 1
                else:
                    bad_mask[i] = False
                    # It was potentially bad before, but now good?
                    # We already handled old_bad above.
                    # Wait, logic:
                    # delta = (new_bad_total - old_bad_total)
                    # new_bad_total = old_bad_total - old_bad_in_affected + new_bad_in_affected
                    # So we just need new_bad_in_affected.
                    
            # Wait, the loop above updates bad_mask.
            # We need to count how many are True in affected after update.
            # new_bad_in_affected = sum(bad_mask[i] for i in affected)
            # old_bad_in_affected = count of bad_mask[i] that were True BEFORE update.
            
            # Let's redo this cleanly.
            # Revert coloring for calculation clarity
            coloring[e_idx] = old_color
            
            old_bad_in_affected = sum(1 for i in affected if bad_mask[i])
            
            # Apply new color
            coloring[e_idx] = new_color
            
            new_bad_in_affected = 0
            for i in affected:
                if len(set(coloring[k4_edges[i]])) < 5:
                    new_bad_in_affected += 1
                    bad_mask[i] = True
                else:
                    bad_mask[i] = False
            
            # If rejected, we must revert bad_mask and coloring
            # But we need delta to decide.
            delta = (new_bad_in_affected - old_bad_in_affected)
            
            # Note: current_bad variable needs update
            # current_bad = current_bad - old_bad_in_affected + new_bad_in_affected
            # But we haven't committed yet.
            
            if delta < 0 or (temp > 0 and random.random() < np.exp(-delta / temp)):
                # Accept
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_coloring = coloring.copy()
                    if best_bad == 0:
                        return best_coloring, 0
            else:
                # Reject
                # Revert coloring
                coloring[e_idx] = old_color
                # Revert bad_mask
                for i in affected:
                    if len(set(coloring[k4_edges[i]])) < 5:
                        bad_mask[i] = True
                    else:
                        bad_mask[i] = False
            
            temp *= cooling
            if temp < 0.001:
                temp = 1.0 # Restart
            steps += 1
            if steps % 1000 == 0 and time.time() > end_time:
                break

        return best_coloring, best_bad

    # Try K=13
    if time.time() < start_time + 50:
        sol, bad = run_sa_delta(13, 50)
        if bad == 0:
            return sol.tolist()
            
    # Try K=14
    if time.time() < start_time + 90:
        sol, bad = run_sa_delta(14, 30)
        if bad == 0:
            return sol.tolist()

    # Try K=15
    if time.time() < start_time + 110:
        sol, bad = run_sa_delta(15, 20)
        if bad == 0:
            return sol.tolist()

    # Fallback: Seed solution (17 colours)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
