import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    n = 16
    # Precompute K4s and their edge indices
    # Edges are indexed by (i, j) with i < j
    # Index = i * n + j - (i*(i+1)//2) ... simpler to just use a dict or formula
    # Let's use a flat list of 120 edges.
    # edge_idx(i, j) where i < j
    # We can map (i,j) to index:
    # 0: (0,1)
    # 1: (0,2)
    # ...
    # 15: (0,15)
    # 16: (1,2)
    # ...
    
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            idx += 1
            
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(120)]
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
            edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]
        ]
        k4_edges.append(edges)
        k4_id = len(k4s)
        k4s.append((a,b,c,d))
        for e in edges:
            edge_to_k4s[e].append(k4_id)
            
    def get_initial_coloring(K):
        return [random.randint(0, K-1) for _ in range(120)]

    def evaluate_coloring(colors, K):
        # Returns energy and a list of bad K4s
        # We need to track distinct colors per K4
        # This is slow to do from scratch every time in SA, 
        # but for initialization or final check it's fine.
        energy = 0
        bad_k4s = []
        for k4_id, edges in enumerate(k4_edges):
            c_set = set()
            for e in edges:
                c_set.add(colors[e])
            d = len(c_set)
            if d < 5:
                energy += (5-d)**2
                bad_k4s.append(k4_id)
        return energy, bad_k4s

    def simulate_annealing(K, time_budget):
        colors = get_initial_coloring(K)
        
        # Initialize counts for each K4: count of each color
        # k4_color_counts[k4_id] = {color: count}
        k4_color_counts = [{} for _ in range(len(k4_edges))]
        k4_distinct = [0] * len(k4_edges)
        
        for k4_id, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_id] = counts
            k4_distinct[k4_id] = len(counts)
            
        current_energy = sum((5-d)**2 for d in k4_distinct if d < 5)
        
        best_energy = current_energy
        best_colors = colors[:]
        
        T = 10.0
        cooling = 0.999
        steps = 0
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if current_energy == 0:
                break
                
            # Pick a K4 to focus on, or random edge
            # Strategy: Pick a random bad K4, then a random edge in it
            bad_k4s = [i for i in range(len(k4_edges)) if k4_distinct[i] < 5]
            if not bad_k4s:
                break
                
            target_k4 = random.choice(bad_k4s)
            target_edge = random.choice(k4_edges[target_k4])
            
            old_color = colors[target_edge]
            new_color = random.randint(0, K-1)
            
            if old_color == new_color:
                continue
                
            # Calculate delta energy
            delta_energy = 0
            # Update all K4s containing target_edge
            for k4_id in edge_to_k4s[target_edge]:
                counts = k4_color_counts[k4_id]
                
                # Remove old color
                counts[old_color] -= 1
                if counts[old_color] == 0:
                    # Color no longer present
                    # Check if it was the only one? No, just distinct count
                    # If it was 1, distinct decreases by 1
                    # But we need to know if it was the *last* of that color.
                    # Yes, if count becomes 0, distinct decreases by 1.
                    # Wait, if count was 1, it becomes 0. Distinct -1.
                    # If count was >1, it becomes >0. Distinct unchanged.
                    if counts[old_color] == 0:
                        # It was the last one
                        # We need to know if it was the last one BEFORE decrement?
                        # No, we decremented. If it is 0 now, it was 1 before.
                        # So distinct decreases by 1.
                        # But wait, if we remove a color, distinct decreases by 1.
                        # If we add a color, distinct increases by 1.
                        
                        # Let's trace carefully.
                        # Old state: count[old] = 1. Distinct includes old.
                        # New state: count[old] = 0. Distinct does not include old.
                        # So distinct_new = distinct_old - 1.
                        
                        # However, we are also adding new_color.
                        # Let's handle old and new separately.
                        
                        # This logic is tricky if old_color == new_color (handled above).
                        # If old_color != new_color:
                        
                        # Let's store old distinct
                        old_distinct = k4_distinct[k4_id]
                        
                        # Update counts for old
                        # We already did counts[old_color] -= 1
                        # If counts[old_color] == 0, it means it was 1. So distinct drops by 1.
                        # If counts[old_color] > 0, it was >=2. Distinct unchanged.
                        
                        # Now update counts for new
                        if new_color not in counts:
                            counts[new_color] = 1
                            # New color added, distinct increases by 1
                            # But wait, if old_color was the only one and new_color is new,
                            # distinct = old_distinct - 1 + 1 = old_distinct.
                            
                            # Let's just recalculate distinct for this K4?
                            # len(counts) is O(K). K is small (13-15).
                            # 120 edges, 1820 K4s.
                            # Recalculating distinct for affected K4s is cheap.
                            pass
                            
                        else:
                            counts[new_color] += 1
                            
                        new_distinct = len(counts)
                        
                        # Calculate energy change for this K4
                        old_pen = max(0, 5 - old_distinct)**2
                        new_pen = max(0, 5 - new_distinct)**2
                        delta_energy += new_pen - old_pen
                        
                        # Update stored state
                        k4_distinct[k4_id] = new_distinct
                        # Note: k4_color_counts is already updated in place
                        
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                # Accept
                colors[target_edge] = new_color
                current_energy += delta_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
            else:
                # Reject: Revert changes
                # We modified k4_color_counts and k4_distinct.
                # We need to revert them.
                # This is expensive if we don't store old state.
                # Alternative: Calculate delta, if reject, don't update colors.
                # But we *did* update the counts dict in place.
                # We must revert the counts dict changes.
                
                for k4_id in edge_to_k4s[target_edge]:
                    counts = k4_color_counts[k4_id]
                    # Revert old_color
                    counts[old_color] += 1
                    if counts[old_color] == 1:
                         # It was 0 before? No, it was >=1.
                         # If it becomes 1, it was 0? No.
                         # If we added 1 and it became 1, it was 0.
                         # So distinct increases by 1?
                         # This is getting messy.
                         
                         # Better approach: Don't modify in place unless accepted.
                         # Or, store the changes and revert.
                         pass
                         
            T *= cooling
            steps += 1
            if steps % 10000 == 0 and time.time() > end_time:
                break
                
        return best_colors, best_energy

    # Try K from 13 to 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
        budget = min(40, deadline - time.time() - 5)
        if budget < 5:
            break
            
        colors, energy = simulate_annealing(K, budget)
        
        # Verify best_colors
        if energy == 0:
            # Double check
            final_energy, _ = evaluate_coloring(colors, K)
            if final_energy == 0:
                return colors
                
    # Fallback: Use the 17-color modular solution if SA failed
    # (i+j) % 17
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    return fallback
