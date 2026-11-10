import random
import time
import math

# Precompute edges and K4s
N = 16
edges = []
edge_idx = [[-1]*N for _ in range(N)]
for i in range(N):
    for j in range(i+1, N):
        edge_idx[i][j] = len(edges)
        edge_idx[j][i] = len(edges)
        edges.append((i, j))

M = len(edges) # 120

k4s = []
# Map edge index to list of K4 indices
edge_to_k4s = [[] for _ in range(M)]

for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Edges in this K4
                e_ab = edge_idx[a][b]
                e_ac = edge_idx[a][c]
                e_ad = edge_idx[a][d]
                e_bc = edge_idx[b][c]
                e_bd = edge_idx[b][d]
                e_cd = edge_idx[c][d]
                
                k4_idx = len(k4s)
                k4s.append((e_ab, e_ac, e_ad, e_bc, e_bd, e_cd))
                
                for e in (e_ab, e_ac, e_ad, e_bc, e_bd, e_cd):
                    edge_to_k4s[e].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_colours = None
    best_score = float('inf')
    
    # Try to find valid colourings with decreasing K
    for K in [15, 14, 13]:
        if time.time() > deadline:
            break
            
        # Run Simulated Annealing for this K
        # Initialize
        colours = [random.randint(0, K-1) for _ in range(M)]
        
        # Calculate initial violations and colour counts
        violations = [0] * len(k4s)
        color_counts = [0] * K
        
        for c in colours:
            color_counts[c] += 1
        
        distinct_colors = sum(1 for c in color_counts if c > 0)
        
        for k4_idx, k4_edges in enumerate(k4s):
            cols = [colours[e] for e in k4_edges]
            if len(set(cols)) < 5:
                violations[k4_idx] = 1
        
        total_violations = sum(violations)
        
        # SA Parameters
        T = 10.0
        cooling = 0.999
        iterations = 0
        
        while time.time() < deadline:
            iterations += 1
            if iterations % 1000 == 0:
                # Check if we found a valid solution
                if total_violations == 0:
                    current_score = distinct_colors
                    if current_score < best_score:
                        best_score = current_score
                        best_colours = colours[:]
                        # Try to improve or break to try smaller K? 
                        # If we found valid with K, we can try to compress or move to next K.
                        # Since we iterate K downwards, finding valid with 15 allows us to try 14.
                        # But we should ensure we return the best valid one.
                        # If we found valid with K=15, distinct_colors might be < 15.
                        # We can break out of SA for this K and try next K.
                        break
                
            # Pick random edge
            e = random.randint(0, M-1)
            old_c = colours[e]
            
            # Pick new colour
            # Bias towards existing colours to reduce distinct count?
            # Or just random.
            new_c = random.randint(0, K-1)
            
            if new_c == old_c:
                continue
                
            # Calculate Delta Violations
            delta_viol = 0
            for k4_idx in edge_to_k4s[e]:
                k4_edges = k4s[k4_idx]
                # Get current colours
                cols = [colours[ee] for ee in k4_edges]
                old_distinct = len(set(cols))
                
                # Simulate change
                # Find index of e in k4_edges? 
                # It's faster to just rebuild set with new colour
                # But we need to be careful.
                # cols is a list of 6 ints.
                # We can just replace the value at the position of e.
                # Precompute position of e in each k4? 
                # Or just: new_cols = cols.copy(); new_cols[pos] = new_c
                # To avoid copy, we can do:
                # old_set = set(cols)
                # new_set = old_set.copy()
                # Remove old_c from set? No, other edges might have old_c.
                # It's safer to just count distinct in a list.
                # Since 6 is small, len(set(...)) is fine.
                
                # Optimization: 
                # old_viol = violations[k4_idx]
                # new_cols_list = [colours[ee] if ee != e else new_c for ee in k4_edges]
                # new_distinct = len(set(new_cols_list))
                # new_viol = 1 if new_distinct < 5 else 0
                # delta_viol += new_viol - old_viol
                
                # This is O(6) per K4. 91 K4s. O(546).
                # Acceptable.
                
                new_cols_list = [colours[ee] if ee != e else new_c for ee in k4_edges]
                new_distinct = len(set(new_cols_list))
                new_viol = 1 if new_distinct < 5 else 0
                delta_viol += new_viol - violations[k4_idx]
            
            # Calculate Delta Distinct Colours
            delta_distinct = 0
            # Update counts temporarily to check
            if color_counts[old_c] == 1:
                delta_distinct -= 1
            if color_counts[new_c] == 0:
                delta_distinct += 1
            
            # Objective Function
            # Score = total_violations * 1000 + distinct_colors
            old_score = total_violations * 1000 + distinct_colors
            new_score = (total_violations + delta_viol) * 1000 + (distinct_colors + delta_distinct)
            
            delta_score = new_score - old_score
            
            accept = False
            if delta_score < 0:
                accept = True
            else:
                # Simulated Annealing
                if T > 0:
                    prob = math.exp(-delta_score / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Apply move
                colours[e] = new_c
                
                # Update violations
                for k4_idx in edge_to_k4s[e]:
                    k4_edges = k4s[k4_idx]
                    new_cols_list = [colours[ee] for ee in k4_edges]
                    new_distinct = len(set(new_cols_list))
                    violations[k4_idx] = 1 if new_distinct < 5 else 0
                
                total_violations += delta_viol
                
                # Update colour counts
                color_counts[old_c] -= 1
                color_counts[new_c] += 1
                distinct_colors += delta_distinct
                
                T *= cooling
                
            # Early exit if valid and low score
            if total_violations == 0:
                current_score = distinct_colors
                if current_score < best_score:
                    best_score = current_score
                    best_colours = colours[:]
                # If we have a valid solution, we can try to polish or break
                # Breaking to try smaller K is good.
                break
        
        # If we found a valid solution with this K, we can try the next K
        # But if we didn't find one, we continue to next K (lower bound)
        
    # Fallback: Return seed if no better found
    if best_colours is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_colours

