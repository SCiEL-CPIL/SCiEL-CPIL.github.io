import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute K4 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
        
    # Precompute K4 edge index arrays for fast access
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append([edge_idx[e] for e in k4e])

    # Map edge to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, idxs in enumerate(k4_edge_idxs):
        for idx in idxs:
            edge_to_k4s[idx].append(k4_idx)

    def count_colors(coloring):
        return len(set(coloring.tolist()))
    
    def check_valid(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        for idxs in k4_edge_idxs:
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    def count_violations(coloring):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for idxs in k4_edge_idxs:
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                violations += 1
        return violations

    def get_violating_k4s(coloring):
        """Return list of indices of K4s that are violating."""
        viol_k4s = []
        for k4_idx, idxs in enumerate(k4_edge_idxs):
            c0 = coloring[idxs[0]]
            c1 = coloring[idxs[1]]
            c2 = coloring[idxs[2]]
            c3 = coloring[idxs[3]]
            c4 = coloring[idxs[4]]
            c5 = coloring[idxs[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                viol_k4s.append(k4_idx)
        return viol_k4s

    best_coloring = None
    best_colors = 120  # worst case
    
    random.seed(42)
    
    # Try to find a valid coloring with K colors for K in 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
            
        # Initialize with (i+j)%K
        coloring = np.array([(i + j) % K for i in range(n) for j in range(i + 1, n)], dtype=int)
        
        # Repair phase
        # We want to minimize violations.
        # Local search: pick a violating K4, change a random edge in it to a random color in 0..K-1.
        # Accept if violations don't increase.
        
        current_violations = count_violations(coloring)
        
        for step in range(20000):
            if time.time() > deadline - 5:
                break
                
            if current_violations == 0:
                break
                
            viol_k4s = get_violating_k4s(coloring)
            if not viol_k4s:
                break
                
            # Pick a random violating K4
            k4_idx = random.choice(viol_k4s)
            idxs = k4_edge_idxs[k4_idx]
            
            # Pick a random edge in this K4
            edge_idx = random.choice(idxs)
            
            old_color = coloring[edge_idx]
            new_color = random.randint(0, K - 1)
            
            if new_color == old_color:
                continue
                
            coloring[edge_idx] = new_color
            
            # Check change in violations
            # Only K4s containing this edge are affected
            affected_k4s = edge_to_k4s[edge_idx]
            
            # Calculate violations for affected K4s
            new_violations_in_affected = 0
            for ak4_idx in affected_k4s:
                ak4_idxs = k4_edge_idxs[ak4_idx]
                c0 = coloring[ak4_idxs[0]]
                c1 = coloring[ak4_idxs[1]]
                c2 = coloring[ak4_idxs[2]]
                c3 = coloring[ak4_idxs[3]]
                c4 = coloring[ak4_idxs[4]]
                c5 = coloring[ak4_idxs[5]]
                if len({c0, c1, c2, c3, c4, c5}) < 5:
                    new_violations_in_affected += 1
            
            # We need the OLD violations in affected K4s to compare.
            # This is expensive to track perfectly without storing state.
            # Simpler: Just recalculate total violations? 1820 K4s.
            # 1820 * 6 lookups is fast.
            # Let's just recalculate total violations for simplicity, it's O(1820).
            new_total_violations = count_violations(coloring)
            
            if new_total_violations <= current_violations:
                current_violations = new_total_violations
                if current_violations == 0:
                    break
            else:
                # Revert
                coloring[edge_idx] = old_color
                # With some probability, accept worse state (Simulated Annealing)
                # But for now, let's stick to Hill Climbing to ensure we find a valid one.
                # If stuck, try random restarts or larger K.
                pass

        if current_violations == 0:
            n_colors = count_colors(coloring)
            if n_colors < best_colors:
                best_colors = n_colors
                best_coloring = coloring.copy()
                
            # If we found 13, we are done (optimal)
            if best_colors == 13:
                break

    # If no valid coloring found with 13-15, try 16
    if best_coloring is None:
        K = 16
        coloring = np.array([(i + j) % K for i in range(n) for j in range(i + 1, n)], dtype=int)
        if check_valid(coloring):
            best_coloring = coloring
            best_colors = count_colors(coloring)
        else:
            # Fallback: distinct colors
            coloring = np.arange(m, dtype=int)
            best_coloring = coloring
            best_colors = m

    # Final Polish: Try to reduce colors on the best coloring found
    if best_colors > 13:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        # Try to merge colors
        # Pick a color with frequency 1, try to change its edge to another color.
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            
            color_counts = {}
            for c in coloring:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            singleton_colors = [c for c, count in color_counts.items() if count == 1]
            
            if not singleton_colors:
                break
                
            target_color = random.choice(singleton_colors)
            target_idx = np.where(coloring == target_color)[0][0]
            
            other_colors = [c for c in color_counts.keys() if c != target_color]
            random.shuffle(other_colors)
            
            success = False
            for new_color in other_colors:
                old_color = coloring[target_idx]
                coloring[target_idx] = new_color
                
                if check_valid(coloring):
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_colors = new_count
                        best_coloring = coloring.copy()
                        success = True
                        break
                    else:
                        coloring[target_idx] = old_color
                        break
                else:
                    coloring[target_idx] = old_color
                    # If one fails, try next?
                    # Trying all is expensive. Let's break.
                    break
            
            if success:
                continue
            else:
                # If we can't remove this singleton, try another or stop
                pass

    # Return the best coloring found
    result = [int(c) for c in best_coloring]
    # Remap colors to 0..K-1
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
