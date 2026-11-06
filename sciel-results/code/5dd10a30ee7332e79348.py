import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            idx += 1
    
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
    
    def count_k4_violations(colors):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for edge_idxs in k4s:
            seen = set()
            for ei in edge_idxs:
                seen.add(colors[ei])
            if len(seen) < 5:
                violations += 1
        return violations
    
    def count_distinct_colors(colors):
        return len(set(colors))
    
    def is_valid_coloring(colors):
        """Check if coloring is valid (all K4s have >= 5 distinct colors)."""
        for edge_idxs in k4s:
            seen = set()
            for ei in edge_idxs:
                seen.add(colors[ei])
            if len(seen) < 5:
                return False
        return True
    
    random.seed(42)
    
    best_colors = 999
    best_col_list = None
    
    # Try multiple initial colorings and local searches
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Generate initial coloring
        if restart == 0:
            # (i+j) % 17
            colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        elif restart == 1:
            # (i*3 + j) % 13
            colors = [(i*3 + j) % 13 for i in range(n) for j in range(i + 1, n)]
        elif restart == 2:
            # (i*5 + j*2) % 11
            colors = [(i*5 + j*2) % 11 for i in range(n) for j in range(i + 1, n)]
        else:
            # Random coloring with small number of colors
            num_init = max(13, 17 - restart // 5)
            colors = [random.randint(0, num_init - 1) for _ in range(120)]
        
        current_violations = count_k4_violations(colors)
        current_colors = count_distinct_colors(colors)
        
        # If valid and better, update best
        if current_violations == 0 and current_colors < best_colors:
            best_colors = current_colors
            best_col_list = colors[:]
        
        # Local search: try to reduce violations first, then reduce colors
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            used_colors = sorted(set(colors))
            
            # If we have violations, try to fix them
            if current_violations > 0:
                # Find a violating K4
                violating_k4 = None
                for edge_idxs in k4s:
                    seen = set()
                    for ei in edge_idxs:
                        seen.add(colors[ei])
                    if len(seen) < 5:
                        violating_k4 = edge_idxs
                        break
                
                if violating_k4 is None:
                    current_violations = 0
                    break
                
                # Try to change one edge in this K4 to add a new color
                improved = False
                for edge_idx in violating_k4:
                    if time.time() > deadline:
                        break
                    
                    old_color = colors[edge_idx]
                    # Try colors not currently in this K4
                    k4_colors = set(colors[ei] for ei in violating_k4)
                    candidate_colors = [c for c in range(20) if c not in k4_colors]
                    random.shuffle(candidate_colors)
                    
                    for new_color in candidate_colors[:5]:
                        colors[edge_idx] = new_color
                        
                        # Check if this reduces violations
                        new_violations = 0
                        for k4_idx in edge_to_k4s[edge_idx]:
                            edge_idxs = k4s[k4_idx]
                            seen = set()
                            for ei in edge_idxs:
                                seen.add(colors[ei])
                            if len(seen) < 5:
                                new_violations += 1
                        
                        if new_violations < current_violations:
                            current_violations = new_violations
                            improved = True
                            break
                        else:
                            colors[edge_idx] = old_color
                    
                    if improved:
                        break
                
                if not improved:
                    # Random perturbation
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        edge_idx = random.randint(0, 119)
                        old_color = colors[edge_idx]
                        new_color = random.randint(0, 19)
                        colors[edge_idx] = new_color
                        
                        new_violations = 0
                        for k4_idx in edge_to_k4s[edge_idx]:
                            edge_idxs = k4s[k4_idx]
                            seen = set()
                            for ei in edge_idxs:
                                seen.add(colors[ei])
                            if len(seen) < 5:
                                new_violations += 1
                        
                        if new_violations <= current_violations:
                            current_violations = new_violations
                        else:
                            colors[edge_idx] = old_color
                    
                    if current_violations == 0:
                        break
            else:
                # No violations, try to reduce number of colors
                if len(used_colors) <= best_colors:
                    break
                
                # Pick a color to try to eliminate - prefer colors with fewer edges
                color_counts = {}
                for c in used_colors:
                    color_counts[c] = sum(1 for x in colors if x == c)
                target_color = min(used_colors, key=lambda c: color_counts[c])
                
                # Find all edges with this color
                edges_with_color = [i for i in range(120) if colors[i] == target_color]
                
                if not edges_with_color:
                    continue
                
                # Try remapping each edge to another color
                improved = False
                for edge_idx in edges_with_color:
                    if time.time() > deadline:
                        break
                    
                    # Try changing this edge to each other color
                    for new_color in used_colors:
                        if new_color == target_color:
                            continue
                        
                        old_color = colors[edge_idx]
                        colors[edge_idx] = new_color
                        
                        # Check only K4s containing this edge
                        new_violations = 0
                        for k4_idx in edge_to_k4s[edge_idx]:
                            edge_idxs = k4s[k4_idx]
                            seen = set()
                            for ei in edge_idxs:
                                seen.add(colors[ei])
                            if len(seen) < 5:
                                new_violations += 1
                        
                        if new_violations == 0:
                            # Check if target_color is still in use
                            target_still_used = False
                            for i in range(120):
                                if colors[i] == target_color:
                                    target_still_used = True
                                    break
                            
                            if not target_still_used:
                                current_colors -= 1
                                if current_colors < best_colors:
                                    best_colors = current_colors
                                    best_col_list = colors[:]
                                improved = True
                                break
                            else:
                                colors[edge_idx] = old_color
                        else:
                            colors[edge_idx] = old_color
                    
                    if improved:
                        break
                
                if not improved:
                    # Try random perturbation to escape local optima
                    perturbed = False
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        edge_idx = random.randint(0, 119)
                        old_color = colors[edge_idx]
                        new_color = random.choice(used_colors)
                        colors[edge_idx] = new_color
                        
                        new_violations = 0
                        for k4_idx in edge_to_k4s[edge_idx]:
                            edge_idxs = k4s[k4_idx]
                            seen = set()
                            for ei in edge_idxs:
                                seen.add(colors[ei])
                            if len(seen) < 5:
                                new_violations += 1
                        
                        if new_violations == 0:
                            perturbed = True
                            break
                        else:
                            colors[edge_idx] = old_color
                    
                    if not perturbed:
                        break
        
        # After local search, check if we found a valid coloring
        if best_col_list is None:
            final_violations = count_k4_violations(colors)
            if final_violations == 0:
                final_colors = count_distinct_colors(colors)
                if final_colors < best_colors:
                    best_colors = final_colors
                    best_col_list = colors[:]
    
    # Return the best coloring found
    if best_col_list is None or count_k4_violations(best_col_list) > 0:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_col_list
