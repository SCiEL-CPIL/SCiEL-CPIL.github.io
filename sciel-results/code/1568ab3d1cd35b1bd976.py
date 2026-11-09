import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Map (u,v) to edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    def count_colors_in_k4(k4_idx, colors):
        e_list = k4s[k4_idx]
        cset = set()
        for ei in e_list:
            cset.add(colors[ei])
        return len(cset)
    
    def is_k4_valid(k4_idx, colors):
        return count_colors_in_k4(k4_idx, colors) >= 5
    
    def is_all_valid(colors):
        for k4_idx in range(num_k4s):
            if not is_k4_valid(k4_idx, colors):
                return False
        return True
    
    def penalty_score(colors):
        """Count number of K4s with fewer than 5 distinct colors."""
        pen = 0
        for k4_idx in range(num_k4s):
            if count_colors_in_k4(k4_idx, colors) < 5:
                pen += 1
        return pen
    
    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)
    
    # Try direct construction with (i+j) mod p for various p
    best_colors = None
    best_count = float('inf')
    
    for p in [13, 14, 15, 16, 17, 18, 19, 20]:
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % p
        
        if is_all_valid(colors):
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    if best_colors is not None:
        return compact(best_colors)[0]
    
    # Simulated annealing with penalty-based approach
    # Start from a good initial solution
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
    
    best_colors = colors[:]
    best_count = len(set(colors))
    best_penalty = penalty_score(colors)
    
    # Try multiple initial seeds
    initial_solutions = []
    for p in [13, 14, 15, 16, 17]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i + j) % p
        initial_solutions.append(init)
    
    # Add random initial solutions
    for _ in range(5):
        init = [random.randint(0, 14) for _ in range(num_edges)]
        initial_solutions.append(init)
    
    # Run simulated annealing from each initial solution
    T_max = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    for init_sol in initial_solutions:
        if time.time() > deadline - 5.0:
            break
        
        colors = init_sol[:]
        current_penalty = penalty_score(colors)
        current_count = len(set(colors))
        
        if current_penalty < best_penalty:
            best_penalty = current_penalty
            best_count = current_count
            best_colors = colors[:]
        elif current_penalty == best_penalty and current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
        
        T = T_max
        while T > T_min and time.time() < deadline - 5.0:
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            # Try a new color
            if random.random() < 0.7:
                # Small perturbation: nearby color
                new_c = (old_c + random.randint(-2, 2)) % 15
            else:
                # Large perturbation: random color
                new_c = random.randint(0, 14)
            
            if new_c == old_c:
                T *= cooling_rate
                continue
            
            colors[e_idx] = new_c
            
            # Compute new penalty (only check affected K4s)
            new_penalty = current_penalty
            for k4_idx in edge_k4s[e_idx]:
                old_cnt = count_colors_in_k4(k4_idx, colors)
                # We need to compute the change carefully
                # Actually, let's just recompute penalty for affected K4s
                # This is complex, so let's use a simpler approach:
                # Track penalty incrementally
                pass
            
            # Simpler: recompute full penalty (slower but correct)
            new_penalty = penalty_score(colors)
            delta_pen = new_penalty - current_penalty
            
            # Acceptance criterion
            if delta_pen <= 0:
                current_penalty = new_penalty
                new_count = len(set(colors))
                if current_penalty < best_penalty or (current_penalty == best_penalty and new_count < best_count):
                    best_penalty = current_penalty
                    best_count = new_count
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta_pen / T):
                    current_penalty = new_penalty
                else:
                    colors[e_idx] = old_c
            
            T *= cooling_rate
    
    # Final local search polish: try to reduce color count
    if best_penalty == 0:
        # Try to reduce number of distinct colors
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = best_colors[e_idx]
            
            # Try to change to a color that's already used elsewhere
            used_colors = set(best_colors)
            if len(used_colors) > 13:
                target_color = random.choice(list(used_colors - {old_c})) if len(used_colors - {old_c}) > 0 else None
                if target_color is None:
                    continue
                best_colors[e_idx] = target_color
                
                valid = True
                for k4_idx in edge_k4s[e_idx]:
                    if not is_k4_valid(k4_idx, best_colors):
                        valid = False
                        break
                
                if valid:
                    pass
                else:
                    best_colors[e_idx] = old_c
    
    return compact(best_colors)[0]
