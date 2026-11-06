"""Edge-colourings of K16 in which every K4 sees five colours."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    """Return 120 integer colours for edges of K16 (lex order i<j).
    Every K4 must span at least 5 distinct colours. Minimise max colour used."""
    t0 = time.time()
    deadline = t0 + 118.0  # safety margin before 120s kill

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120

    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s and their edges
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4_edges.append([edge_idx[x] for x in e])
    num_k4 = len(k4s)  # 1820

    # Precompute: for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(m)]
    for ki, ke in enumerate(k4_edges):
        for ei in ke:
            edge_to_k4s[ei].append(ki)

    # Initial solution: (i+j) mod 17
    sol = np.array([(i + j) % 17 for i, j in edges], dtype=np.int32)

    def count_colors_in_k4(s, ke):
        return len(set([s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]]]))

    def is_valid(s):
        for ke in k4_edges:
            if len(set([s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]]])) < 5:
                return False
        return True

    def is_valid_after_change(s, edge_i, new_c):
        """Check validity after changing edge_i to new_c. Only check K4s containing edge_i."""
        for ki in edge_to_k4s[edge_i]:
            ke = k4_edges[ki]
            if len(set([s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]]])) < 5:
                return False
        return True

    # Local search to minimize max color
    def local_search(s, time_limit):
        s = s.copy()
        current_max = int(np.max(s))
        
        # Phase 1: Greedy reduction - try to reduce max color
        for _ in range(100):
            if time.time() > time_limit:
                break
            
            current_max = int(np.max(s))
            if current_max < 13:
                break
            
            # Find edges with current_max color
            max_edges = [idx for idx in range(m) if s[idx] == current_max]
            random.shuffle(max_edges)
            
            improved = False
            for idx in max_edges:
                if time.time() > time_limit:
                    break
                old_c = s[idx]
                # Try colors from 0 to current_max-1, random order
                candidates = list(range(current_max))
                random.shuffle(candidates)
                
                for new_c in candidates:
                    s[idx] = new_c
                    if is_valid_after_change(s, idx, new_c):
                        improved = True
                        break
                    else:
                        s[idx] = old_c
                
                if improved:
                    break
            
            if not improved:
                # Try swapping colors between two edges
                all_edges = list(range(m))
                random.shuffle(all_edges)
                swap_improved = False
                for i in range(min(200, len(all_edges))):
                    if time.time() > time_limit:
                        break
                    e1 = all_edges[i]
                    for j in range(i+1, min(i+50, len(all_edges))):
                        if time.time() > time_limit:
                            break
                        e2 = all_edges[j]
                        c1, c2 = s[e1], s[e2]
                        if c1 == c2:
                            continue
                        s[e1], s[e2] = c2, c1
                        if is_valid_after_change(s, e1, c2) and is_valid_after_change(s, e2, c1):
                            swap_improved = True
                            break
                        else:
                            s[e1], s[e2] = c1, c2
                    if swap_improved:
                        break
                
                if not improved and not swap_improved:
                    break
        
        # Phase 2: Try to eliminate specific colors
        for target_color in range(int(np.max(s)), 12, -1):
            if time.time() > time_limit:
                break
            
            edges_with_color = [idx for idx in range(m) if s[idx] == target_color]
            if not edges_with_color:
                continue
            
            random.shuffle(edges_with_color)
            for idx in edges_with_color:
                if time.time() > time_limit:
                    break
                old_c = s[idx]
                candidates = [c for c in range(target_color) if c != old_c]
                random.shuffle(candidates)
                
                for new_c in candidates:
                    s[idx] = new_c
                    if is_valid_after_change(s, idx, new_c):
                        break
                    else:
                        s[idx] = old_c
        
        return s, int(np.max(s))

    best_sol = sol.copy()
    best_max = int(np.max(best_sol))

    # Multiple restarts with different initial solutions
    random.seed(42)
    
    # Try the base solution first
    if time.time() < deadline:
        s2, max2 = local_search(best_sol, deadline - 1.0)
        if max2 < best_max:
            best_sol = s2
            best_max = max2

    # Try perturbed versions
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
        
        # Create perturbed version
        perturbed = best_sol.copy()
        for _ in range(random.randint(5, 20)):
            idx = random.randint(0, m-1)
            old_c = perturbed[idx]
            new_c = random.randint(0, best_max)
            perturbed[idx] = new_c
            if not is_valid_after_change(perturbed, idx, new_c):
                perturbed[idx] = old_c
        
        s3, max3 = local_search(perturbed, deadline - 1.0)
        if max3 < best_max:
            best_sol = s3
            best_max = max3

    # Final polish: try to reduce further
    if time.time() < deadline - 1.0:
        s4, max4 = local_search(best_sol, deadline - 0.5)
        if max4 < best_max:
            best_sol = s4
            best_max = max4

    # Convert to list of ints
    return [int(c) for c in best_sol]

# EVOLVE-BLOCK-END
