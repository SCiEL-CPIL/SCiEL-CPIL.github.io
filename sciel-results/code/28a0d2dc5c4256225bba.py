import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    max_time = 140.0  # Leave 10s margin for 150s limit
    
    # Precompute all triples and their indices for fast access
    def build_triple_index(n):
        """Map (a,b,c) -> index in lex-ordered list of all triples"""
        triple_list = []
        triple_idx = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_idx[(a,b,c)] = len(triple_list)
                    triple_list.append((a,b,c))
        return triple_list, triple_idx
    
    # Precompute for all n up to 40
    triple_data = {}
    for n in range(1, 41):
        triple_data[n] = build_triple_index(n)
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    def count_violations(colors, n):
        """Count red K4 and blue K5 violations. Returns (red_viol, blue_viol)."""
        triple_list, triple_idx = triple_data[n]
        C = np.array(colors, dtype=np.int8)
        
        # Build adjacency-like structure: for each pair (i,j), which triples contain it
        # Actually, let's just iterate over all 4-subsets for red and 5-subsets for blue
        red_viol = 0
        blue_viol = 0
        
        # Red K4: all 4 triples of a 4-set are red
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (C[triple_idx[(a,b,c)]] == 0 and 
                C[triple_idx[(a,b,d)]] == 0 and 
                C[triple_idx[(a,c,d)]] == 0 and 
                C[triple_idx[(b,c,d)]] == 0):
                red_viol += 1
        
        # Blue K5: all 10 triples of a 5-set are blue
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            all_blue = True
            for triple in combinations(quint, 3):
                if C[triple_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_viol += 1
        
        return red_viol, blue_viol
    
    def greedy_construct(n, p_red=0.6, seed=0):
        """Greedy construction with biased random choices."""
        rng = random.Random(seed)
        C = [[[-1 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        
        for v in range(n):
            for i in range(v):
                for j in range(i + 1, v):
                    # Check if Red (0) is forbidden
                    red_forbidden = False
                    for k in range(i):
                        if C[k][i][j] == 0 and C[k][i][v] == 0 and C[k][j][v] == 0:
                            red_forbidden = True
                            break
                    
                    # Check if Blue (1) is forbidden
                    blue_forbidden = False
                    for k in range(i):
                        for l in range(k + 1, i):
                            if (C[k][l][i] == 1 and C[k][l][j] == 1 and C[k][l][v] == 1 and
                                C[k][i][j] == 1 and C[k][i][v] == 1 and C[k][j][v] == 1 and
                                C[l][i][j] == 1 and C[l][i][v] == 1 and C[l][j][v] == 1):
                                blue_forbidden = True
                                break
                        if blue_forbidden:
                            break
                    
                    if red_forbidden and blue_forbidden:
                        return None
                    elif red_forbidden:
                        C[i][j][v] = 1
                    elif blue_forbidden:
                        C[i][j][v] = 0
                    else:
                        C[i][j][v] = 0 if rng.random() < p_red else 1
        
        # Extract colors in lex order
        colors = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    colors.append(C[a][b][c])
        return colors
    
    def local_search(colors, n, time_budget=10.0, seed=0):
        """Simulated annealing local search to eliminate violations."""
        rng = random.Random(seed)
        triple_list, triple_idx = triple_data[n]
        C = np.array(colors, dtype=np.int8)
        
        # Precompute: for each triple, which 4-sets and 5-sets it belongs to
        # For faster violation counting after a flip
        # triple_to_quads[t] = list of 4-tuples (as sorted 4-tuples) containing triple t
        # triple_to_quints[t] = list of 5-tuples containing triple t
        
        # Build mapping from triple index to list of 4-sets it's in
        triple_to_quads = [[] for _ in range(len(triple_list))]
        for quad in combinations(range(n), 4):
            for t in combinations(quad, 3):
                triple_to_quads[triple_idx[t]].append(quad)
        
        triple_to_quints = [[] for _ in range(len(triple_list))]
        for quint in combinations(range(n), 5):
            for t in combinations(quint, 3):
                triple_to_quints[triple_idx[t]].append(quint)
        
        # Precompute quad_triples: for each 4-set, the 4 triple indices
        quad_triples = []
        for quad in combinations(range(n), 4):
            ts = [triple_idx[t] for t in combinations(quad, 3)]
            quad_triples.append(ts)
        
        # Precompute quint_triples: for each 5-set, the 10 triple indices
        quint_triples = []
        for quint in combinations(range(n), 5):
            ts = [triple_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(ts)
        
        # Count initial violations
        def count_viol_fast(C):
            red_viol = 0
            for ts in quad_triples:
                if C[ts[0]] == 0 and C[ts[1]] == 0 and C[ts[2]] == 0 and C[ts[3]] == 0:
                    red_viol += 1
            blue_viol = 0
            for ts in quint_triples:
                all_blue = True
                for ti in ts:
                    if C[ti] != 1:
                        all_blue = False
                        break
                if all_blue:
                    blue_viol += 1
            return red_viol, blue_viol
        
        red_viol, blue_viol = count_viol_fast(C)
        total_viol = red_viol + blue_viol
        if total_viol == 0:
            return C.tolist()
        
        best_C = C.copy()
        best_total = total_viol
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        deadline = time.time() + time_budget
        
        while time.time() < deadline and T > min_T:
            # Pick a random triple to flip
            t_idx = rng.randint(0, len(triple_list) - 1)
            
            # Compute change in violations if we flip triple t_idx
            old_color = C[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Check quads containing this triple
            for quad in triple_to_quads[t_idx]:
                # Find the quad's triple list
                # We need to know which quad index this is
                # Actually, let's just recompute the quad's violation status
                pass
            
            # Simpler approach: just flip and count the difference
            # But that's O(1) per quad/quint, which is fine
            C[t_idx] = new_color
            new_red, new_blue = count_viol_fast(C)
            new_total = new_red + new_blue
            delta = new_total - total_viol
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                total_viol = new_total
                if total_viol < best_total:
                    best_total = total_viol
                    best_C = C.copy()
                    if best_total == 0:
                        break
            else:
                C[t_idx] = old_color
            
            T *= cooling
        
        return best_C.tolist()
    
    # Try to build larger and larger n
    for target_n in range(10, 41):
        if time.time() - start_time > max_time:
            break
        
        # Try multiple greedy constructions with different seeds and biases
        found = False
        for seed in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Try different red biases
            for p_red in [0.5, 0.6, 0.7]:
                colors = greedy_construct(target_n, p_red=p_red, seed=seed)
                if colors is None:
                    continue
                
                # Check if valid
                red_v, blue_v = count_violations(colors, target_n)
                if red_v == 0 and blue_v == 0:
                    if target_n > best_n:
                        best_n = target_n
                        best_colors = colors
                    found = True
                    break
                else:
                    # Try local search
                    remaining_time = max_time - (time.time() - start_time)
                    if remaining_time > 5:
                        search_budget = min(5.0, remaining_time * 0.5)
                        improved = local_search(colors, target_n, time_budget=search_budget, seed=seed)
                        red_v2, blue_v2 = count_violations(improved, target_n)
                        if red_v2 == 0 and blue_v2 == 0:
                            if target_n > best_n:
                                best_n = target_n
                                best_colors = improved
                            found = True
                            break
        
        if found:
            # Try to go higher
            continue
        else:
            # Can't build this n, try next
            continue
    
    return {'n': best_n, 'colors': best_colors}
