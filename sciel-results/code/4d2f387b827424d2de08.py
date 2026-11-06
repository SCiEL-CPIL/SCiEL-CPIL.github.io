"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    target_n = 34
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
    
    # For each 4-set, the 4 triples that form K4^(3)
    k4_triple_indices = []
    for quad in combinations(range(target_n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
    
    # For each 5-set, the 10 triples that form K5^(3)
    k5_triple_indices = []
    for quint in combinations(range(target_n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
    
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Build reverse maps: triple -> which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
    
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)
    
    def compute_violations(red_k4_count, blue_k5_count):
        v = 0
        for i in range(num_k4):
            if red_k4_count[i] == 4:
                v += 1
        for i in range(num_k5):
            if blue_k5_count[i] == 10:
                v += 1
        return v
    
    def run_greedy(seed):
        """Greedy sequential construction with given random seed for tie-breaking."""
        rng = random.Random(seed)
        colors = [0] * num_triples
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for tri_idx in range(num_triples):
            if time.time() > deadline:
                return None
            
            # Evaluate both colors
            best_color = 0
            best_score = float('inf')
            
            for test_color in [0, 1]:
                score = 0
                if test_color == 0:  # red
                    for k4_i in triple_to_k4s[tri_idx]:
                        new_count = red_k4_count[k4_i] + 1
                        if new_count >= 3:
                            score += (new_count - 2) ** 2
                else:  # blue
                    for k5_i in triple_to_k5s[tri_idx]:
                        new_count = blue_k5_count[k5_i] + 1
                        if new_count >= 8:
                            score += (new_count - 7) ** 2
                
                # Add small random perturbation for tie-breaking
                score += rng.random() * 0.001
                
                if score < best_score:
                    best_score = score
                    best_color = test_color
            
            colors[tri_idx] = best_color
            if best_color == 0:
                for k4_i in triple_to_k4s[tri_idx]:
                    red_k4_count[k4_i] += 1
            else:
                for k5_i in triple_to_k5s[tri_idx]:
                    blue_k5_count[k5_i] += 1
        
        v = compute_violations(red_k4_count, blue_k5_count)
        return (v, colors, red_k4_count, blue_k5_count)
    
    # Try a few greedy runs with different seeds
    best_result = None
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        if time.time() > deadline:
            break
        result = run_greedy(seed)
        if result is None:
            break
        v, colors, r4c, b5c = result
        if best_result is None or v < best_result[0]:
            best_result = (v, colors, r4c, b5c)
        if v == 0:
            break
    
    if best_result is not None and best_result[0] == 0:
        return {'n': target_n, 'colors': best_result[1]}
    
    # If greedy didn't find valid solution, use SA starting from best greedy result
    if best_result is not None:
        sa_colors = best_result[1][:]
        red_k4_count = best_result[2][:]
        blue_k5_count = best_result[3][:]
    else:
        sa_colors = [random.randint(0, 1) for _ in range(num_triples)]
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        for tri_idx in range(num_triples):
            if sa_colors[tri_idx] == 0:
                for k4_i in triple_to_k4s[tri_idx]:
                    red_k4_count[k4_i] += 1
            else:
                for k5_i in triple_to_k5s[tri_idx]:
                    blue_k5_count[k5_i] += 1
    
    current_violations = compute_violations(red_k4_count, blue_k5_count)
    best_violations = current_violations
    best_colors = sa_colors[:]
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    no_improve_steps = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        t = random.randrange(num_triples)
        
        delta_v = 0
        old_color = sa_colors[t]
        
        if old_color == 0:  # red -> blue
            for k4_i in triple_to_k4s[t]:
                if red_k4_count[k4_i] == 4:
                    delta_v -= 1
            for k5_i in triple_to_k5s[t]:
                if blue_k5_count[k5_i] == 10:
                    delta_v -= 1
        else:  # blue -> red
            for k4_i in triple_to_k4s[t]:
                if red_k4_count[k4_i] == 3:
                    delta_v += 1
            for k5_i in triple_to_k5s[t]:
                if blue_k5_count[k5_i] == 9:
                    delta_v += 1
        
        if delta_v <= 0:
            accept = True
        else:
            if T > T_min:
                if random.random() < math.exp(-delta_v / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            sa_colors[t] = 1 - old_color
            if old_color == 0:
                for k4_i in triple_to_k4s[t]:
                    red_k4_count[k4_i] -= 1
                for k5_i in triple_to_k5s[t]:
                    blue_k5_count[k5_i] -= 1
            else:
                for k4_i in triple_to_k4s[t]:
                    red_k4_count[k4_i] += 1
                for k5_i in triple_to_k5s[t]:
                    blue_k5_count[k5_i] += 1
            
            current_violations += delta_v
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = sa_colors[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            T *= cooling
        else:
            no_improve_steps += 1
            T *= cooling
        
        if best_violations == 0:
            break
    
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
    
    # Fallback: small valid construction
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
