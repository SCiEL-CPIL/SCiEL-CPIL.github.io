"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 85.0  # safety margin
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
    
    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations
    
    def count_distinct_colors(coloring):
        return len(set(coloring))
    
    def greedy_construct(k, seed=None):
        """Build coloring greedily, position by position, choosing color that minimizes future violations."""
        if seed is not None:
            random.seed(seed)
        coloring = [None] * N
        for pos in range(N):
            best_color = None
            best_score = float('inf')
            # Try all k colors, score by how many violations this creates with already-colored positions
            for c in range(k):
                coloring[pos] = c
                # Count violations involving this position with already-colored positions
                violations = 0
                for ap_id in ap_indices[pos]:
                    ap = aps[ap_id]
                    # Check if all other positions are colored
                    others = [ap[j] for j in range(4) if ap[j] != pos]
                    if all(coloring[o] is not None for o in others):
                        colors_in_ap = [coloring[ap[j]] for j in range(4)]
                        if len(set(colors_in_ap)) < 3:
                            violations += 1
                coloring[pos] = None
                if violations < best_score:
                    best_score = violations
                    best_color = c
                elif violations == best_score:
                    # Tie-break: prefer less common color to spread usage
                    pass
            coloring[pos] = best_color
        return coloring
    
    def simulated_annealing(coloring, k, max_steps=50000, deadline_t=None):
        """Refine coloring using SA with incremental violation counting."""
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        
        while steps < max_steps and (deadline_t is None or time.time() < deadline_t):
            steps += 1
            idx = random.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = random.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta_violations == 0:
                    accept = random.random() < 0.5
                else:
                    if T > 0:
                        accept = random.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        return best_coloring, best_violations
    
    # Try k=3 first, then k=4, etc.
    for k in range(3, 129):
        if time.time() > deadline - 15.0:
            break
        
        found_valid = False
        best_coloring = None
        best_violations = float('inf')
        
        # Multiple restarts with different seeds
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Greedy construction with different seeds
            coloring = greedy_construct(k, seed=42 + restart * 1000)
            
            # Refine with SA
            refined, violations = simulated_annealing(coloring, k, max_steps=30000, deadline_t=deadline - 3.0)
            
            if violations < best_violations:
                best_violations = violations
                best_coloring = refined
            
            if violations == 0:
                found_valid = True
                break
        
        if found_valid or best_violations == 0:
            return best_coloring
    
    # Fallback: return a valid coloring with more colors
    # Use distinct colors to guarantee validity
    return list(range(N))
# EVOLVE-BLOCK-END
