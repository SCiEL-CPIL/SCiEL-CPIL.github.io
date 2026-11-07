"""Improved program using Simulated Annealing to maximize distinct induced degrees."""
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_MIN_V = P['minimum_vertices']
_EDGES = P['edges']

# Precompute adjacency sets and matrix for fast degree calculation
_ADJ_SETS = [set() for _ in range(_N)]
for u, v in _EDGES:
    _ADJ_SETS[u].add(v)
    _ADJ_SETS[v].add(u)

# Adjacency matrix for O(1) edge check
_ADJ_MAT = [[False] * _N for _ in range(_N)]
for u, v in _EDGES:
    _ADJ_MAT[u][v] = True
    _ADJ_MAT[v][u] = True

def _calc_distinct_degrees(vertices):
    """Calculate the number of distinct degrees in the induced subgraph."""
    if not vertices:
        return 0
    vset = set(vertices)
    degrees = []
    for v in vset:
        d = 0
        for u in vset:
            if _ADJ_MAT[v][u]:
                d += 1
        degrees.append(d)
    return len(set(degrees))

def _exhaustive_search_k(k):
    """Exhaustive search for a fixed size k. Only feasible for small k."""
    best_score = 0
    best_set = []
    if k > 10: # C(17, 10) is ~19448, C(17, 11) is ~12376, C(17, 12) is ~6188
        # We can actually do up to 12 or 13 quite fast in Python if optimized, 
        # but let's stick to 8-10 for safety and use SA for larger.
        # Actually C(17, 8) = 24310, C(17, 9) = 24310, C(17, 10) = 19448.
        # Let's do exhaustive for 8, 9, 10.
        pass
        
    for comb in combinations(range(_N), k):
        s = _calc_distinct_degrees(comb)
        if s > best_score:
            best_score = s
            best_set = list(comb)
    return best_set, best_score

def _simulated_annealing(k, restarts=50, max_iters=1000, T_start=1.0, T_end=0.01):
    """Simulated Annealing to find a good set of size k."""
    best_global_score = 0
    best_global_set = []
    
    for r in range(restarts):
        # Initialize with random set of size k
        current_set = set(random.sample(range(_N), k))
        current_score = _calc_distinct_degrees(current_set)
        
        best_local_score = current_score
        best_local_set = set(current_set)
        
        for i in range(max_iters):
            # Cooling
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Generate a neighbor
            move_type = random.random()
            if move_type < 0.33 and len(current_set) < _N:
                # Add a random vertex not in set
                candidates = [v for v in range(_N) if v not in current_set]
                if not candidates: continue
                v_add = random.choice(candidates)
                neighbor_set = current_set | {v_add}
            elif move_type < 0.66 and len(current_set) > _MIN_V:
                # Remove a random vertex from set
                if not current_set: continue
                v_rem = random.choice(list(current_set))
                neighbor_set = current_set - {v_rem}
            else:
                # Swap a random in/out pair
                if len(current_set) == 0 or len(current_set) == _N:
                    continue
                v_in = random.choice(list(current_set))
                candidates_out = [v for v in range(_N) if v not in current_set]
                if not candidates_out: continue
                v_out = random.choice(candidates_out)
                neighbor_set = (current_set - {v_in}) | {v_out}
            
            # Ensure we don't go below min size if we want to be strict, 
            # but for SA we can allow temporary dips if we track best valid.
            # However, the task requires returning at least 8. 
            # If we allow size < 8 during search, we must ensure final result is valid.
            # Let's constrain moves to keep size >= _MIN_V to simplify validity.
            if len(neighbor_set) < _MIN_V:
                continue
                
            neighbor_score = _calc_distinct_degrees(neighbor_set)
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = neighbor_set
                current_score = neighbor_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = set(current_set)
                    
        if best_local_score > best_global_score:
            best_global_score = best_local_score
            best_global_set = best_local_set
            
    return list(best_global_set), best_global_score

def construct():
    # 1. Exhaustive search for small sizes (8, 9, 10) to guarantee strong baseline
    best_overall_set = []
    best_overall_score = 0
    
    for k in range(_MIN_V, 11):
        s, score = _exhaustive_search_k(k)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = s
            
    # 2. Simulated Annealing for larger sizes (11 to 17)
    # We also re-run SA for 8-10 to see if it finds something better (unlikely if exhaustive is done, but good for consistency)
    # Actually, exhaustive is optimal for those sizes, so we skip SA for them.
    for k in range(11, _N + 1):
        s, score = _simulated_annealing(k, restarts=20, max_iters=500)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = s
            
    # Ensure we have a valid result
    if not best_overall_set or len(best_overall_set) < _MIN_V:
        # Fallback
        best_overall_set = list(range(_MIN_V))
        
    # Convert to list of Python ints
    return [int(v) for v in best_overall_set]
# EVOLVE-BLOCK-END
