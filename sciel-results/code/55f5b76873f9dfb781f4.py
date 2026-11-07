"""Rewritten program using Simulated Annealing to maximize distinct induced degrees."""
from itertools import combinations
from collections import defaultdict
import random
import math

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

_ADJ = _build_adj()
_N = P['n']
_MIN_V = P['minimum_vertices']

def _score(vertices):
    """Compute the number of distinct degrees in the induced subgraph."""
    vset = set(vertices)
    degs = set()
    for v in vertices:
        d = sum(1 for u in vset if u in _ADJ[v])
        degs.add(d)
    return len(degs)

def _simulated_annealing():
    """
    Perform Simulated Annealing to find a subset of vertices (size >= _MIN_V)
    that maximizes the number of distinct degrees.
    """
    random.seed(42)
    best_subset = None
    best_score = 0
    
    # We will run multiple independent SA runs
    n_runs = 100
    max_iterations = 5000
    initial_temp = 1.0
    cooling_rate = 0.99
    
    for run in range(n_runs):
        # Initialize with a random subset of size between _MIN_V and _N
        # Smaller subsets often have more distinct degrees relative to their size,
        # but larger subsets might offer more variety. Let's start around 10-12.
        size = random.randint(_MIN_V, _N)
        current_subset = set(random.sample(range(_N), size))
        current_score = _score(list(current_subset))
        
        if current_score > best_score:
            best_score = current_score
            best_subset = list(current_subset)
            
        temp = initial_temp
        
        for i in range(max_iterations):
            # Generate a neighbor
            # Move types:
            # 1. Add a vertex (if size < _N)
            # 2. Remove a vertex (if size > _MIN_V)
            # 3. Swap a vertex
            
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(current_subset) < _N:
                candidates = [v for v in range(_N) if v not in current_subset]
                if not candidates: continue
                v = random.choice(candidates)
                new_subset = current_subset | {v}
            elif move_type == 'remove' and len(current_subset) > _MIN_V:
                v = random.choice(list(current_subset))
                new_subset = current_subset - {v}
            elif move_type == 'swap' and len(current_subset) > 0:
                in_v = random.choice(list(current_subset))
                out_candidates = [v for v in range(_N) if v not in current_subset]
                if not out_candidates: continue
                out_v = random.choice(out_candidates)
                new_subset = (current_subset - {in_v}) | {out_v}
            else:
                # If move type is invalid (e.g., try to remove but at min size), try swap or add
                if len(current_subset) < _N:
                    candidates = [v for v in range(_N) if v not in current_subset]
                    if candidates:
                        v = random.choice(candidates)
                        new_subset = current_subset | {v}
                    else:
                        continue
                else:
                    # Must swap or remove
                    if len(current_subset) > _MIN_V:
                        v = random.choice(list(current_subset))
                        new_subset = current_subset - {v}
                    else:
                        in_v = random.choice(list(current_subset))
                        out_candidates = [v for v in range(_N) if v not in current_subset]
                        if not out_candidates: continue
                        out_v = random.choice(out_candidates)
                        new_subset = (current_subset - {in_v}) | {out_v}

            new_score = _score(list(new_subset))
            delta = new_score - current_score
            
            if delta > 0:
                current_subset = new_subset
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_subset = list(current_subset)
            else:
                # Accept with probability exp(delta / temp)
                # Since delta is integer and temp is float, this is standard
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_subset = new_subset
                    current_score = new_score
                    
            temp *= cooling_rate
            
            # Early termination if we've hit the theoretical max (size of subset)
            # Max distinct degrees for k vertices is k.
            if best_score == _N: # Can't have more distinct degrees than vertices
                return [int(v) for v in best_subset]

    return [int(v) for v in best_subset]

def construct():
    # Run Simulated Annealing
    result = _simulated_annealing()
    
    # Fallback: Ensure validity
    if not isinstance(result, list) or len(result) < _MIN_V or len(set(result)) != len(result):
        # If SA failed or returned garbage, fall back to a simple exhaustive search for size 8
        # This is a safety net, SA should be much better
        best = None
        best_score = 0
        for comb in combinations(range(_N), _MIN_V):
            s = _score(list(comb))
            if s > best_score:
                best_score = s
                best = list(comb)
        if best is None:
            best = list(range(_MIN_V))
        return [int(v) for v in best]
        
    return [int(v) for v in result]
# EVOLVE-BLOCK-END
