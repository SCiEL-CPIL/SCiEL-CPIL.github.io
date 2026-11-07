"""Improved program using Simulated Annealing to maximize distinct induced degrees."""
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

def _score_mask(mask):
    """Compute the number of distinct degrees in the induced subgraph defined by mask."""
    # Count bits to ensure minimum size constraint is considered in scoring or handled outside
    # For SA, we want to maximize distinct degrees. We can penalize size < _MIN_V.
    if mask.bit_count() < _MIN_V:
        return -1000 + mask.bit_count() # Heavy penalty for invalid size
    
    degs = set()
    for v in range(_N):
        if (mask >> v) & 1:
            # Count neighbors in the induced subgraph
            d = 0
            for u in _ADJ[v]:
                if (mask >> u) & 1:
                    d += 1
            degs.add(d)
    return len(degs)

def construct():
    random.seed(42)
    
    # Initial best solution: a random valid set
    best_mask = 0
    for _ in range(_MIN_V):
        v = random.randint(0, _N - 1)
        best_mask |= (1 << v)
    
    # If random didn't give exactly _MIN_V distinct (unlikely with set logic but just in case)
    # Actually, let's just pick the first _MIN_V vertices for a deterministic start
    best_mask = (1 << _MIN_V) - 1
    best_score = _score_mask(best_mask)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.95
    steps = 20000
    
    current_mask = best_mask
    current_score = best_score
    
    for i in range(steps):
        # Generate neighbor: flip a random bit
        v = random.randint(0, _N - 1)
        neighbor_mask = current_mask ^ (1 << v)
        
        neighbor_score = _score_mask(neighbor_mask)
        
        delta = neighbor_score - current_score
        
        if delta >= 0:
            # Accept if better or equal
            current_mask = neighbor_mask
            current_score = neighbor_score
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_mask = neighbor_mask
                current_score = neighbor_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Extract vertices from best_mask
    result = []
    for v in range(_N):
        if (best_mask >> v) & 1:
            result.append(v)
            
    # Ensure we have at least _MIN_V vertices
    # If SA ended up with too few (should be rare due to penalty), add random vertices
    while len(result) < _MIN_V:
        # Find a vertex not in result
        for v in range(_N):
            if v not in result:
                result.append(v)
                break
                
    return [int(v) for v in result]
# EVOLVE-BLOCK-END
