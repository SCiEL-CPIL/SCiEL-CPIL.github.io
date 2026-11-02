import numpy as np
import itertools
import time
import random

D = 11

def _generate_weight_2_vectors():
    """Generate all vectors with exactly two non-zero entries, each +/-1."""
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(np.array(v, dtype=np.int64))
    return vectors

def _generate_weight_3_vectors():
    """Generate all vectors with exactly three non-zero entries, each +/-1."""
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            vectors.append(np.array(v, dtype=np.int64))
    return vectors

def _build_conflict_graph(vectors):
    """
    Build adjacency list for conflict graph.
    Conflict if ||u-v||^2 < max_norm_sq.
    Here all vectors have norm_sq = 3.
    So conflict if ||u-v||^2 < 3.
    ||u-v||^2 = 6 - 2u.dot(v).
    6 - 2u.dot(v) < 3  =>  3 < 2u.dot(v)  =>  u.dot(v) > 1.5.
    Since u.dot(v) is integer, u.dot(v) >= 2.
    """
    n = len(vectors)
    adj = [[] for _ in range(n)]
    # Precompute dot products? Or just check on the fly.
    # For n=1320, n^2 is 1.7M.
    # 1.7M dot products of length 11 is fast enough in numpy?
    # Let's use a matrix.
    V = np.array(vectors)
    # Compute all pairwise dot products
    # Dot = V @ V.T
    # But we only need to check u.dot(v) >= 2.
    # We can do this in a loop to save memory if needed, but 1320x1320 is small.
    dot_matrix = V @ V.T
    
    for i in range(n):
        for j in range(i+1, n):
            if dot_matrix[i, j] >= 2:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _greedy_mis(adj, order):
    """
    Greedy Maximum Independent Set.
    order: list of vertex indices to consider in this order.
    Returns list of selected vertices.
    """
    n = len(adj)
    selected = []
    in_set = [False] * n
    # We process vertices in the given order.
    # If a vertex is not "blocked" by a previously selected neighbor, we select it.
    # A vertex is blocked if any of its neighbors is selected.
    # Actually, standard greedy:
    # Iterate through order. If v is not adjacent to any selected vertex, select v.
    
    selected_set = set()
    for v in order:
        # Check if v is adjacent to any selected vertex
        conflict = False
        for u in adj[v]:
            if u in selected_set:
                conflict = True
                break
        if not conflict:
            selected_set.add(v)
            selected.append(v)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Baseline: Weight 2 vectors (220)
    w2_vectors = _generate_weight_2_vectors()
    best_solution = w2_vectors
    best_size = len(best_solution)
    
    # 2. Try Weight 3 vectors
    w3_vectors = _generate_weight_3_vectors()
    n = len(w3_vectors)
    
    if time.time() > deadline:
        return np.array(best_solution, dtype=np.int64)
        
    adj = _build_conflict_graph(w3_vectors)
    
    if time.time() > deadline:
        return np.array(best_solution, dtype=np.int64)
        
    # Random Greedy with multiple restarts
    best_mis_size = 0
    best_mis_indices = []
    
    # We can also try a simple greedy first
    # Sort by degree (descending)? Or ascending?
    # Usually, picking high degree vertices first removes more options, so pick low degree?
    # Or just random.
    
    max_restarts = 500
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        order = list(range(n))
        random.shuffle(order)
        
        # Optimization: We can skip if current best is already very high
        # Greedy is fast.
        mis_indices = _greedy_mis(adj, order)
        
        if len(mis_indices) > best_mis_size:
            best_mis_size = len(mis_indices)
            best_mis_indices = mis_indices
            
            # If we beat the baseline, update best_solution
            if best_mis_size > best_size:
                best_size = best_mis_size
                best_solution = [w3_vectors[idx] for idx in best_mis_indices]
                
                # If we get significantly better, we might stop?
                # Or continue to see if we can go higher.
                
    # Local Search on the best MIS?
    # Since it's an independent set, we can't add more.
    # We can try to swap? No, swapping doesn't change size.
    # So random restarts is the way.
    
    # Verification
    if len(best_solution) > 1:
        norms_sq = [np.dot(v, v) for v in best_solution]
        max_ns = max(norms_sq)
        valid = True
        # Quick check for a few pairs? Or all?
        # If size is large, O(N^2) check might be slow.
        # But N is at most 1320. 1320^2 / 2 = 870k checks.
        # Each check is a dot product of length 11.
        # 870k * 11 ops = 9.5M ops. Fast in Python? Maybe 1-2 seconds.
        # Let's do it to be safe.
        for i in range(len(best_solution)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(best_solution)):
                diff = best_solution[i] - best_solution[j]
                if np.dot(diff, diff) < max_ns:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to weight 2
            best_solution = w2_vectors
            best_size = len(best_solution)
            
    return np.array(best_solution, dtype=np.int64)
