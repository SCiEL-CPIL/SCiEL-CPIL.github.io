"""
Strategy: Direct Maximum Independent Set (MIS) on the vector conflict graph.
Instead of restricting to supports with intersection <= 2 (which guarantees 
validity for all signs but is suboptimal), we construct the graph of all 
candidate integer vectors (norms 1-3) and solve for the MIS directly.
This allows mixing vectors with different norms and signs, potentially 
yielding a higher kissing number.
"""

import time
import numpy as np
from itertools import combinations, product

D = 11
DEADLINE = time.time() + 140  # Safety margin for 150s limit

def _generate_candidate_vectors():
    """
    Generate a set of candidate integer vectors.
    We include vectors with small norms to maximize packing.
    Norm squared can be 1, 2, 3.
    - Norm^2 = 1: 22 vectors (+/- e_i)
    - Norm^2 = 2: 2 * C(11,2) * 2^2 = 2 * 55 * 4 = 440 vectors (entries +/-1 on 2 indices)
    - Norm^2 = 3: 2 * C(11,3) * 2^3 = 2 * 165 * 8 = 2640 vectors (entries +/-1 on 3 indices)
    Total ~ 3100 vectors.
    """
    vecs = []
    
    # Norm^2 = 1
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            vecs.append(v)
            
    # Norm^2 = 2
    for i, j in combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                vecs.append(v)
                
    # Norm^2 = 3
    for i, j, k in combinations(range(D), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    vecs.append(v)
                    
    return np.array(vecs, dtype=np.int64)

def _build_conflict_graph(vecs):
    """
    Build adjacency matrix for the conflict graph.
    Two vectors u, v conflict if ||u - v||^2 < max(||u||^2, ||v||^2).
    Let M_u = ||u||^2, M_v = ||v||^2.
    Constraint: ||u - v||^2 >= max(M_u, M_v).
    ||u - v||^2 = M_u + M_v - 2 u.v
    So: M_u + M_v - 2 u.v >= max(M_u, M_v)
    
    Case 1: M_u >= M_v.
    M_u + M_v - 2 u.v >= M_u  =>  M_v - 2 u.v >= 0  =>  u.v <= M_v / 2
    
    Case 2: M_v > M_u.
    M_u + M_v - 2 u.v >= M_v  =>  M_u - 2 u.v >= 0  =>  u.v <= M_u / 2
    
    In general: u.v <= min(M_u, M_v) / 2.
    Since u.v is an integer, u.v <= floor(min(M_u, M_v) / 2).
    """
    n = len(vecs)
    norms_sq = np.sum(vecs**2, axis=1)
    # Dot product matrix
    # N=3100, D=11. 3100x11 @ 11x3100 is fast.
    dots = vecs @ vecs.T
    
    # min_norms_sq[i,j] = min(norms_sq[i], norms_sq[j])
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    
    # Threshold for conflict: dot > min_norms / 2
    # We want to find pairs where dot > min_norms / 2
    # Since dot and min_norms are integers, dot > min_norms/2 is equivalent to:
    # If min_norms is even: dot >= min_norms/2 + 1
    # If min_norms is odd: dot >= (min_norms+1)/2
    # Simpler: 2 * dot > min_norms
    
    conflict_mask = (2 * dots > min_norms)
    
    # No self-conflict
    np.fill_diagonal(conflict_mask, False)
    
    return conflict_mask

def _greedy_mis(adj, n):
    """
    Greedy heuristic for MIS.
    Returns a set of nodes (indices).
    """
    selected = set()
    removed = [False] * n
    
    # Order by degree (ascending) for better packing? Or random?
    # Let's try multiple orderings or a simple greedy with random restarts
    # For speed in the bound, we need a fast greedy.
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(int)
    
    # Greedy: pick node with min degree among remaining, remove neighbors
    # This is O(N^2) per restart, but N~3000. 3000^2 = 9M ops. Fast in numpy.
    
    remaining = set(range(n))
    current_degrees = degrees.copy()
    
    while remaining:
        # Find node in remaining with min degree
        # To avoid O(N) scan every time, we could use a heap, but for N=3000 simple scan is ok.
        # Actually, let's just pick a random node from remaining to allow diversity in restarts
        # Or min-degree is a standard heuristic.
        
        # For the bound in B&B, we need a *good* lower bound.
        # Min-degree greedy is usually decent.
        
        # Optimization: If we are in B&B, we want a fast lower bound.
        # Let's just do a simple greedy: pick first available, remove neighbors.
        # But randomization helps escape local optima.
        
        # Let's implement a randomized greedy
        nodes_list = list(remaining)
        np.random.shuffle(nodes_list)
        
        for u in nodes_list:
            if u not in remaining:
                continue
            selected.add(u)
            # Remove u and its neighbors from remaining
            neighbors = np.where(adj[u])[0]
            for v in neighbors:
                if v in remaining:
                    remaining.remove(v)
            remaining.remove(u)
            
    return selected

def _branch_and_bound(adj, n, deadline):
    """
    Branch and Bound for Maximum Independent Set.
    Uses a simple recursive search with pruning.
    """
    best_set = set()
    best_size = 0
    
    # Precompute degrees for ordering
    degrees = adj.sum(axis=1).astype(int)
    
    # Sort nodes by degree for better branching? 
    # Standard MIS B&B often works on a permutation of vertices.
    # Let's stick to original indices but use degree in heuristics.
    
    def search(current_set, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Pruning: upper bound
        # Simple bound: len(candidates) + len(current_set)
        # Better bound: Greedy coloring bound on the subgraph induced by candidates
        # For speed, we use a simpler bound: 
        # Max independent set size <= sum of 1/deg(v) + 1? No.
        # Let's use a greedy independent set on the candidate subgraph as a lower bound for the *remaining* part?
        # Actually, we want an upper bound on the *total* size to prune.
        # Upper bound = len(current_set) + len(candidates) is weak.
        
        # Let's try a simple recursive branching:
        # Pick a vertex v from candidates.
        # Branch 1: Include v. Remove v and its neighbors from candidates.
        # Branch 2: Exclude v. Remove v from candidates.
        
        # Heuristic: Pick v with highest degree in the induced subgraph?
        # To keep it fast, pick the first one or one with high degree.
        
        # Optimization: If candidates size + current size <= best_size, prune.
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Get the induced subgraph on candidates
        # To pick a pivot, let's pick the one with max degree in the candidate set
        # This is expensive to compute exactly every time.
        # Approximation: Use global degree.
        
        # Pick pivot
        # Sort candidates by global degree descending
        sorted_cands = sorted(candidates, key=lambda x: degrees[x], reverse=True)
        if not sorted_cands:
            return
        v = sorted_cands[0]
        
        # Branch 1: Include v
        new_current = current_set | {v}
        # New candidates: candidates minus v and neighbors of v
        neighbors = set(np.where(adj[v])[0])
        new_candidates = [u for u in candidates if u != v and u not in neighbors]
        
        search(new_current, new_candidates)
        
        # Branch 2: Exclude v
        new_candidates_excl = [u for u in candidates if u != v]
        search(current_set, new_candidates_excl)

    # Initial lower bound from greedy
    init_set = _greedy_mis(adj, n)
    best_set = init_set
    best_size = len(init_set)
    
    # Start search with all nodes
    search(set(), list(range(n)))
    
    return best_set

def construct():
    start_time = time.time()
    
    # 1. Generate candidates
    vecs = _generate_candidate_vectors()
    n = len(vecs)
    
    # 2. Build conflict graph
    adj = _build_conflict_graph(vecs)
    
    # 3. Solve MIS
    # We have ~3100 nodes. 
    # B&B on 3100 nodes is likely too slow for 120s if not very pruned.
    # However, the graph is sparse?
    # Degree of a node:
    # If v has norm^2=1 (e.g., e1), it conflicts with any u where u.v > 0.5 => u.v >= 1.
    # u.v >= 1 means u has a 1 at index 0.
    # How many such u? 
    # Norm 1: 1 (e1 itself, but no self loop. -e1 has dot -1, ok. So 0 conflicts from norm 1?)
    # Wait, if u=e1, v=e1, dot=1. min_norm=1. 2*1 > 1. Conflict.
    # So e1 conflicts with itself? No, diagonal is False.
    # e1 conflicts with any vector having +1 at index 0.
    # Norm 2: 2 vectors (+1 at 0, +1 at j; +1 at 0, -1 at j). 2*(D-1) = 20.
    # Norm 3: 4 vectors (+1 at 0, ...). 4 * C(10,2) = 4*45 = 180.
    # Total degree ~ 200.
    # N=3100, Avg Degree 200. 
    # B&B might struggle.
    
    # Alternative: Use a specialized MIS library? Not available.
    # Use a better heuristic: Local Search / Simulated Annealing on the MIS?
    # Or use the "Max Cut" reduction?
    
    # Let's try a faster approach: 
    # 1. Get a good initial solution with multiple greedy restarts.
    # 2. Use local search (swap moves) to improve.
    
    best_set = set()
    best_size = 0
    
    # Greedy restarts
    for i in range(50):
        if time.time() > DEADLINE - 10:
            break
        s = _greedy_mis(adj, n)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Local Search: Try to swap nodes in and out to increase size.
    # This is effectively finding a better independent set.
    # We can model this as: given current independent set S, 
    # can we find a node u not in S such that removing its neighbors from S 
    # and adding u increases the size?
    # i.e., |S \ N(u)| + 1 > |S|  =>  |S \cap N(u)| < 1.
    # Since S is independent, S \cap N(u) is the set of neighbors of u in S.
    # We want a u such that u has 0 neighbors in S. Then we can add u.
    # But if u has 0 neighbors in S, it means u is not adjacent to any node in S.
    # Then S U {u} is independent.
    # So, simply: while there exists a node u not in S with no neighbors in S, add it.
    # This is just completing the independent set greedily.
    
    # More powerful local search: 1-swap.
    # Remove u from S, add v.
    # Condition: v not in S, v not adjacent to (S \ {u}), and u not adjacent to v (if we keep u? No, we remove u).
    # Actually, we want to replace a "bad" node with a "good" one?
    # Or just find a larger independent set.
    
    # Let's try a simple hill climbing on the set of indices.
    # Represent set as a boolean array.
    # Try flipping bits.
    
    # Given time constraints, B&B with strong pruning is risky.
    # Let's stick to a robust Greedy + Local Search (Add-only first, then Swap).
    
    # Phase 1: Complete the set greedily
    current_set = set(best_set)
    # Find all nodes not in current_set
    candidates = [i for i in range(n) if i not in current_set]
    
    # Sort by degree
    candidates.sort(key=lambda x: degrees[x])
    
    for u in candidates:
        if time.time() > DEADLINE - 5:
            break
        # Check if u is adjacent to any in current_set
        # adj[u] is a boolean array. 
        # We need to check if any node in current_set has adj[u, node] == True.
        # This is slow in Python loop.
        # Vectorized: 
        # Let's maintain a boolean array 'in_set'
        pass
        
    # Let's implement a faster local search using numpy
    in_set = np.zeros(n, dtype=bool)
    for i in best_set:
        in_set[i] = True
        
    # Greedy completion
    # Iterate through nodes not in set
    not_in_set = np.where(~in_set)[0]
    # Sort by degree
    not_in_set = not_in_set[np.argsort(degrees[not_in_set])]
    
    for u in not_in_set:
        if time.time() > DEADLINE - 5:
            break
        # Check conflicts with in_set
        # adj[u] & in_set
        if not np.any(adj[u] & in_set):
            in_set[u] = True
            current_set.add(u)
            
    if np.sum(in_set) > best_size:
        best_size = np.sum(in_set)
        best_set = set(np.where(in_set)[0])
        
    # Phase 2: Swap moves (1-out, 1-in)
    # Try to find a node v not in set, and a node u in set, such that
    # removing u and adding v keeps independence and increases size? 
    # Size doesn't change in a 1-1 swap. We want to increase size.
    # So we need 1-out, 2-in? Or just find a larger independent set structure.
    
    # Actually, the "Add-only" greedy is often quite good for sparse graphs.
    # Let's do a few random restarts of the greedy + completion.
    
    for restart in range(20):
        if time.time() > DEADLINE - 5:
            break
        
        # Randomized greedy
        s = _greedy_mis(adj, n)
        in_set = np.zeros(n, dtype=bool)
        for i in s:
            in_set[i] = True
            
        # Complete
        not_in_set = np.where(~in_set)[0]
        np.random.shuffle(not_in_set)
        
        for u in not_in_set:
            if time.time() > DEADLINE - 5:
                break
            if not np.any(adj[u] & in_set):
                in_set[u] = True
                
        curr_size = np.sum(in_set)
        if curr_size > best_size:
            best_size = curr_size
            best_set = set(np.where(in_set)[0])
            
    # Construct result
    if not best_set:
        return np.zeros((0, D), dtype=np.int64)
        
    result = vecs[list(best_set)]
    return result

# EVOLVE-BLOCK-END
