"""Improved program using Tomita MIS on a larger candidate pool."""
import itertools
import numpy as np
import sys
from collections import defaultdict

D = 11

# EVOLVE-BLOCK-START

def construct():
    # Generate candidate vectors: entries in {-1, 0, 1} with weight <= 3
    candidates = []
    indices = range(D)
    
    # Weight 1
    for i in indices:
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(indices, 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = s1
                v[j] = s2
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(v)
    
    n = len(candidates)
    
    # Precompute norms squared
    norms2 = np.array([sum(x*x for x in v) for v in candidates])
    
    # Build conflict graph
    # Conflict if ||u-v||^2 < max(||u||^2, ||v||^2)
    # We want an Independent Set in the conflict graph.
    # Adjacency list
    adj = [set() for _ in range(n)]
    
    # To speed up, we can use vectorized operations or simple loops.
    # n is ~1562. n^2 is ~2.4M. This is fast in Python if optimized or using NumPy.
    
    # Let's use NumPy for distance calculation
    C = np.array(candidates, dtype=np.int64)
    # Compute pairwise distances squared
    # dist2[i, j] = ||C[i] - C[j]||^2
    # We can compute this as norms2[:, None] + norms2[None, :] - 2 * C @ C.T
    
    dots = C @ C.T
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    
    # max_norm2 for pair (i, j) is max(norms2[i], norms2[j])
    max_norms2 = np.maximum(norms2[:, None], norms2[None, :])
    
    # Conflict mask: dist2 < max_norms2
    conflict_mask = dist2 < max_norms2
    
    # Diagonal is False (no self conflict in independent set context usually, but we don't pick self)
    # We build adjacency from this mask
    
    # Convert mask to adjacency lists
    for i in range(n):
        adj[i] = set(np.where(conflict_mask[i])[0])
        
    # Tomita BB-MCS Solver
    # Reference: "An improved branch and bound algorithm for the maximum clique problem"
    # Adapted for Maximum Independent Set (which is Max Clique on complement graph, 
    # but we can run Max Clique on the non-conflict graph directly, or Max Independent Set on conflict graph).
    # Let's run Max Clique on the "compatibility" graph (non-conflict).
    # Compat[i] = {j | not conflict[i][j]}
    
    compat = [set() for _ in range(n)]
    for i in range(n):
        # All nodes except i and its neighbors in conflict graph
        # compat[i] = set(range(n)) - adj[i] - {i}
        # This is O(n) per node, total O(n^2).
        pass # We'll do it more efficiently or just use the mask
        
    # Actually, let's just implement a simple recursive branch and bound for MIS on the conflict graph.
    # State: P (candidates to add), X (excluded), C (current set)
    # Bound: |C| + |P|
    # Better bound: Coloring of P.
    
    # Given time constraints, let's try a simpler greedy + local search first if Tomita is too complex to write correctly in few lines.
    # But the directive says Tomita.
    
    # Let's implement a standard Tomita coloring bound for MIS.
    # We will find the maximum independent set in the conflict graph.
    
    # To make it faster, we can reduce the graph.
    # Note: The 220 weight-2 vectors form an independent set.
    # We can start with this set and try to expand? No, MIS finds global max.
    
    # Implementation of Tomita's algorithm for Maximum Clique (on complement graph)
    # Complement graph edges: (i, j) if NOT conflict.
    # This graph is dense. Tomita is usually for sparse graphs? No, Tomita works well on both, 
    # but the coloring bound is tighter on sparse graphs?
    # Actually, the conflict graph is sparse (degree ~33). The compatibility graph is dense (degree ~1500).
    # Finding Max Independent Set in a sparse graph is equivalent to Max Clique in a dense graph.
    # Tomita is designed for Max Clique. It uses coloring on the candidate set P.
    # If P is dense, the coloring bound is weak (chromatic number is high, close to |P|).
    # If P is sparse, coloring bound is tight.
    # Here, the conflict graph is sparse. So the compatibility graph is dense.
    # Running Max Clique on the dense compatibility graph might be slow.
    
    # Alternative: Run Max Independent Set directly on the sparse conflict graph.
    # The bound is the chromatic number of the induced subgraph on P (in the conflict graph).
    # Since the conflict graph is sparse, the chromatic number is likely small (e.g., 3 or 4).
    # This would provide a very tight bound!
    
    # Let's implement MIS with coloring bound on the conflict graph.
    
    def color_sort(P, adj):
        # P is a list of nodes
        # Returns list of (node, color) sorted by color descending
        # Greedy coloring
        colors = {}
        for v in P:
            # Find smallest color not used by neighbors in P
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            
        # Sort by color descending
        sorted_P = sorted(P, key=lambda x: -colors[x])
        return sorted_P, colors

    def expand(P, X, C, adj):
        # P: list of nodes to consider
        # X: set of nodes excluded (for coloring? No, standard Tomita uses P and X for ordering)
        # C: current independent set
        # adj: adjacency list of conflict graph
        
        if not P:
            return C
        
        # Color P
        P_sorted, colors = color_sort(P, adj)
        
        # Branch on nodes in reverse color order (highest color first)
        # Actually, Tomita iterates from last to first in the sorted list (highest color)
        for i in range(len(P_sorted) - 1, -1, -1):
            v = P_sorted[i]
            c = colors[v]
            
            # Bound: if len(C) + c <= best_size, prune
            # We need a global best_size
            global best_size
            if len(C) + c <= best_size:
                break
            
            # Recurse
            new_C = C + [v]
            new_P = [u for u in P_sorted[:i] if u not in adj[v]]
            # Note: P_sorted[:i] contains nodes with color <= c, but we need to filter those adjacent to v.
            # Actually, standard Tomita uses P and X.
            # Let's stick to a simpler recursive MIS with greedy coloring bound.
            
            # To avoid deep recursion issues, we rely on the bound.
            expand(new_P, set(), new_C, adj)
            
            # Remove v from P
            P_sorted.pop(i)
            # Also, we should remove v from consideration in future branches?
            # The standard algorithm modifies P in place.
            
    # The above implementation is a bit messy. Let's use a more robust library-like implementation.
    # Given the 120s limit, if the solver is too slow, we should fallback.
    
    # Let's try a different approach: 
    # 1. Start with the 220 weight-2 vectors.
    # 2. Try to add weight-3 vectors one by one, removing conflicting weight-2 vectors if necessary, 
    #    and check if the total count increases.
    #    This is a local search / greedy approach.
    
    # However, the directive insists on MIS.
    # Let's implement a correct Tomita MIS.
    
    best_set = []
    
    # To ensure it finishes, we'll add a timeout or node limit.
    # But let's try the solver first.
    
    # Optimization: The conflict graph is static.
    
    # We will use a recursive function with a global counter for nodes visited.
    nodes_visited = 0
    TIMEOUT_NODES = 1000000 # Arbitrary limit
    
    def solve_mis(P, C, adj):
        nonlocal best_set, nodes_visited
        nodes_visited += 1
        if nodes_visited > TIMEOUT_NODES:
            return
        
        if not P:
            if len(C) > len(best_set):
                best_set = C[:]
            return
        
        # Greedy coloring of P
        # We want to color the induced subgraph on P.
        # We process nodes in some order.
        # Let's use the order from the previous level or arbitrary.
        
        # Simple greedy coloring:
        colors = {}
        P_list = list(P)
        for v in P_list:
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            
        # Sort P by color descending
        P_sorted = sorted(P_list, key=lambda x: -colors[x])
        
        # Iterate
        for i in range(len(P_sorted) - 1, -1, -1):
            v = P_sorted[i]
            c = colors[v]
            
            if len(C) + c <= len(best_set):
                break
            
            # Branch
            new_C = C + [v]
            # New P: nodes in P_sorted[0:i] that are not adjacent to v
            new_P = [u for u in P_sorted[:i] if u not in adj[v]]
            
            solve_mis(new_P, new_C, adj)
            
            # Remove v from P_sorted for next iteration?
            # In Tomita, we remove v from P and also from X?
            # Actually, we just move to the next node.
            # The list P_sorted is modified by popping.
            P_sorted.pop(i)
            
    # Initial call
    # P is all nodes.
    # This might be too slow if the bound is weak.
    # But for sparse graphs, coloring bound is strong.
    
    # Let's try it.
    # If it times out or finds nothing better than 220, we return 220.
    
    # To be safe, let's run it.
    # If best_set is empty or small, we might need a better initial guess.
    # We can initialize best_set with the 220 weight-2 vectors.
    
    # Identify indices of weight-2 vectors
    w2_indices = [i for i in range(n) if norms2[i] == 2]
    best_set = w2_indices[:]
    
    # Run solver
    # We pass the full set of nodes as P.
    # But we know w2_indices is a valid IS.
    
    # To speed up, we can prune branches that can't beat 220.
    # The coloring bound will handle this.
    
    solve_mis(list(range(n)), [], adj)
    
    if len(best_set) > 220:
        result = np.array([candidates[i] for i in best_set], dtype=np.int64)
        return result
    else:
        # Return the known good 220 set
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
