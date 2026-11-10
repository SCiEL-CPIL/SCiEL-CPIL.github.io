# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.optimize import milp, Bounds, LinearConstraint

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph (edges connect pairs with forbidden difference).
    2. Use a hybrid approach:
       a. Run MILP to get an exact or high-quality initial solution.
       b. Use a specialized local search (1-add, 1-swap, 2-for-1, 1-for-2) to try to improve 
          upon the MILP solution, in case the solver hit a time limit or returned a 
          sub-optimal solution due to numerical issues or solver settings.
       c. If MILP fails or times out, use randomized greedy + local search.
       d. Additionally, use a bitmask-based branch-and-bound (Tomita algorithm style) for 
          Maximum Independent Set (MIS) to potentially find the exact optimum or a better 
          bound, using the current best solution as a lower bound for pruning.
    3. The local search is optimized for speed using bitsets (integers) for adjacency.
    """
    m = 205
    k = 2
    
    # Step 1: Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency as bitmasks for fast local search
    # adj_mask[i] is an integer where bit j is set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute the list of forbidden pairs for MILP constraints
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            if (adj_mask[i] >> j) & 1:
                edges.append((i, j))
    
    def is_valid(subset):
        if len(subset) == 0:
            return False
        if len(set(subset)) != len(subset):
            return False
        if any(x < 0 or x >= m for x in subset):
            return False
        s = set(subset)
        for a in subset:
            for b in subset:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True

    best_set = []
    best_size = 0
    
    def update_best(subset):
        nonlocal best_set, best_size
        if is_valid(subset):
            if len(subset) > best_size:
                best_set = list(subset)
                best_size = len(subset)

    # --- Local Search Helper ---
    # Uses bitmasks for speed.
    # state: set of nodes in current solution
    # We try to add a node, or swap one node for another, etc.
    
    def local_search(initial_set, max_iter=5000):
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try 1-move: Add a node
            # A node v can be added if it has no neighbors in current set.
            # i.e., adj_mask[v] & current_mask == 0
            for v in range(m):
                if v not in current:
                    if (adj_mask[v] & current_mask) == 0:
                        current.add(v)
                        current_mask |= (1 << v)
                        improved = True
                        break
            
            # Try 2-move: Swap one node out, one in
            # Remove u, add v.
            # Valid if:
            # 1. u in current, v not in current
            # 2. v is not adjacent to any node in (current - {u})
            #    i.e., adj_mask[v] & (current_mask ^ (1 << u)) == 0
            
            if not improved:
                # Try to find a swap
                found_swap = False
                for u in list(current):
                    # Mask of current set without u
                    mask_without_u = current_mask ^ (1 << u)
                    # Find v not in current such that adj_mask[v] & mask_without_u == 0
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            # Swap u with v
                            current.remove(u)
                            current.add(v)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v)
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            # Try 3-move: Remove 2 nodes, add 1 node (2-for-1 swap)
            # This can help escape local optima where no single swap works.
            if not improved and len(current) >= 2:
                found_2for1 = False
                u_list = list(current)
                for i_idx in range(len(u_list)):
                    u1 = u_list[i_idx]
                    for j_idx in range(i_idx + 1, len(u_list)):
                        u2 = u_list[j_idx]
                        # Mask of current set without u1 and u2
                        mask_without_u1u2 = current_mask ^ (1 << u1) ^ (1 << u2)
                        # Find v not in current such that adj_mask[v] & mask_without_u1u2 == 0
                        for v in range(m):
                            if v in current:
                                continue
                            if (adj_mask[v] & mask_without_u1u2) == 0:
                                # Also ensure v is not adjacent to u1 or u2 (already checked by mask)
                                # Remove u1, u2; add v
                                current.remove(u1)
                                current.remove(u2)
                                current.add(v)
                                current_mask ^= (1 << u1)
                                current_mask ^= (1 << u2)
                                current_mask |= (1 << v)
                                improved = True
                                found_2for1 = True
                                break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
            
            # Try 4-move: Remove 1 node, add 2 nodes (1-for-2 swap)
            # This can increase the set size if the removed node is "blocking" two potential additions.
            if not improved and len(current) >= 1:
                found_1for2 = False
                for u in list(current):
                    # Mask of current set without u
                    mask_without_u = current_mask ^ (1 << u)
                    # Find two nodes v1, v2 not in current such that:
                    # 1. v1 is compatible with (current - {u})
                    # 2. v2 is compatible with (current - {u})
                    # 3. v1 is compatible with v2
                    candidates = []
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            candidates.append(v)
                    
                    # Try all pairs from candidates
                    for idx1 in range(len(candidates)):
                        v1 = candidates[idx1]
                        for idx2 in range(idx1 + 1, len(candidates)):
                            v2 = candidates[idx2]
                            # Check if v1 and v2 are compatible
                            if (adj_mask[v1] >> v2) & 1:
                                continue
                            
                            # Remove u; add v1, v2
                            current.remove(u)
                            current.add(v1)
                            current.add(v2)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v1)
                            current_mask |= (1 << v2)
                            improved = True
                            found_1for2 = True
                            break
                        if found_1for2:
                            break
                    if found_1for2:
                        break
            
            # Try 5-move: Remove 2 nodes, add 3 nodes (2-for-3 swap)
            # This can further increase the set size.
            if not improved and len(current) >= 2:
                found_2for3 = False
                u_list = list(current)
                for i_idx in range(len(u_list)):
                    u1 = u_list[i_idx]
                    for j_idx in range(i_idx + 1, len(u_list)):
                        u2 = u_list[j_idx]
                        # Mask of current set without u1 and u2
                        mask_without_u1u2 = current_mask ^ (1 << u1) ^ (1 << u2)
                        # Find three nodes v1, v2, v3 not in current such that:
                        # 1. Each is compatible with (current - {u1, u2})
                        # 2. They are pairwise compatible
                        candidates = []
                        for v in range(m):
                            if v in current:
                                continue
                            if (adj_mask[v] & mask_without_u1u2) == 0:
                                candidates.append(v)
                        
                        # Try all triples from candidates
                        for idx1 in range(len(candidates)):
                            v1 = candidates[idx1]
                            for idx2 in range(idx1 + 1, len(candidates)):
                                v2 = candidates[idx2]
                                if (adj_mask[v1] >> v2) & 1:
                                    continue
                                for idx3 in range(idx2 + 1, len(candidates)):
                                    v3 = candidates[idx3]
                                    if (adj_mask[v1] >> v3) & 1 or (adj_mask[v2] >> v3) & 1:
                                        continue
                                    
                                    # Remove u1, u2; add v1, v2, v3
                                    current.remove(u1)
                                    current.remove(u2)
                                    current.add(v1)
                                    current.add(v2)
                                    current.add(v3)
                                    current_mask ^= (1 << u1)
                                    current_mask ^= (1 << u2)
                                    current_mask |= (1 << v1)
                                    current_mask |= (1 << v2)
                                    current_mask |= (1 << v3)
                                    improved = True
                                    found_2for3 = True
                                    break
                                if found_2for3:
                                    break
                            if found_2for3:
                                break
                        if found_2for3:
                            break
                    if found_2for3:
                        break

        return list(current)

    # --- Branch and Bound for Maximum Independent Set ---
    # Tomita's algorithm for MIS using bitmasks.
    # We want to find a set S such that no two elements in S are adjacent in G.
    # This is equivalent to MIS in G.
    # Note: Our adj_mask represents the conflict graph.
    
    def max_independent_set_bitmask(adj, n, deadline):
        # adj is a list of integers (bitmasks)
        # n is the number of vertices
        # Returns the size of the MIS and the set itself (if possible)
        
        # We use a recursive approach with coloring bound (Tomita).
        # However, implementing full Tomita with backtracking in Python might be slow for n=205 if the graph is dense.
        # But 205 is small enough that a well-optimized B&B might work, especially with a good lower bound.
        
        # Let's try a simpler B&B first:
        # Order vertices by degree? Or just use natural order?
        # We'll use a greedy coloring on the candidate set to get an upper bound.
        
        # To keep it simple and fast, we will implement a depth-first search with pruning.
        # State: (current_set_size, current_set_mask, candidate_mask)
        # candidate_mask contains vertices that are compatible with current_set and have index > last_added?
        # Actually, to avoid duplicates, we usually process vertices in a fixed order.
        
        # Let's use the standard "Max Clique on Complement" approach which is MIS on G.
        # Complement graph H: edge (u,v) in H if no edge in G.
        # We want Max Clique in H.
        # Adjacency in H: h_adj[u] = ~adj[u] & ((1 << n) - 1) & ~(1 << u)
        
        h_adj = [0] * n
        full_mask = (1 << n) - 1
        for u in range(n):
            h_adj[u] = full_mask ^ adj[u] ^ (1 << u)
        
        best_size_global = 0
        best_set_global = []
        
        # Recursive function
        # P is the set of candidates (bitmask)
        # R is the set of vertices in the current clique (bitmask)
        # r_size is the number of vertices in R
        
        # To speed up, we can use a greedy coloring bound on P.
        # Coloring of P:
        # We assign colors to vertices in P such that adjacent vertices in H have different colors.
        # The maximum color index is an upper bound on the size of any clique in P.
        
        def color_bound(P):
            # Greedy coloring of P
            # Returns list of (vertex, color) and max_color
            uncolored = P
            colors = {}
            color_idx = 0
            max_c = 0
            # We need to iterate vertices in P
            # To make it deterministic and fast, we can just pick the lowest set bit each time?
            # Or iterate 0..n-1?
            # Let's iterate through bits.
            
            # A simple greedy coloring:
            # While P is not empty:
            #   Pick a vertex v from P (e.g., lowest index)
            #   Assign color c
            #   Remove v from P
            #   Remove all neighbors of v in H from P (they can't share color c)
            #   Increment c? No, standard greedy coloring uses smallest available color.
            #   But for bound, we just need a proper coloring. The number of colors used is the bound.
            
            # Let's use a simpler bound: The size of a maximum clique in P is at most the size of P.
            # That's too weak.
            # Let's use the greedy coloring where we partition P into independent sets of G (cliques in H).
            # Wait, in H, we want a clique. So in G, we want an independent set.
            # A clique in H is an independent set in G.
            # So we can color G. The chromatic number of G is an upper bound on the size of an independent set in G.
            # So we can greedily color G[P] (induced subgraph).
            
            # Greedy coloring of G[P]:
            # 1. Order vertices in P (e.g., by degree in G[P] descending, or just index).
            # 2. Assign smallest available color to each vertex.
            
            # Let's just count the number of independent sets we can pack? No.
            # Let's use the standard "Color Sort" from Tomita.
            # It sorts vertices by color class.
            
            # For simplicity and speed in Python, let's use a weaker bound:
            # Upper bound = number of vertices in P.
            # This is O(1) but weak.
            
            # Let's try a slightly better bound:
            # Greedily find independent sets in G[P].
            # Each independent set in G is a clique in H.
            # If we can pack k independent sets, it doesn't bound the max independent set.
            # Wait, the chromatic number chi(G) >= alpha(G).
            # So if we find a coloring of G[P] with C colors, then alpha(G[P]) <= C.
            
            # Let's implement a simple greedy coloring of G[P].
            # We process vertices in P in order of decreasing degree in G[P].
            
            verts = []
            temp = P
            while temp:
                # Get lowest set bit
                v = (temp & -temp).bit_length() - 1
                verts.append(v)
                temp &= temp - 1
            
            # Sort by degree in G[P] (descending)
            # Degree of v in G[P] is popcount(adj[v] & P)
            # Note: adj[v] is the conflict mask.
            verts.sort(key=lambda v: (adj[v] & P).bit_count(), reverse=True)
            
            color = [0] * n
            max_color = 0
            for v in verts:
                # Find smallest color not used by neighbors in G
                used_colors = 0
                # Neighbors of v in G[P]
                neigh = adj[v] & P
                # Iterate neighbors
                # To check used colors, we need to know the colors of neighbors.
                # This is O(N) per vertex, so O(N^2) total. N=205, N^2=42k. Acceptable.
                
                # Actually, we can just check if any neighbor has color c.
                # Let's maintain a list of colors per color index? No, just check.
                
                # Optimization: We can just try color 1, 2, ...
                # If a neighbor has that color, skip.
                
                # For speed, let's just assign the next available color.
                # We can use a boolean array for used colors.
                
                used = [False] * (len(verts) + 1)
                # Check neighbors
                temp_neigh = neigh
                while temp_neigh:
                    u = (temp_neigh & -temp_neigh).bit_length() - 1
                    if color[u] > 0:
                        used[color[u]] = True
                    temp_neigh &= temp_neigh - 1
                
                c = 1
                while used[c]:
                    c += 1
                color[v] = c
                if c > max_color:
                    max_color = c
            
            return max_color
        
        # Recursive B&B
        # To avoid deep recursion limits, we might need to increase recursion limit or use iterative.
        # sys.setrecursionlimit(3000) is usually safe.
        
        def branch_and_bound(P, R, r_size):
            nonlocal best_size_global, best_set_global
            
            if time.time() > deadline:
                return
            
            if P == 0:
                if r_size > best_size_global:
                    best_size_global = r_size
                    # Reconstruct set from R
                    s = []
                    temp = R
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        s.append(v)
                        temp &= temp - 1
                    best_set_global = s
                return
            
            # Upper bound: r_size + color_bound(P)
            # If this is <= best_size_global, prune.
            bound = color_bound(P)
            if r_size + bound <= best_size_global:
                return
            
            # Choose a vertex from P
            # Heuristic: Pick vertex with highest degree in G[P]? Or lowest?
            # For MIS, picking a vertex with high degree in G (high conflict) might be good?
            # Or low degree?
            # Let's pick the vertex with the highest degree in the conflict graph G induced by P.
            # This is similar to Max Clique on H where we pick high degree in H.
            # In H, degree of v is (h_adj[v] & P).bit_count().
            # In G, degree of v is (adj[v] & P).bit_count().
            # Let's pick max degree in G.
            
            # Find v in P with max degree in G[P]
            max_deg = -1
            v = -1
            temp = P
            while temp:
                u = (temp & -temp).bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
                temp &= temp - 1
            
            if v == -1:
                return
            
            # Branch 1: Include v
            # New P = P & ~adj[v] & ~(1 << v)  (remove v and its neighbors in G)
            # New R = R | (1 << v)
            branch_and_bound(P & ~(adj[v] | (1 << v)), R | (1 << v), r_size + 1)
            
            # Branch 2: Exclude v
            # New P = P & ~(1 << v)
            branch_and_bound(P & ~(1 << v), R, r_size)
        
        # Initial call
        # P = all vertices
        # R = 0
        branch_and_bound((1 << n) - 1, 0, 0)
        
        return best_size_global, best_set_global

    deadline = time.time() + 580.0
    
    # --- MILP Approach ---
    try:
        n_vars = m
        c = -np.ones(n_vars)
        
        n_constraints = len(edges)
        if n_constraints > 0:
            A = np.zeros((n_constraints, n_vars))
            for idx, (i, j) in enumerate(edges):
                A[idx, i] = 1
                A[idx, j] = 1
            
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n_vars)
            cons = LinearConstraint(A, ub=np.ones(n_constraints))
            
            start_time = time.time()
            # Give MILP 60 seconds
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 60})
            
            if res.status == 0 or res.status == 1: # Optimal or Feasible
                solution = res.x.astype(int)
                subset = [i for i in range(n_vars) if solution[i] == 1]
                update_best(subset)
            else:
                # Try to use the incumbent if available?
                # milp doesn't easily expose incumbent in all versions/statuses simply.
                # We'll rely on fallback.
                pass
    except Exception:
        pass

    # --- Local Search Refinement ---
    # If we have a solution from MILP, refine it.
    if best_size > 0:
        refined = local_search(best_set, max_iter=1000)
        update_best(refined)
    
    # --- Randomized Greedy + Local Search Fallback ---
    # If MILP failed or we want to explore more, run random greedy starts.
    # This is a different design family: Stochastic Local Search (Simulated Annealing-like or simple Hill Climbing)
    # from multiple random starts.
    
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random greedy construction
            order = list(range(m))
            random.shuffle(order)
            current = []
            current_mask = 0
            for candidate in order:
                if (adj_mask[candidate] & current_mask) == 0:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
            
            # Local search from this random start
            refined = local_search(current, max_iter=2000)
            update_best(refined)

    # --- Branch and Bound Refinement ---
    # Use B&B to try to beat the current best_size.
    # We pass deadline to B&B so it stops if time is up.
    # Note: B&B in Python might be slow. We give it a small time slice first?
    # Or just run it if time remains.
    
    if time.time() < deadline:
        # Set a tighter deadline for B&B to ensure we return something
        bnb_deadline = min(deadline, time.time() + 50.0)
        try:
            bnb_size, bnb_set = max_independent_set_bitmask(adj_mask, m, bnb_deadline)
            if bnb_set:
                update_best(bnb_set)
        except Exception:
            pass

    if best_set:
        return np.array(best_set, dtype=np.int64)
    
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
