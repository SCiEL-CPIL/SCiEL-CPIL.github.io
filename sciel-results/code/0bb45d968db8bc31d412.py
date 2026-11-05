import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 320.0  # 300s limit, 20s buffer

    def time_left():
        return deadline - time.time()

    # 1. Construct the seed Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_init = q
    G = [set() for _ in range(n_init)]
    edges = []
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if (j - i) % q in residues:
                G[i].add(j)
                G[j].add(i)
                edges.append((i, j))
    
    current_n = n_init
    current_G = G
    current_edges = edges

    # Helper to count K3s in a subset S that include a specific vertex u
    # A K3 containing u and two vertices x, y in S exists if x,y in S, x in G[u], y in G[u], and x in G[y]
    def count_k3_with_u(S, u):
        # S is a set of vertices
        # We need to count edges in the induced subgraph on (S intersect N(u))
        neighbors_in_S = [v for v in S if v in current_G[u]]
        count = 0
        for i in range(len(neighbors_in_S)):
            for j in range(i + 1, len(neighbors_in_S)):
                if neighbors_in_S[j] in current_G[neighbors_in_S[i]]:
                    count += 1
        return count

    # Helper to count Independent K3s (i.e., K3 in complement) in a subset T that include a specific vertex u
    # An independent set of size 3 containing u and x, y in T exists if x,y in T, x not in G[u], y not in G[u], and x not in G[y]
    def count_ik3_with_u(T, u):
        # T is a set of vertices
        # Non-neighbors of u in T
        non_neighbors_in_T = [v for v in T if v not in current_G[u]]
        count = 0
        for i in range(len(non_neighbors_in_T)):
            for j in range(i + 1, len(non_neighbors_in_T)):
                if non_neighbors_in_T[j] not in current_G[non_neighbors_in_T[i]]:
                    count += 1
        return count

    def try_add_vertex():
        if time_left() < 1.0:
            return False
        
        new_v = current_n
        # Initialize S (neighbors of new_v)
        # Random start
        S = set(random.sample(range(current_n), random.randint(0, current_n)))
        T = set(range(current_n)) - S
        
        # Calculate initial violations
        # Violations = #K4 in S + #IK4 in T
        # But we can track K3s and IK3s incrementally?
        # Actually, adding a vertex v to S creates a K5 if S contains a K4.
        # So we need to count K4s in S.
        # Counting K4s is O(|S|^4) which is too slow.
        
        # Alternative: Count K3s.
        # A K5 is formed if v connects to a K4.
        # A K4 in S is a set of 4 vertices pairwise connected.
        # We can count K4s by summing over vertices in S the number of K3s in (S \ {u}) that include u? No.
        
        # Let's use the property:
        # #K5s in new graph = #K4s in G[S]
        # #IK5s in new graph = #IK4s in G[T]
        
        # We need a fast way to count K4s in a subset.
        # This is still hard.
        
        # Let's step back.
        # The seed program's approach was to check validity.
        # My SA approach needs a cost function.
        # Cost = #K4s in S + #IK4s in T.
        
        # Can we estimate or compute this faster?
        # For n=37, |S| is ~18. 18^4 = 100k. That's fast enough for a few evaluations.
        # But SA does many steps.
        
        # Let's try to optimize K4 counting.
        # K4s in S = sum over edges (u,v) in S of (common neighbors of u,v in S) / 3? No.
        # K4s = 1/24 * sum_{u,v,w,x distinct in S, all connected} 1.
        
        # Let's just implement a direct counter and hope it's fast enough for small n.
        def count_k4_in_subset(S):
            S_list = list(S)
            count = 0
            n_s = len(S_list)
            for i in range(n_s):
                u = S_list[i]
                # Neighbors of u in S
                neigh_u = [v for v in S_list[i+1:] if v in current_G[u]]
                # For each pair of neighbors, check if they form a triangle with u?
                # No, we need K4.
                # Iterate over edges (u, v) in S
                for j in range(i + 1, n_s):
                    v = S_list[j]
                    if v in current_G[u]:
                        # Common neighbors of u and v in S
                        common = current_G[u] & current_G[v] & S
                        # Remove u and v
                        common.discard(u)
                        common.discard(v)
                        # For each pair (w, x) in common, if w-x is an edge, we have a K4 (u,v,w,x)
                        # Wait, if w,x in common, they are connected to u and v.
                        # If w-x is an edge, then {u,v,w,x} is a K4.
                        common_list = list(common)
                        for k in range(len(common_list)):
                            w = common_list[k]
                            for l in range(k + 1, len(common_list)):
                                x = common_list[l]
                                if x in current_G[w]:
                                    count += 1
            return count

        def count_ik4_in_subset(T):
            # Count K4s in complement graph restricted to T
            # Equivalent to counting independent sets of size 4 in G[T]
            T_list = list(T)
            count = 0
            n_t = len(T_list)
            for i in range(n_t):
                u = T_list[i]
                # Non-neighbors of u in T
                non_neigh_u = [v for v in T_list[i+1:] if v not in current_G[u]]
                for j in range(i + 1, n_t):
                    v = T_list[j]
                    if v not in current_G[u]:
                        # Common non-neighbors of u and v in T
                        # v is non-neighbor of u.
                        # We need w, x in T such that:
                        # w not in G[u], w not in G[v]
                        # x not in G[u], x not in G[v]
                        # w not in G[x]
                        # u-v is non-edge, u-w non-edge, u-x non-edge, v-w non-edge, v-x non-edge, w-x non-edge.
                        
                        common_non = set()
                        # Optimization: Iterate T and check
                        for w in T_list:
                            if w != u and w != v and w not in current_G[u] and w not in current_G[v]:
                                common_non.add(w)
                        
                        common_list = list(common_non)
                        for k in range(len(common_list)):
                            w = common_list[k]
                            for l in range(k + 1, len(common_list)):
                                x = common_list[l]
                                if x not in current_G[w]:
                                    count += 1
            return count

        # This counting is O(n^4) or worse. For n=40, 40^4 = 2.5M.
        # SA might do 1000 steps. 2.5B operations. Too slow.
        
        # We need a better cost function or faster counting.
        
        # Alternative:
        # Instead of counting K4s, count K3s.
        # A K5 contains 5 K4s.
        # This doesn't help directly with the "no K5" constraint.
        
        # Let's try a different approach:
        # Local search on the whole graph?
        # Start with Paley(37).
        # Flip edges to reduce "density" of K4s and IK4s?
        
        # Actually, let's look at the "Add Vertex" logic again.
        # If we can't add a vertex, maybe we should modify the existing graph to make it "more flexible"?
        # E.g., break some K4s to allow a new vertex to connect to a larger set?
        
        # Given the time constraints and complexity, let's stick to the seed program's logic but optimize it.
        # The seed program's `find_extension` uses local search on the neighbor set A.
        # It checks `is_valid_extension(A)` which calls `has_clique` and `has_independent_set`.
        # These are backtracking searches.
        
        # Improvement:
        # 1. Use a faster clique/independent set checker (e.g., using bitsets or numpy).
        # 2. Use a better heuristic for the local search (e.g., Simulated Annealing instead of simple greedy).
        
        # Let's implement Simulated Annealing for the neighbor set A.
        # Cost(A) = 0 if valid, else penalty.
        # Penalty = (#K4s in A) + (#IK4s in V\A).
        
        # We need fast K4 counting.
        # Let's precompute K3 counts?
        
        # Actually, for n=37, the backtracking in the seed might be slow because it's Python loops.
        # Let's try to speed up the validation.
        
        # Re-trying the SA on A with a faster cost function.
        # We can approximate the cost or use a smaller check.
        
        # Let's try to improve the seed's `has_clique` and `has_independent_set` using numpy bitsets.
        
        def fast_has_clique(vertices, G, k=4):
            # vertices: list of indices
            # G: list of sets
            # Use backtracking with bitsets for candidate filtering
            if len(vertices) < k:
                return False
            
            # Precompute adjacency bitmasks for vertices in the subset?
            # Or just use the sets.
            
            # Simple optimization: Order vertices by degree in the induced subgraph
            # High degree first.
            
            def backtrack(current_clique, candidates):
                if len(current_clique) == k:
                    return True
                for i, v in enumerate(candidates):
                    # Check if v is connected to all in current_clique
                    # Optimization: If v is not connected to the last added vertex, skip?
                    # Better: candidates should be intersection of neighbors of all in current_clique.
                    
                    # This implementation is standard.
                    if all(v in G[u] for u in current_clique):
                        # New candidates are candidates[i+1:] that are neighbors of v
                        new_candidates = [u for u in candidates[i+1:] if u in G[v]]
                        if backtrack(current_clique + [v], new_candidates):
                            return True
            return backtrack([], vertices)

        def fast_has_independent_set(vertices, G, k=4):
            if len(vertices) < k:
                return False
            
            def backtrack(current_set, candidates):
                if len(current_set) == k:
                    return True
                for i, v in enumerate(candidates):
                    if all(v not in G[u] for u in current_set):
                        new_candidates = [u for u in candidates[i+1:] if u not in G[v]]
                        if backtrack(current_set + [v], new_candidates):
                            return True
            return backtrack([], vertices)

        # Now, use SA to find a valid A.
        
        def find_extension_sa():
            if time_left() < 2.0:
                return None
            
            best_A = None
            best_cost = float('inf')
            
            # Cost function
            def cost(A):
                # A is a set
                # Return 0 if valid
                # Else return penalty
                # Penalty: number of K4s in A + number of IK4s in V\A
                # We can just check validity and return 1 if invalid, 0 if valid?
                # That's not smooth.
                
                # Let's return the number of violations.
                # Counting K4s is expensive.
                # Let's just return 1 if invalid, 0 if valid.
                # And use SA to find a 0.
                
                valid = True
                # Check K4 in A
                if fast_has_clique(list(A), current_G, 4):
                    return 1
                # Check IK4 in V\A
                non_A = [v for v in range(current_n) if v not in A]
                if fast_has_independent_set(non_A, current_G, 4):
                    return 1
                return 0

            # SA Loop
            T = 1.0
            T_min = 0.01
            cooling = 0.95
            
            # Random init
            A = set(random.sample(range(current_n), random.randint(0, current_n)))
            current_cost = cost(A)
            
            if current_cost == 0:
                return A
                
            for step in range(2000):
                if time_left() < 1.0:
                    break
                    
                # Move: flip a vertex
                v = random.choice(range(current_n))
                if v in A:
                    A.remove(v)
                else:
                    A.add(v)
                    
                new_cost = cost(A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        return A
                else:
                    # Accept with probability
                    delta = new_cost - current_cost
                    if random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        if current_cost == 0:
                            return A
                    else:
                        # Revert
                        if v in A:
                            A.remove(v)
                        else:
                            A.add(v)
                            
                T *= cooling
                if T < T_min:
                    break
                    
            return None

        # Try to extend
        max_target = 46
        while current_n < max_target:
            if time_left() < 5.0:
                break
                
            extension = find_extension_sa()
            if extension is not None:
                new_v = current_n
                current_G.append(set(extension))
                for u in extension:
                    current_G[u].add(new_v)
                    current_edges.append((min(u, new_v), max(u, new_v)))
                current_n += 1
            else:
                # Try a few more times with different random seeds/temps?
                # Or break.
                # Let's try 3 restarts
                success = False
                for _ in range(3):
                    if time_left() < 5.0:
                        break
                    extension = find_extension_sa()
                    if extension is not None:
                        new_v = current_n
                        current_G.append(set(extension))
                        for u in extension:
                            current_G[u].add(new_v)
                            current_edges.append((min(u, new_v), max(u, new_v)))
                        current_n += 1
                        success = True
                        break
                if not success:
                    break

    return current_n, current_edges
