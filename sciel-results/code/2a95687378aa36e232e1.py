"""Clique partition of split graphs using global edge-coloring optimization."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # 1. Build graph structures
        adj_c = [set() for _ in range(k)]  # Neighbors in I for each c in C
        adj_i = [set() for _ in range(n)]  # Neighbors in C for each i in I
        c_edges = []
        cross_edges = []
        
        for (u, v) in edges:
            if u < k and v < k:
                c_edges.append((u, v))
            else:
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.append((c, i))
                adj_c[c].add(i)
                adj_i[i].add(c)

        active_i = [i for i in range(k, n) if adj_i[i]]
        
        # If no cross edges, just cover C edges
        if not cross_edges:
            return cover_c_edges(c_edges, k)

        # 2. Initialize Coloring
        # We assign colors to cross_edges.
        # Let's use a greedy initialization: 
        # For each i, assign colors to its neighbors such that no two neighbors 
        # of i have the same color if they are NOT connected in C? 
        # Wait, C is a clique, so all neighbors of i in C are connected.
        # The constraint for a clique {i} U S is that S is a subset of N(i).
        # Since C is a clique, ANY subset of N(i) is a valid S.
        # So there is NO constraint on which neighbors share a color, EXCEPT 
        # we want to minimize the number of colors per i (to reduce clique count)
        # AND we want to cover C-edges.
        
        # Actually, if we assign all edges incident to i the SAME color, 
        # we get 1 clique {i} U N(i). This covers all C-edges within N(i).
        # This is the best case for i.
        # The problem arises if we do this for ALL i.
        # Then C-edges are covered multiple times? 
        # Rule: "every edge must lie in exactly one listed clique".
        # If we form {i} U N(i) for all i, a C-edge (u,v) is in {i}U N(i) 
        # if u,v in N(i). If this happens for i1 and i2, the edge is in two cliques. INVALID.
        
        # So we must ensure that for any C-edge (u,v), there is at most one i 
        # such that u,v in S_{i,c} for some color c.
        # Ideally, we want to assign colors such that the sets S_{i,c} 
        # form a partition of the C-edges (plus maybe some vertices not covering edges).
        
        # Let's refine the cost function and search.
        # State: color[e] for e in cross_edges.
        # We can represent state as a list of colors for each cross_edge.
        
        # Initialization: 
        # Assign random colors 1..k_max to each edge?
        # Better: Start with a valid solution.
        # A simple valid solution: 
        # For each cross edge (u,i), assign a unique color based on the edge index.
        # This creates cliques of size 2: {u,i}.
        # This covers all cross edges.
        # C-edges are NOT covered.
        # So we need to cover C-edges separately.
        # Total cliques = |E_cross| + MinCliqueCover(C_edges).
        
        # Can we do better?
        # If we merge edges (u1, i) and (u2, i) into color c, 
        # we replace 2 cliques {u1,i}, {u2,i} with 1 clique {u1, u2, i}.
        # This saves 1 clique.
        # BUT it covers C-edge (u1, u2).
        # If (u1, u2) was previously covered by a pure C-clique, we might save more?
        # Or if it was uncovered, we just cover it.
        # The risk is if (u1, u2) is ALSO covered by another clique {u1, u2, j}.
        
        # Let's use a greedy construction that is smarter than unique colors.
        # We want to maximize the number of C-edges covered by mixed cliques 
        # without double covering.
        
        # Algorithm:
        # 1. Start with empty color assignment.
        # 2. We will build cliques one by one.
        # 3. However, the directive asks for a "minimum clique cover of a bipartite graph... vertex cover in auxiliary graph".
        # This suggests a different modeling.
        
        # Let's try the Simulated Annealing on the "Merging" decisions.
        # Actually, let's stick to the Edge Coloring formulation but optimize the assignment.
        
        # State: For each i in I, we have a partition of N(i) into sets S_{i,1}, S_{i,2}, ...
        # Each set corresponds to a color.
        # Cost = Sum_i (num_sets_i) + MinCliqueCover( Uncovered_C_Edges )
        
        # Initialization:
        # For each i, N(i) is one set. (All edges incident to i have color 1).
        # This is valid? 
        # If N(i) is one set, clique is {i} U N(i).
        # If we do this for all i, C-edges are covered multiple times.
        # We need to fix this.
        
        # Repair Strategy:
        # 1. Start with all N(i) as single sets.
        # 2. Identify conflicts: C-edges covered by >1 clique.
        # 3. Resolve conflicts by splitting sets.
        
        # Let's implement a "Split and Repair" local search.
        
        # Data Structures:
        # colors[i] = dict mapping color_id -> set of C-vertices
        # Initially, for each i, colors[i] = {1: N(i)}
        
        # We need to track which C-edges are covered by which clique.
        # covered_by[edge] = set of clique_ids
        
        # Let's assign unique IDs to cliques.
        # cliques = []
        # clique_id_map = {} # (i, color_id) -> clique_id
        
        # This is getting complex to track dynamically.
        
        # Alternative: 
        # Since k is small (10, 20, 30), we can use a bitset for C-vertices.
        # But let's try a simpler heuristic first: 
        # Greedy Assignment with Backtracking? No, too slow.
        
        # Let's try the "Vertex Cover" inspiration.
        # Transform: 
        # We want to select a set of cliques.
        # Each mixed clique is defined by (i, S) where S subset N(i).
        # This clique covers cross edges {(i, u) : u in S} and C-edges {(u,v) : u,v in S}.
        
        # Let's use a Greedy Set Cover with a twist:
        # Prioritize covering C-edges.
        # 1. List all C-edges.
        # 2. For each C-edge (u,v), find all i such that u,v in N(i).
        #    These i's can form a clique {i, u, v} that covers this edge.
        # 3. We want to select cliques to cover all C-edges and all Cross-edges.
        #    Note: Cross-edges MUST be covered by mixed cliques.
        #    So for each i, we MUST select at least one clique involving i?
        #    No, if i has no cross edges, we don't need cliques for i.
        #    If i has cross edges, every cross edge (i,u) must be in some clique {i} U S.
        #    So the union of S's for a fixed i must be N(i).
        
        # So the problem is:
        # For each i, partition N(i) into sets S_{i,1}, ..., S_{i,k_i}.
        # Minimize Sum k_i + MinCliqueCover( C-edges not covered by any {i} U S_{i,j} ).
        
        # Let's implement a Local Search on the partitions.
        # Representation:
        # For each i, a list of sets.
        
        # Init:
        # For each i, start with N(i) as one set.
        # Calculate covered C-edges.
        # If any C-edge is covered >1 times, it's invalid.
        # We need to split sets to resolve conflicts.
        
        # Cost Function:
        # If valid (no double coverage):
        #   Cost = Sum k_i + MinCliqueCover(remaining C-edges)
        # If invalid:
        #   Cost = Large Number + (number of conflicts)
        
        # Moves:
        # 1. Split a set S_{i,j} into two parts A and B.
        #    This increases k_i by 1.
        #    It might resolve conflicts (if the conflict was caused by S_{i,j} sharing edges with other cliques).
        # 2. Merge two sets S_{i,j} and S_{i,l} into one.
        #    This decreases k_i by 1.
        #    It might create conflicts.
        
        # Let's implement this.
        
        def min_clique_cover_c(edges, k):
            # Greedy max clique cover for C-edges
            if not edges: return 0
            uncovered = set(edges)
            count = 0
            while uncovered:
                # Pick a vertex with max degree in uncovered
                deg = {}
                for u, v in uncovered:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                if not deg: break
                u_max = max(deg, key=deg.get)
                # Build clique
                S = {u_max}
                candidates = [v for (u,v) in uncovered if u==u_max or v==u_max]
                for v in candidates:
                    ok = True
                    for x in S:
                        if x == v: continue
                        e = (min(x,v), max(x,v))
                        if e not in uncovered:
                            ok = False
                            break
                    if ok:
                        S.add(v)
                        for x in S:
                            if x == v: continue
                            e = (min(x,v), max(x,v))
                            uncovered.discard(e)
                if len(S) > 1:
                    count += 1
                else:
                    # Should not happen if u_max has degree > 0
                    # Just remove u_max? No, must cover edges.
                    # If S is size 1, we didn't cover any edge.
                    # This implies u_max had no neighbors in uncovered? 
                    # But we picked it based on degree.
                    # So this shouldn't happen.
                    pass
            return count

        # State: partitions[i] = list of sets (frozensets for hashing?)
        # Let's use lists of sets.
        
        partitions = {}
        for i in active_i:
            partitions[i] = [set(adj_i[i])]
        
        # Helper to calculate coverage and cost
        def evaluate(partitions):
            # Returns (is_valid, cost)
            # is_valid: True if no C-edge covered >1 time
            # cost: Sum k_i + MinCliqueCover(uncovered C-edges)
            
            covered_count = {} # edge -> count
            # Also track which cliques cover which edges for potential debugging? No, just count.
            
            total_k = 0
            for i in active_i:
                for S in partitions[i]:
                    total_k += 1
                    # Cover C-edges within S
                    for u, v in combinations(S, 2):
                        e = (min(u,v), max(u,v))
                        covered_count[e] = covered_count.get(e, 0) + 1
            
            # Check validity
            is_valid = True
            for e, cnt in covered_count.items():
                if cnt > 1:
                    is_valid = False
                    break
            
            if not is_valid:
                # Count conflicts for penalty
                conflicts = sum(1 for cnt in covered_count.values() if cnt > 1)
                return False, 10000 + conflicts + total_k
            
            # Calculate uncovered C-edges
            uncovered_c = set()
            for e in c_edges:
                if covered_count.get(e, 0) == 0:
                    uncovered_c.add(e)
            
            c_cost = min_clique_cover_c(list(uncovered_c), k)
            return True, total_k + c_cost

        # Local Search
        current_partitions = {i: [set(s) for s in partitions[i]] for i in partitions}
        current_valid, current_cost = evaluate(current_partitions)
        
        best_partitions = {i: [set(s) for s in current_partitions[i]] for i in current_partitions}
        best_valid, best_cost = current_valid, current_cost
        
        t = 1000.0
        t_min = 1.0
        cooling = 0.99
        
        steps = 0
        while time.time() < deadline - 10 and t > t_min:
            steps += 1
            if steps % 100 == 0 and time.time() > deadline - 20:
                break
            
            # Pick a random move
            # 50% Split, 50% Merge
            if random.random() < 0.5:
                # Split
                # Pick random i
                if not active_i: continue
                i = random.choice(active_i)
                if not partitions[i]: continue
                # Pick random set
                idx = random.randint(0, len(partitions[i]) - 1)
                S = partitions[i][idx]
                if len(S) < 2: continue
                # Split S into two non-empty subsets
                # Random split
                S_list = list(S)
                random.shuffle(S_list)
                split_point = random.randint(1, len(S_list) - 1)
                A = set(S_list[:split_point])
                B = set(S_list[split_point:])
                
                new_partitions = {j: [set(s) for s in partitions[j]] for j in partitions}
                new_partitions[i] = [set(s) for s in partitions[i]]
                new_partitions[i][idx] = A
                new_partitions[i].append(B)
                
            else:
                # Merge
                # Pick random i
                if not active_i: continue
                i = random.choice(active_i)
                if len(partitions[i]) < 2: continue
                # Pick two random distinct sets
                idx1, idx2 = random.sample(range(len(partitions[i])), 2)
                S1 = partitions[i][idx1]
                S2 = partitions[i][idx2]
                
                new_partitions = {j: [set(s) for s in partitions[j]] for j in partitions}
                new_partitions[i] = [set(s) for s in partitions[i]]
                # Merge S1 and S2
                merged = S1.union(S2)
                # Remove S2 and replace S1 with merged
                # Be careful with indices
                new_partitions[i].pop(idx2)
                new_idx1 = idx1 if idx1 < idx2 else idx1 - 1
                new_partitions[i][new_idx1] = merged
            
            new_valid, new_cost = evaluate(new_partitions)
            
            # Acceptance
            if new_cost < current_cost:
                current_partitions = new_partitions
                current_valid, current_cost = new_valid, new_cost
                if new_cost < best_cost:
                    best_partitions = {i: [set(s) for s in new_partitions[i]] for i in new_partitions}
                    best_valid, best_cost = new_valid, new_cost
            else:
                delta = new_cost - current_cost
                if random.random() < pow(2.718, -delta / t):
                    current_partitions = new_partitions
                    current_valid, current_cost = new_valid, new_cost
            
            t *= cooling
            
        # Use best found
        final_partitions = best_partitions
        
        # Construct cliques from final_partitions
        cliques = []
        covered_c_edges = set()
        
        for i in active_i:
            for S in final_partitions[i]:
                if len(S) == 0: continue
                clique = [i] + sorted(list(S))
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Mark C-edges as covered
                    for u, v in combinations(S, 2):
                        e = (min(u,v), max(u,v))
                        covered_c_edges.add(e)
        
        # Cover remaining C-edges
        remaining_c = [e for e in c_edges if e not in covered_c_edges]
        if remaining_c:
            # Use greedy max clique cover
            uncovered = set(remaining_c)
            while uncovered:
                deg = {}
                for u, v in uncovered:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                if not deg: break
                u_max = max(deg, key=deg.get)
                S = {u_max}
                candidates = [v for (u,v) in uncovered if u==u_max or v==u_max]
                for v in candidates:
                    ok = True
                    for x in S:
                        if x == v: continue
                        e = (min(x,v), max(x,v))
                        if e not in uncovered:
                            ok = False
                            break
                    if ok:
                        S.add(v)
                        for x in S:
                            if x == v: continue
                            e = (min(x,v), max(x,v))
                            uncovered.discard(e)
                if len(S) >= 2:
                    cliques.append(sorted(list(S)))
                else:
                    # Should not happen
                    pass
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        final_cliques = []
        for c in cliques:
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
