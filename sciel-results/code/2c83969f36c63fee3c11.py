import time
import random
import math
from functools import lru_cache

def construct():
    n = 20
    deadline = time.time() + 130  # Leave margin for 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)  # 190
    
    # Build conflict graph: two edges conflict if they cannot both be in a K2,3-free graph
    # Edge e1=(a,b) and e2=(c,d) conflict if adding both would create a K2,3
    # This happens when:
    #   - They share a vertex, say a=c, and |N(b)∩N(d)| >= 2 (then b,d have >=2 common neighbors including a, plus potentially others)
    #   Wait, let me think more carefully.
    # 
    # A K2,3 is formed by two vertices x,y with at least 3 common neighbors.
    # If we have edges (x,a), (x,b), (x,c), (y,a), (y,b), (y,c) that's a K2,3.
    # 
    # For the conflict graph approach: vertex i in conflict graph = edge e_i in original graph.
    # Two conflict-graph vertices i,j are connected if edges e_i and e_j cannot coexist in any K2,3-free graph.
    # 
    # When can two edges not coexist? 
    # Edge (u,v) and edge (w,x): they conflict if there exists a K2,3 that must contain both.
    # Actually, it's simpler: they conflict if adding both to ANY valid graph would create a K2,3.
    # But that's not right either - it depends on what other edges are present.
    #
    # Let me reconsider. The correct formulation:
    # Two edges e1=(a,b) and e2=(c,d) are incompatible if their presence forces a K2,3.
    # This happens when: a,c are two vertices and b,d are common neighbors, AND there's a third common neighbor.
    # More precisely: if a=c (share vertex), then b and d need at most 2 common neighbors besides a.
    # If a!=c, b!=d, a!=d, b!=c (disjoint edges): no direct conflict.
    # If a=d or b=c (share one vertex in "cross" pattern): similar to sharing a vertex.
    #
    # Actually, the cleanest way: precompute for each pair of edges whether they can coexist
    # in some K2,3-free graph. Two edges conflict if NO K2,3-free graph contains both.
    #
    # For two edges (u,v) and (w,x):
    # Case 1: u=w (share first endpoint). Then v and x are both neighbors of u.
    #   They conflict if |N(v) ∩ N(x) \ {u}| >= 2, i.e., v and x have at least 2 other common neighbors.
    #   But we don't know N yet... This is circular.
    #
    # Better approach: The conflict is structural. Two edges (u,v) and (w,x) conflict if:
    #   - They form part of a forced K2,3. Specifically:
    #   - If u=w: v and x share u as a common neighbor. If v and x also have 2+ other forced common neighbors... 
    #   This is getting complicated. Let me use a different characterization.
    #
    # KEY INSIGHT: Two edges (a,b) and (c,d) are incompatible in a K2,3-free graph if and only if:
    #   There exist vertices p, q such that {a,b} ∩ {c,d} contains p, and the other two vertices 
    #   (say b and d if a=c=p) have at least 2 common neighbors in {a,b,c,d} \ {p} ∪ {forced neighbors}.
    #
    # Actually, the simplest correct check: two edges (a,b) and (c,d) conflict if:
    #   - a = c: then b and d are both adjacent to a. A K2,3 with center pair (a, ?) requires 
    #     b and d to share 3 neighbors. They already share a. So they need 2 more common neighbors.
    #     But we can't force those. So they DON'T necessarily conflict.
    #   - a = d (or b = c): similar.
    #   - a = b or c = d: impossible since edges are simple.
    #   - All four distinct: no direct conflict.
    #
    # Hmm, so actually NO pair of edges is inherently incompatible! Any two edges can coexist in a K2,3-free graph.
    # The incompatibility is higher-order: three or more edges together might force a K2,3.
    #
    # So the conflict graph approach with pairwise conflicts doesn't directly work. 
    # We need a different formulation.
    #
    # ALTERNATIVE: Use the fact that the constraint is |N(u) ∩ N(v)| <= 2 for all u,v.
    # This is equivalent to: for every pair (u,v), at most 2 of the 180 other edges can be "witnesses".
    #
    # Let me try a different B&B: search over edge subsets, using the constraint that
    # for each pair (u,v), the number of common neighbors <= 2.
    # 
    # Actually, let me just implement a smarter local search + exact completion.
    # The current program already does this. Let me focus on making the completion
    # phase more effective and adding a proper B&B on small subproblems.
    
    # Precompute edge index for fast lookup
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # For each edge (u,v), precompute which other edges are "adjacent" in the sense that
    # they share a vertex with (u,v) - these are the ones that affect the K2,3 constraint
    edge_neighbors = [[] for _ in range(num_edges_total)]
    for i, (u, v) in enumerate(all_edges):
        for j, (a, b) in enumerate(all_edges):
            if i == j:
                continue
            # Edges that share a vertex with (u,v)
            if a == u or a == v or b == u or b == v:
                edge_neighbors[i].append(j)
    
    # Helper to check if adding edge (u,v) to adj is valid (no K2,3)
    # Uses a common-neighbors cache for speedup
    def can_add(adj, u, v, cn_cache=None):
        if adj[u] & (1 << v):
            return True
        
        # Check: |N(u) ∩ N(v)| <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # For each w in N(u), |N(w) ∩ N(v)| <= 1
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if cn_cache is not None:
                key = (w, v) if w < v else (v, w)
                cnt = cn_cache.get(key, -1)
                if cnt == -1:
                    cnt = (adj[w] & adj[v]).bit_count()
                    cn_cache[key] = cnt
            else:
                cnt = (adj[w] & adj[v]).bit_count()
            if cnt > 1:
                return False
            mask ^= lb
        
        # For each w in N(v), |N(w) ∩ N(u)| <= 1
        mask = adj[v]
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            if cn_cache is not None:
                key = (w, u) if w < u else (u, w)
                cnt = cn_cache.get(key, -1)
                if cnt == -1:
                    cnt = (adj[w] & adj[u]).bit_count()
                    cn_cache[key] = cnt
            else:
                cnt = (adj[w] & adj[u]).bit_count()
            if cnt > 1:
                return False
            mask ^= lb
        
        return True
    
    # Branch and bound: find maximum K2,3-free subgraph from a set of candidate edges
    # given a current adjacency. Uses upper bound based on remaining edges.
    def branch_and_bound(adj, count, candidate_mask, upper_bound, best_count_ref):
        """
        adj: current adjacency (list of bitmasks)
        count: current edge count
        candidate_mask: bitmask of candidate edges (indices into all_edges) that could be added
        upper_bound: upper bound on edges we can add from candidates
        best_count_ref: list [best_count] for mutation
        Returns: (adj, count, best_count)
        """
        if time.time() > deadline - 5:
            return adj, count, best_count_ref[0]
        
        # Update best
        if count > best_count_ref[0]:
            best_count_ref[0] = count
        
        # No more candidates
        if candidate_mask == 0:
            return adj, count, best_count_ref[0]
        
        # Upper bound check
        if count + upper_bound <= best_count_ref[0]:
            return adj, count, best_count_ref[0]
        
        # Select a candidate edge to branch on
        # Heuristic: pick edge with most conflicts (most constrained)
        lb = candidate_mask & (-candidate_mask)
        edge_i = lb.bit_length() - 1
        candidate_mask ^= lb
        
        u, v = all_edges[edge_i]
        
        # Try including this edge
        if can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            new_count = count + 1
            
            # Recompute candidates: remove edges that now conflict
            new_candidate_mask = 0
            m = candidate_mask
            while m:
                lb2 = m & (-m)
                j = lb2.bit_length() - 1
                a, b = all_edges[j]
                if can_add(adj, a, b):
                    new_candidate_mask |= (1 << j)
                m ^= lb2
            
            # Upper bound: count remaining valid candidates
            remaining_count = new_candidate_mask.bit_count()
            
            adj2, count2, best2 = branch_and_bound(
                adj, new_count, new_candidate_mask, remaining_count, best_count_ref
            )
            if best2 > best_count_ref[0]:
                best_count_ref[0] = best2
            if count2 > count:
                # Restore adj to state before this branch for the other branch
                pass
            
            # Undo
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Try excluding this edge (just recurse without it)
        remaining_count = candidate_mask.bit_count()
        adj, count, best_count_ref[0] = branch_and_bound(
            adj, count, candidate_mask, remaining_count, best_count_ref
        )
        
        return adj, count, best_count_ref[0]

    # Systematic completion: try adding all missing edges in order
    def systematic_complete(adj, count):
        cn_cache = {}
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        continue
                    if can_add(adj, u, v, cn_cache):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
                        improved = True
                        # Invalidate cache entries involving u or v
                        for key in list(cn_cache.keys()):
                            if key[0] in (u, v) or key[1] in (u, v):
                                del cn_cache[key]
        return adj, count
    
    # Advanced B&B: find optimal completion from current state
    def optimal_complete(adj, count, time_limit=10):
        """Use branch and bound to find optimal set of edges to add."""
        start_time = time.time()
        
        # Build list of candidate edges that can be added
        candidates = []
        for i, (u, v) in enumerate(all_edges):
            if not (adj[u] & (1 << v)):
                if can_add(adj, u, v):
                    candidates.append(i)
        
        if not candidates:
            return adj, count
        
        # Build conflict graph among candidates
        # Two candidates conflict if they cannot both be added
        num_cand = len(candidates)
        cand_set = set(candidates)
        conflict = [0] * num_cand  # bitmask of conflicts
        
        for i in range(num_cand):
            u1, v1 = all_edges[candidates[i]]
            for j in range(i + 1, num_cand):
                u2, v2 = all_edges[candidates[j]]
                # Check if these two edges can coexist
                # Temporarily add both to a copy and check
                test_adj = adj[:]
                test_adj[u1] |= (1 << v1)
                test_adj[v1] |= (1 << u1)
                test_adj[u2] |= (1 << v2)
                test_adj[v2] |= (1 << u2)
                
                # Check if this creates any K2,3
                valid = True
                for a in range(n):
                    for b in range(a + 1, n):
                        if (test_adj[a] & test_adj[b]).bit_count() >= 3:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # B&B on the conflict graph (Maximum Independent Set)
        best_size = 0
        best_set = [False] * num_cand
        
        def bbs(idx, current_mask, current_size):
            nonlocal best_size, best_set
            if time.time() > start_time + time_limit:
                return
            
            # Upper bound
            remaining = num_cand - idx
            if current_size + remaining <= best_size:
                return
            
            if idx >= num_cand:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_mask[:]
                return
            
            # Option 1: include vertex idx
            if not any(current_mask[k] for k in range(idx)):
                # Check if idx conflicts with any already selected
                conflicts_with_selected = False
                for k in range(idx):
                    if current_mask[k] and (conflict[idx] & (1 << k)):
                        conflicts_with_selected = True
                        break
                
                if not conflicts_with_selected:
                    current_mask[idx] = True
                    bbs(idx + 1, current_mask, current_size + 1)
                    current_mask[idx] = False
            
            # Option 2: exclude vertex idx
            bbs(idx + 1, current_mask, current_size)
        
        # Use a better ordering: sort by degree in conflict graph
        degrees = [conflict[i].bit_count() for i in range(num_cand)]
        order = sorted(range(num_cand), key=lambda x: -degrees[x])
        
        # Remap
        remap = [0] * num_cand
        for new_idx, old_idx in enumerate(order):
            remap[old_idx] = new_idx
        
        # Rebuild conflict in new order
        conflict2 = [0] * num_cand
        for i in range(num_cand):
            for j in range(num_cand):
                if i != j and (conflict[i] & (1 << j)):
                    conflict2[remap[i]] |= (1 << remap[j])
        
        bbs(0, [False] * num_cand, 0)
        
        # Apply best set
        for i in range(num_cand):
            if best_set[remap[i]]:
                u, v = all_edges[candidates[i]]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # Check entire graph for K2,3
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
    
    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lb
        return edges
    
    # Greedy construction with random ordering
    def greedy():
        adj = [0] * n
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        count = 0
        cn_cache = {}
        for u, v in edge_list:
            if can_add(adj, u, v, cn_cache):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                # Invalidate cache entries involving u or v
                for key in list(cn_cache.keys()):
                    if key[0] in (u, v) or key[1] in (u, v):
                        del cn_cache[key]
        return adj, count
    
    # Local search: try adding missing edges
    def local_add(adj, count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            cn_cache = {}
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue
                if can_add(adj, u, v, cn_cache):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    improved = True
                    # Invalidate cache entries involving u or v
                    for key in list(cn_cache.keys()):
                        if key[0] in (u, v) or key[1] in (u, v):
                            del cn_cache[key]
        return adj, count
    
    # Edge swap: remove one, try to add up to two
    def local_swap(adj, count, max_iters=200):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count == 0:
                break
            
            # Pick random edge to remove
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two random edges
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            cn_cache = {}
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b, cn_cache):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
                    for key in list(cn_cache.keys()):
                        if key[0] in (a, b) or key[1] in (a, b):
                            del cn_cache[key]
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # 2-opt style: remove two edges, try to add three
    def local_swap2(adj, count, max_iters=100):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count < 2:
                break
            
            # Pick two random edges to remove
            e1 = random.choice(all_edges)
            e2 = random.choice(all_edges)
            if e1 == e2:
                continue
            if not (adj[e1[0]] & (1 << e1[1])):
                continue
            if not (adj[e2[0]] & (1 << e2[1])):
                continue
            
            # Remove both
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            count -= 2
            
            # Try to add three edges
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            cn_cache = {}
            for a, b in edge_list:
                if added >= 3:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b, cn_cache):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
                    for key in list(cn_cache.keys()):
                        if key[0] in (a, b) or key[1] in (a, b):
                            del cn_cache[key]
            
            if added < 1:
                # Restore both
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                count += 2
        
        return adj, count
    
    # Degree-based greedy: prioritize adding edges between low-degree vertices
    def degree_greedy():
        adj = [0] * n
        degrees = [0] * n
        edge_list = all_edges[:]
        count = 0
        remaining = set(all_edges)
        cn_cache = {}
        while remaining:
            best_edge = None
            best_score = float('inf')
            sample_size = min(len(remaining), 50)
            sample = random.sample(list(remaining), sample_size)
            for u, v in sample:
                score = degrees[u] + degrees[v]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            if can_add(adj, u, v, cn_cache):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                count += 1
                for key in list(cn_cache.keys()):
                    if key[0] in (u, v) or key[1] in (u, v):
                        del cn_cache[key]
            remaining.discard(best_edge)
        return adj, count
    
    # Simulated annealing-like: accept worse moves with probability
    def simulated_annealing(adj, count, steps=5000):
        T = 1.0
        T_min = 0.01
        for i in range(steps):
            if time.time() > deadline - 2:
                break
            T = T_min + (T - T_min) * (1 - i / steps)
            
            # Pick random edge swap
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two
            added = 0
            edge_list = random.sample(all_edges, min(50, num_edges_total))
            cn_cache = {}
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b, cn_cache):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
                    for key in list(cn_cache.keys()):
                        if key[0] in (a, b) or key[1] in (a, b):
                            del cn_cache[key]
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # Genetic Algorithm with edge-bitset chromosomes
    def ga_crossover(adj1, adj2):
        """Crossover: combine edges from both parents, keeping only valid subsets."""
        adj = [0] * n
        count = 0
        
        # Collect all edges from both parents
        candidate_edges = []
        for u in range(n):
            mask = adj1[u] | adj2[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    candidate_edges.append((u, v))
                mask ^= lb
        
        random.shuffle(candidate_edges)
        cn_cache = {}
        
        for u, v in candidate_edges:
            if can_add(adj, u, v, cn_cache):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                for key in list(cn_cache.keys()):
                    if key[0] in (u, v) or key[1] in (u, v):
                        del cn_cache[key]
        
        return adj, count
    
    def ga_mutation(adj, count, mutation_rate=0.05):
        """Mutation: randomly remove and add edges."""
        # Remove some random edges
        num_remove = int(count * mutation_rate)
        for _ in range(num_remove):
            if count <= 0:
                break
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
        
        # Add some random edges
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        cn_cache = {}
        for u, v in edge_list:
            if adj[u] & (1 << v):
                continue
            if can_add(adj, u, v, cn_cache):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                for key in list(cn_cache.keys()):
                    if key[0] in (u, v) or key[1] in (u, v):
                        del cn_cache[key]
        
        return adj, count
    
    def genetic_algorithm(pop_size=20, generations=50):
        """Full GA with selection, crossover, mutation, and local search polish."""
        # Initialize population
        population = []
        for _ in range(pop_size):
            adj, count = greedy()
            adj, count = local_add(adj, count)
            population.append((adj, count))
        
        # Sort by count descending
        population.sort(key=lambda x: x[1], reverse=True)
        
        best_adj = population[0][0][:]
        best_count = population[0][1]
        
        for gen in range(generations):
            if time.time() > deadline - 10:
                break
            
            new_population = []
            
            # Elitism: keep top 2
            for i in range(min(2, len(population))):
                new_population.append((population[i][0][:], population[i][1]))
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                candidates = random.sample(population, min(3, len(population)))
                parent1 = max(candidates, key=lambda x: x[1])[0]
                candidates = random.sample(population, min(3, len(population)))
                parent2 = max(candidates, key=lambda x: x[1])[0]
                
                # Crossover
                child_adj, child_count = ga_crossover(parent1, parent2)
                
                # Mutation
                child_adj, child_count = ga_mutation(child_adj, child_count, mutation_rate=0.03)
                
                # Local search polish
                child_adj, child_count = local_add(child_adj, child_count)
                
                new_population.append((child_adj, child_count))
            
            # Sort new population
            new_population.sort(key=lambda x: x[1], reverse=True)
            population = new_population
            
            # Update best
            if population[0][1] > best_count:
                best_count = population[0][1]
                best_adj = population[0][0][:]
        
        return best_adj, best_count
    
    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Many diverse greedy starts
    start_time = time.time()
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        adj, count = greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Degree-based greedy
    for i in range(10):
        if time.time() > deadline - 10:
            break
        adj, count = degree_greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 3: Intensive local search on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        # Multiple rounds of improvement
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap2(adj, count, max_iters=100)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            # Systematic completion after swaps
            adj, count = systematic_complete(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 4: Genetic Algorithm
    if time.time() < deadline - 30:
        ga_adj, ga_count = genetic_algorithm(pop_size=15, generations=30)
        if ga_count > best_count:
            best_count = ga_count
            best_adj = ga_adj[:]
        
        # Polish the GA result
        if best_adj is not None:
            adj = best_adj[:]
            count = best_count
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap(adj, count, max_iters=200)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = systematic_complete(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 5: Restart from best with perturbations
    if best_adj is not None and time.time() < deadline - 15:
        for i in range(15):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            count = best_count
            
            # Perturb: remove 2-4 random edges
            remove_count = random.randint(2, 4)
            for _ in range(remove_count):
                if count <= 0:
                    break
                u = random.randint(0, n - 1)
                mask = adj[u]
                if not mask:
                    continue
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if v < u:
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
            
            # Improve
            adj, count = local_add(adj, count)
            adj, count = local_swap(adj, count, max_iters=200)
            adj, count = local_add(adj, count)
            adj, count = systematic_complete(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 6: Final intensive polish
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        count = best_count
        
        # Aggressive final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swaps
        adj, count = local_swap(adj, count, max_iters=500)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final systematic completion
        adj, count = systematic_complete(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    # Final validity check and fix
    if not is_valid(best_adj):
        # Fallback: rebuild from scratch with systematic approach
        adj = [0] * n
        count = 0
        cn_cache = {}
        for u, v in all_edges:
            if can_add(adj, u, v, cn_cache):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
                for key in list(cn_cache.keys()):
                    if key[0] in (u, v) or key[1] in (u, v):
                        del cn_cache[key]
        best_adj = adj
        best_count = count
    
    return adj_to_edges(best_adj)
