def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals):
        n = len(vals)
        if n == 0:
            return set()
        
        # Build conflict graph
        # Node i conflicts with node j if vals[i] + vals[j] is in vals
        # We need to find Max Independent Set in this graph.
        # Equivalently, Max Clique in the complement graph (where edges represent "can be together").
        # But constructing complement of dense graph is expensive.
        # Let's stick to MIS on conflict graph.
        
        val_set = set(vals)
        # adj[i] is a set of indices j such that vals[i] + vals[j] is in val_set
        # Note: if vals[i] + vals[i] is in val_set, i conflicts with i (self-loop). 
        # In MIS, self-loops mean the node cannot be in the set.
        # Actually, sum-free definition: no a+b=c. If a=b, then 2a=c.
        # So if 2*vals[i] in val_set, then vals[i] cannot be in the sum-free set?
        # Wait. If S is sum-free, then for any a, b in S, a+b not in S.
        # If a=b, then 2a not in S.
        # So if x is in S, then 2x cannot be in S.
        # This means x and 2x cannot both be in S.
        # Also if x+y=z, then x,y,z cannot all be in S.
        
        # Conflict Graph G:
        # Nodes: 0..n-1
        # Edge (i, j) exists if:
        # 1. i != j and vals[i] + vals[j] is in val_set
        # 2. i == j and 2*vals[i] is in val_set (self-conflict, effectively removes node from consideration? 
        #    No, if i has a self-conflict, it means if i is in S, then 2*vals[i] must NOT be in S.
        #    But the condition "a+b=c with a,b,c in S" is violated if a=b=i and c is some k.
        #    So i and k cannot both be in S.
        #    This is just an edge (i, k) where vals[k] = 2*vals[i].
        #    So we don't need self-loops in the graph for MIS logic, just edges between distinct nodes.
        #    However, if vals[i] + vals[i] = vals[i] (implies vals[i]=0), but 0 is excluded.
        #    So self-loops don't exist in the sense of "i conflicts with i". 
        #    The constraint is between i and k where vals[k] = 2*vals[i].
        
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                vj = vals[j]
                s = vi + vj
                if s in val_set:
                    # Find index of s? 
                    # We need to know if s is in vals. 
                    # If s is in vals, say at index k, then the triple (i, j, k) is a conflict.
                    # This means we cannot have all three in S.
                    # This is NOT a simple graph edge problem!
                    # MIS on a graph handles "cannot have both endpoints".
                    # Here we have "cannot have all three".
                    # This is a 3-uniform hypergraph independent set problem.
                    # Or, we can model it differently.
                    
                    # Let's re-read the problem.
                    # "no a + b = c with a, b, c in it"
                    # This is a 3-set packing problem (or 3-uniform hypergraph independent set).
                    
                    # However, the inspiration mentioned "Maximum 3-Set Packing" and "MIS on conflict graph".
                    # How do we reduce 3-HIP to MIS?
                    # We can't directly.
                    # But notice the structure.
                    # If we pick a subset S, it is sum-free if for every x, y in S, x+y not in S.
                    
                    # Alternative approach:
                    # The current program used a greedy add approach.
                    # The inspiration 1 used "MIS solver on the 'conflict graph'".
                    # Is it possible they constructed a graph where vertices are elements and edges represent "conflict" 
                    # such that an independent set is sum-free?
                    # If edge (i,j) exists if vals[i] + vals[j] is in vals, then an independent set I 
                    # means for all i,j in I, vals[i]+vals[j] is NOT in vals.
                    # This is EXACTLY the definition of sum-free!
                    # Wait. 
                    # Definition: S is sum-free if for all a, b in S, a+b not in S.
                    # Graph G: V=vals, E={(i,j) | vals[i]+vals[j] in vals}.
                    # If I is an independent set in G, then for all i, j in I, (i,j) not in E.
                    # This means vals[i]+vals[j] is NOT in vals.
                    # Since I is a subset of vals, this means vals[i]+vals[j] is NOT in I.
                    # Yes! An independent set in this conflict graph IS a sum-free set.
                    
                    # So we just need the Maximum Independent Set of this graph.
                    # The graph can be dense.
                    # n=240.
                    # Tomita algorithm for Max Clique on the complement graph.
                    # Complement graph H: Edge (i,j) in H if (i,j) NOT in G.
                    # i.e., vals[i]+vals[j] NOT in vals.
                    # We want Max Clique in H.
                    
                    # Building H is O(n^2).
                    # n=240, n^2 ~ 57k. Very fast.
                    
                    # Let's build H adjacency lists.
                    pass
        
        # Build Complement Graph H
        # h_adj[i] is a set of j such that vals[i] + vals[j] is NOT in val_set
        h_adj = [set() for _ in range(n)]
        
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                vj = vals[j]
                s = vi + vj
                if s in val_set:
                    # Conflict, so NO edge in H
                    pass
                else:
                    # No conflict, so EDGE in H
                    h_adj[i].add(j)
                    h_adj[j].add(i)
        
        # Now solve Max Clique in H using Tomita's algorithm.
        # Tomita algorithm uses coloring to bound the search.
        # We will implement a standard recursive branch and bound.
        
        best_clique = []
        best_size = 0
        
        # Order vertices by degree in H (descending) to help pruning?
        # Tomita usually orders by color in the coloring step.
        # Let's use the standard MCS (Maximum Cardinality Search) or just degree ordering for initial ordering.
        
        # For n=240, we need an efficient implementation.
        # Python recursion might be slow.
        # We will use a stack-based or optimized recursive approach.
        
        # Tomita's Algorithm (MCQ):
        # 1. Order vertices.
        # 2. Color the subgraph induced by remaining vertices.
        # 3. Branch on vertices in order of decreasing color.
        
        # Let's implement a simplified but effective version.
        
        # To speed up, we can use bitsets for adjacency if n is small enough, 
        # but n=240 requires 4 64-bit words or just Python sets.
        # Python sets are okay for n=240.
        
        def color_sort(P, h_adj):
            # Greedy coloring to get upper bound
            # Returns list of (vertex, color) sorted by color descending
            uncolored = set(P)
            colors = {}
            color_class = [] # list of sets of vertices
            
            while uncolored:
                # Start new color class
                current_color_set = set()
                # Pick vertex with max degree in remaining? Or just first?
                # Standard greedy: pick vertex, remove it and its neighbors from uncolored for this color
                # Actually, simple greedy coloring:
                # Iterate through uncolored, assign smallest available color.
                # But for Tomita, we want to partition into color classes where each class is an independent set in H (clique in G).
                # Wait, in Tomita, we color the graph H.
                # An independent set in H is a clique in G.
                # We want a clique in H.
                # The coloring bound: The size of the max clique is <= the number of colors used.
                # Specifically, if we color H, any clique can contain at most one vertex of each color.
                
                # Greedy coloring of H:
                # We process vertices in P.
                # Let's use a simple greedy coloring.
                available_colors = [set() for _ in range(len(P) + 1)]
                # To make it fast, we just assign colors.
                
                # Better: Use the specific Tomita coloring which also orders vertices.
                # Let's implement a standard greedy coloring that returns the color of each vertex.
                # Then sort vertices by color.
                
                # For n=240, we can just do a simple greedy coloring.
                # Order of processing affects bound quality.
                # Let's process in reverse degree order?
                
                # Simple implementation:
                # uncolored is a set.
                # We will assign colors 1, 2, ...
                # For each color c, we find an independent set in H (which is a clique in G? No, independent in H).
                # Wait.
                # H is the "compatible" graph.
                # We want Max Clique in H.
                # Coloring H: If v and u have same color, they are NOT adjacent in H.
                # So they cannot be in the same clique in H.
                # So Max Clique size <= Max Color.
                
                # Greedy Coloring of H:
                # 1. Order vertices in P (e.g., by degree in H).
                # 2. For each vertex v in order:
                #    Assign the smallest color c such that no neighbor of v has color c.
                
                # This is O(n^2) per coloring step.
                # With n=240, this is fine.
                
                # Let's optimize by using the "color bound" directly in the recursion.
                
                pass # Placeholder for actual implementation below

        # Since implementing full Tomita with coloring in Python from scratch is verbose and error-prone,
        # and n=240 is on the edge for pure Python speed, 
        # let's use a simpler heuristic if Tomita is too slow, or a very optimized Tomita.
        
        # Alternative: Since the graph is random-ish, maybe a good greedy + local search is enough?
        # The current program scores 1.45.
        # The inspirations score 1.58.
        # Let's try a robust Greedy + Local Search (Simulated Annealing or Hill Climbing) which is easier to implement correctly.
        
        # Actually, let's try the MIS via Branch and Bound with a simple degree heuristic, which is often sufficient for sparse/dense random graphs.
        
        # Let's implement a basic Max Clique solver on H.
        
        # To avoid deep recursion limits and speed issues, we'll use a stack.
        
        # However, for n=240, even a good B&B can be slow in Python if the graph is hard.
        # Random graphs are usually easy for B&B.
        
        # Let's use the following strategy:
        # 1. Greedy construction of a large clique.
        # 2. Local search (k-exchange) to improve.
        
        def greedy_clique(h_adj):
            # Pick vertex with max degree, add to clique, remove neighbors, repeat.
            # But we want MAX clique.
            # Greedy: 
            # Candidates = all vertices
            # Clique = []
            # While Candidates:
            #   Pick v with max degree in induced subgraph of Candidates
            #   Add v to Clique
            #   Remove v and its neighbors from Candidates
            # This finds A clique, not necessarily max.
            
            # Better Greedy:
            # Order vertices by degree.
            # Iterate, add if compatible with all in clique.
            
            # Let's use a randomized greedy with multiple restarts.
            pass

        # Given the time constraints and the need for a structural change, 
        # let's implement a **Simulated Annealing** on the set of vertices.
        # State: Binary vector x of length n.
        # Objective: Maximize sum(x) subject to sum-free constraint.
        # But checking sum-free is O(n^2) or O(n log n).
        # With n=240, O(n^2) = 57k ops. 
        # If we do 10,000 steps, that's 570M ops. Too slow in Python.
        
        # We need a faster check or update.
        # When we flip one bit, we only need to check conflicts involving that element.
        # Conflict check for adding x: 
        #   For all y in S, x+y not in S.
        #   This is O(|S|) lookups.
        # When we remove x:
        #   No new conflicts created.
        
        # So a local search step (add/remove/swap) can be checked in O(n).
        # This makes Simulated Annealing or Hill Climbing feasible.
        
        # Strategy:
        # 1. Start with a greedy sum-free set.
        # 2. Local Search: Try to add any vertex not in S. If it causes conflict, try to remove one conflicting vertex.
        #    Specifically, look for a "swap" (add x, remove y) that maintains sum-free property and increases size (or keeps size same for diversity).
        #    Or just "add" if possible.
        #    If no add possible, try "swap".
        #    If no swap improves size, try "random restart" or "perturbation".
        
        def greedy_sum_free(vals):
            # Sort by absolute value? Or just random order?
            # Let's try sorting by value.
            S = set()
            # Try adding in order of increasing "conflict potential"?
            # Simple: iterate through sorted vals.
            for v in sorted(vals):
                # Check if v can be added
                ok = True
                for u in S:
                    if v + u in S:
                        ok = False
                        break
                if ok:
                    S.add(v)
            return S

        def local_search(vals, S, time_limit):
            S_list = list(S)
            S_set = set(S)
            not_in_S = [v for v in vals if v not in S_set]
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                
                # Try to add any vertex from not_in_S
                # Sort not_in_S by potential (e.g., number of conflicts)
                # To be fast, just iterate.
                
                # First, try simple additions
                new_not_in_S = []
                for v in not_in_S:
                    # Check if v can be added to S
                    ok = True
                    for u in S_set:
                        if v + u in S_set:
                            ok = False
                            break
                    if ok:
                        S_set.add(v)
                        S_list.append(v)
                        improved = True
                        # Remove v from not_in_S (handled by rebuilding or flag)
                        # We'll rebuild not_in_S after the loop for simplicity if many added
                    else:
                        new_not_in_S.append(v)
                
                not_in_S = new_not_in_S
                
                if not not_in_S:
                    break
                
                # If no simple additions, try swaps
                # For each v in not_in_S, find conflicts.
                # If we remove one conflict u, can we add v?
                # This is O(n^3) in worst case if not careful.
                
                # Let's try a few random swaps.
                for _ in range(100): # Limit swaps per iteration
                    if time.time() > time_limit:
                        break
                    if not not_in_S:
                        break
                    
                    v = not_in_S[Random().randint(0, len(not_in_S)-1)]
                    
                    # Find conflicts of v with S
                    conflicts = []
                    for u in S_set:
                        if v + u in S_set:
                            conflicts.append(u)
                        # Also check if v + v in S_set? 
                        # If 2v in S_set, then v conflicts with itself? 
                        # No, v is not in S. 
                        # But if we add v, we need 2v not in S.
                        # So if 2v in S_set, that's a conflict too (with itself, effectively).
                        # The check `if v + u in S_set` covers u=v if v was in S, but v is not.
                        # So we must explicitly check 2v.
                        if 2*v in S_set:
                            # This is a conflict that requires removing 2v
                            # But 2v is in S. 
                            # So removing 2v resolves this.
                            # Is 2v in conflicts? 
                            # If u = 2v, then v+u = 3v. Not necessarily in S.
                            # The condition for v to be addable is:
                            # 1. 2v not in S
                            # 2. For all u in S, v+u not in S
                            
                            # So if 2v in S, we MUST remove 2v to add v.
                            # Let's add 2v to conflicts if it's not already there.
                            # Actually, the loop `for u in S_set` checks `v+u in S_set`.
                            # If u=2v, it checks `3v in S_set`.
                            # It does NOT check `2v in S_set` directly unless u=v (which is false).
                            # So we need to handle 2v separately.
                            
                            if 2*v not in conflicts:
                                conflicts.append(2*v)
                            break # No need to check other u for the 2v part? 
                            # Wait, there might be other conflicts too.
                            # We want to remove ONE vertex to resolve ALL conflicts?
                            # No, removing one vertex u resolves conflicts where u is involved.
                            # But v might have multiple conflicts.
                            # If v has conflicts {u1, u2}, removing u1 doesn't resolve conflict with u2.
                            # So a single swap (add v, remove u) only works if v has EXACTLY ONE conflict u.
                            # Or if removing u resolves all conflicts?
                            # Conflict (v, u, w) means v+u=w.
                            # If we remove u, this triple is gone.
                            # But v might have another conflict (v, u2, w2).
                            # So we need v to have only one "essential" conflict?
                            # Actually, if we remove u, we must ensure no other conflicts remain.
                            # So we check: if we remove u, is v addable?
                            
                            pass
                    
                    if not conflicts:
                        # Should have been added in previous step
                        continue
                    
                    # Try removing each conflict
                    # To be efficient, pick a random conflict
                    u_to_remove = Random().choice(conflicts)
                    
                    # Check if removing u_to_remove allows adding v
                    # New S = (S - {u_to_remove}) U {v}
                    # Check sum-free property of new S.
                    # This is O(n).
                    
                    # Optimization: 
                    # We know S is sum-free.
                    # New conflicts can only involve v or u_to_remove.
                    # Since u_to_remove is removed, conflicts involving it are gone.
                    # Conflicts involving v:
                    # 1. 2v in S - {u_to_remove}
                    # 2. v + u in S - {u_to_remove} for all u in S - {u_to_remove}
                    
                    # Check 1:
                    if 2*v in S_set and 2*v != u_to_remove:
                        continue
                    
                    # Check 2:
                    ok = True
                    for u in S_set:
                        if u == u_to_remove:
                            continue
                        if v + u in S_set and (v + u) != u_to_remove:
                            # Conflict remains
                            ok = False
                            break
                    
                    if ok:
                        # Perform swap
                        S_set.discard(u_to_remove)
                        S_set.add(v)
                        S_list = list(S_set)
                        # Update not_in_S
                        not_in_S = [x for x in not_in_S if x != v]
                        not_in_S.append(u_to_remove)
                        improved = True
                        break # Restart addition phase? Or continue?
                        # Let's restart the loop to try more additions.
            
            return S_set

        # Execute for each instance
        S = greedy_sum_free(vals)
        # Allocate time for local search
        # We have ~40s per instance roughly.
        # Let's give 30s to local search.
        ls_deadline = time.time() + 30.0
        S = local_search(vals, S, ls_deadline)
        
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback: return odd numbers for remaining
            vals = instance(size, seed)
            S = set(x for x in vals if x % 2 != 0)
            results.append(sorted(S))
            continue
            
        vals = instance(size, seed)
        S = solve_instance(vals)
        
        # Final verification
        if not is_sum_free(S):
            # Fallback
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))

    return results
