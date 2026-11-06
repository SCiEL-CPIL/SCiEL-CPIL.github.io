def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S:
                    return False
        return True

    def min_vertex_cover_size(adj, nodes):
        """
        Find minimum vertex cover size for a small graph.
        adj: dict of node -> set of neighbors
        nodes: list of nodes in the subgraph
        """
        if not nodes:
            return 0, set()
        
        # If graph is very small, brute force or simple heuristic
        # Since conflict graphs in this context are usually sparse and small, 
        # we can use a simple greedy or exact search for small n.
        # For n <= 20, exact is fast.
        
        if len(nodes) <= 20:
            # Exact minimum vertex cover via branching (or just simple heuristic if too slow, 
            # but 20 is small enough for a decent heuristic or even brute force with pruning)
            # Actually, let's use a simple greedy first, then local search on the cover
            # Or better: since we just need the size to compare with gain, 
            # and we need the set to remove.
            
            # Simple Greedy Vertex Cover: pick node with max degree, remove it and its edges.
            # This is a 2-approximation, but often optimal for these sparse conflict graphs.
            remaining_nodes = set(nodes)
            remaining_edges = set()
            for u in nodes:
                for v in adj[u]:
                    if v in remaining_nodes and v != u:
                        remaining_edges.add(frozenset((u, v)))
            
            cover = set()
            deg = {u: 0 for u in remaining_nodes}
            for e in remaining_edges:
                u, v = tuple(e)
                deg[u] += 1
                deg[v] += 1
            
            while remaining_edges:
                # Find node with max degree
                best_node = None
                max_deg = -1
                for u in remaining_nodes:
                    if deg[u] > max_deg:
                        max_deg = deg[u]
                        best_node = u
                
                if best_node is None:
                    break
                    
                cover.add(best_node)
                remaining_nodes.discard(best_node)
                
                # Remove edges incident to best_node
                new_edges = set()
                for e in remaining_edges:
                    u, v = tuple(e)
                    if u == best_node or v == best_node:
                        # Edge removed
                        other = v if u == best_node else u
                        if other in remaining_nodes:
                            deg[other] -= 1
                    else:
                        new_edges.add(e)
                remaining_edges = new_edges
                
            return len(cover), cover
        
        # Fallback for larger (unlikely)
        return len(nodes), set(nodes)

    def solve_instance(vals):
        n = len(vals)
        # Start with all odd numbers (sum-free)
        S = set(x for x in vals if x % 2 != 0)
        
        # Get even numbers
        evens = [x for x in vals if x % 2 == 0]
        
        # For each even number, try to add it
        # We process them in a specific order? Maybe by "cost" of addition?
        # Or just iterate.
        
        # To make it more robust, we can do multiple passes or try adding all at once?
        # Adding one by one is safer.
        
        # Sort evens? Maybe by absolute value or something?
        # Let's try adding them in the order they appear, or sorted.
        # Let's try sorted by value.
        evens.sort()
        
        for x in evens:
            if time.time() > deadline:
                break
            
            # Check if x can be added to current S without conflict
            # Conflict if:
            # 1. x + a = b for a, b in S
            # 2. a + b = x for a, b in S
            # 3. x + a = x (a=0, but 0 not in set)
            
            conflicts = set()
            
            # Case 1: x is the sum of two elements in S
            # a + b = x
            # We need to find pairs (a,b) in S such that a+b=x.
            # If such pairs exist, we must remove at least one from each pair.
            # This forms a set of constraints.
            
            # Case 2: x is part of a sum with an element in S
            # x + a = b (b in S) => a = b - x
            # a + x = b (b in S) => same
            # Also x + x = 2x. If 2x in S, conflict.
            
            # Let's identify all "bad" triples involving x and S.
            # A triple (a, b, c) is bad if a+b=c and all in S U {x}.
            # Since S is sum-free, no bad triples within S.
            # Bad triples must involve x.
            # Types:
            # (x, x, 2x): if 2x in S
            # (x, a, b): if a, b in S and x+a=b
            # (a, b, x): if a, b in S and a+b=x
            
            # Let's build a conflict graph H where nodes are elements of S that are involved in a conflict with x.
            # An edge (u, v) in H means u and v form a conflict that requires removing at least one.
            # Specifically:
            # If u+v = x, then edge (u,v).
            # If u+x = v, then edge (u, v) ? No, if u+x=v, then to add x, we must remove u or v. 
            # Wait, if we remove u, then x+v is not in S (v is still there, x is added, x+v? no, x+v = 2x+u... wait).
            # Let's re-verify.
            # We want S' = (S \ R) U {x} to be sum-free.
            # Conflicts in S' involving x:
            # 1. x + x = 2x. If 2x in S\R, bad.
            # 2. x + u = v. If u, v in S\R, bad.
            # 3. u + v = x. If u, v in S\R, bad.
            
            # So, for each pair (u,v) in S such that u+v=x, we have a constraint: not (u in S\R and v in S\R). i.e., u in R or v in R.
            # For each pair (u,v) in S such that x+u=v (i.e., v-u=x), we have constraint: not (u in S\R and v in S\R). i.e., u in R or v in R.
            # Note: u+v=x and v-u=x are different.
            # v-u=x => v = u+x.
            # u+v=x => v = x-u.
            
            # Also 2x in S: constraint x in S\R (always true since x not in S) OR 2x in R. So if 2x in S, we MUST remove 2x.
            
            # Let's collect constraints.
            # Each constraint is a set of 1 or 2 nodes from S that must be hit by R.
            # This is a Hitting Set problem (or Vertex Cover if all sets size 2).
            # If we have a constraint {2x} (size 1), 2x MUST be in R.
            # If we have constraints {u, v}, it's a Vertex Cover problem on the graph of these pairs.
            
            must_remove = set()
            pairs = []
            
            # Check 2x
            if 2*x in S:
                must_remove.add(2*x)
            
            # Check u+v=x
            # Iterate through S? O(|S|^2) is too slow if done naively for every x.
            # But |S| is at most 240. |S|^2 ~ 50k. 240 * 50k = 12M ops. Acceptable in Python? 
            # 12M set lookups might be ~1-2s. We have 140s. 
            # But we can optimize.
            
            # Let's just iterate.
            S_list = list(S)
            # To avoid duplicates in pairs, we can use a set of frozensets or just process carefully.
            
            # For u in S:
            #   if x - u in S and x - u != u:
            #       pair (u, x-u)
            #   if x - u == u: # 2u=x
            #       # This implies u = x/2. 
            #       # Constraint: not (u in S\R and u in S\R) => u in R.
            #       # So if x/2 in S, must_remove x/2.
            #       if x % 2 == 0 and (x//2) in S:
            #           must_remove.add(x//2)
            
            # For u in S:
            #   v = u + x
            #   if v in S:
            #       pair (u, v)
            
            for u in S_list:
                # Case 1: u + v = x
                v = x - u
                if v in S:
                    if v == u:
                        # 2u = x
                        must_remove.add(u)
                    else:
                        # Ensure we add each pair once
                        # We can just add to a set of pairs
                        if u < v:
                            pairs.append((u, v))
                        else:
                            pairs.append((v, u))
                
                # Case 2: u + x = v
                v = u + x
                if v in S:
                    if u < v:
                        pairs.append((u, v))
                    else:
                        pairs.append((v, u))
            
            # Remove duplicates in pairs
            pairs = list(set(pairs))
            
            # If we have must_remove items, they are forced.
            # Any pair involving a must_remove item is satisfied.
            # Remove those pairs.
            filtered_pairs = []
            for p in pairs:
                if p[0] in must_remove or p[1] in must_remove:
                    continue
                filtered_pairs.append(p)
            
            # Now we need a Vertex Cover for filtered_pairs on the graph induced by nodes in filtered_pairs.
            # Nodes involved:
            nodes_involved = set()
            for p in filtered_pairs:
                nodes_involved.add(p[0])
                nodes_involved.add(p[1])
            
            # Build adjacency
            adj = {n: set() for n in nodes_involved}
            for p in filtered_pairs:
                u, v = p
                adj[u].add(v)
                adj[v].add(u)
            
            # Find Min Vertex Cover
            vc_size, vc_set = min_vertex_cover_size(adj, list(nodes_involved))
            
            # Total removals
            total_removes = must_remove | vc_set
            total_remove_size = len(total_removes)
            
            # Gain
            gain = 1 - total_remove_size
            
            if gain > 0:
                # It is beneficial to add x
                S = (S - total_removes) | {x}
            # If gain == 0, it's neutral. We could add it to increase "potential" for future?
            # Usually, keeping the set smaller (fewer conflicts) is better? 
            # Or larger? 
            # If gain == 0, size doesn't change. 
            # Adding x might block future additions or enable them.
            # Let's stick to strictly positive gain for simplicity, or try gain >= 0?
            # If gain == 0, we remove some and add x. 
            # Let's try gain >= 0 to explore more states? 
            # Risk: we might replace a "good" element with a "bad" one.
            # Let's stick to gain > 0 for now to ensure monotonic increase in size.
            # Wait, if gain > 0, size increases.
            # If we only allow gain > 0, size is monotonically non-decreasing.
            # This is safe.

        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        S = solve_instance(vals)
        
        # Final verification
        if not is_sum_free(S):
            # Fallback
            S = set(x for x in vals if x % 2 != 0)
        
        results.append(sorted(S))
        
        if time.time() > deadline:
            # If we ran out of time, return what we have
            # But we need to process all 3? 
            # The contract says "three subsets". 
            # If we time out, we should return best effort.
            # Since we process sequentially, if we time out on instance 1, we have no results for 2 and 3.
            # We should allocate time per instance.
            pass

    return results
