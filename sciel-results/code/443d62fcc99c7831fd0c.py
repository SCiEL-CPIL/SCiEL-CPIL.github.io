import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    vertices = list(range(n))
    
    # We want to maximize edges.
    # A star graph (current seed) has n-1 edges.
    # A complete graph has n*(n-1)/2 edges.
    # Constraint: No simple path of length 3 (4 vertices s-u-v-t) can have deg(s) == deg(t).
    
    # Strategy: Greedy addition.
    # Start with empty graph.
    # Try adding edges in a specific order or random order?
    # To maximize edges, we should try to add "central" edges first? 
    # Or maybe random restarts with different edge orderings?
    
    # Let's try a greedy approach that prioritizes edges between high-degree potential nodes?
    # Actually, degrees change as we add edges.
    # If we add edges randomly, we might get stuck in a local minimum.
    # Let's try multiple random restarts.
    
    best_edges = []
    best_count = 0
    
    # Pre-calculate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        
        current_edges = set()
        degrees = [0] * n
        adj = [set() for _ in range(n)]
        
        valid = True
        current_count = 0
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u, v) creates a violation.
            # A violation exists if there is a path s-u-v-t or s-v-u-t such that deg(s) == deg(t).
            # Note: deg(s) and deg(t) are degrees in the graph AFTER adding (u, v)?
            # "degrees are measured in the whole graph" -> Yes, the candidate graph including the new edge.
            # So if we add (u, v):
            # deg(u) becomes deg(u)+1
            # deg(v) becomes deg(v)+1
            # Other degrees remain same.
            
            # The path is s-u-v-t.
            # s is a neighbor of u (s != v, s != t)
            # t is a neighbor of v (t != u, t != s)
            # s != t
            # Condition: deg(s) == deg(t) in the new graph.
            
            # Let's simulate the degrees.
            # new_deg[u] = deg[u] + 1
            # new_deg[v] = deg[v] + 1
            # new_deg[x] = deg[x] for x not in {u, v}
            
            # We need to check all pairs (s, t) such that:
            # s in adj[u] (s != v)
            # t in adj[v] (t != u)
            # s != t
            # AND deg_new[s] == deg_new[t]
            
            # Optimization:
            # deg_new[s] is either deg[s] (if s not u,v) or deg[u]+1 (if s==u, but s!=v and s in adj[u] implies s!=u in simple graph? No, s is neighbor of u, so s!=u).
            # Wait, s is a neighbor of u. So s != u.
            # t is a neighbor of v. So t != v.
            # So s can be u? No, s is neighbor of u, so s != u.
            # Can s be v? No, s != v is required for simple path s-u-v-t (distinct vertices).
            # So s is not u or v. Thus deg_new[s] = deg[s].
            # Similarly, t is not u or v. Thus deg_new[t] = deg[t].
            
            # Wait, is it possible that s == v? No, path is simple, vertices distinct.
            # Is it possible that t == u? No.
            
            # So the condition simplifies to:
            # There exist s in adj[u] \ {v} and t in adj[v] \ {u} such that s != t and deg[s] == deg[t].
            
            # This check is O(deg(u) * deg(v)).
            
            # Let's verify the logic.
            # Path s-u-v-t.
            # Edges: (s,u), (u,v), (v,t).
            # s != t.
            # s != v (since s-u-v, if s=v then loop or back, but simple path).
            # t != u (since u-v-t).
            # s != u (simple path).
            # t != v (simple path).
            
            # Degrees in full candidate graph (with new edge (u,v)):
            # deg_new(u) = deg(u) + 1
            # deg_new(v) = deg(v) + 1
            # deg_new(s) = deg(s) (since s is not u or v)
            # deg_new(t) = deg(t) (since t is not u or v)
            
            # So we just check if any s in N(u)\{v} and t in N(v)\{u} with s!=t have deg(s) == deg(t).
            
            # If such a pair exists, adding (u,v) is INVALID.
            
            is_valid_edge = True
            neigh_u = [x for x in adj[u] if x != v]
            neigh_v = [x for x in adj[v] if x != u]
            
            # We can optimize this check.
            # If there is any degree value d that appears in both deg[s] for s in neigh_u and deg[t] for t in neigh_v,
            # AND there exists s != t with that degree, then it's invalid.
            
            # Actually, if deg[s] == deg[t], we just need s != t.
            # If the sets of neighbors with a specific degree are disjoint, it's fine?
            # No. If s1 in neigh_u has deg d, and t1 in neigh_v has deg d.
            # If s1 != t1, it's a violation.
            # If s1 == t1, is it a violation?
            # Path s-u-v-s? No, s != t. If s1 == t1, then s == t, which is not allowed for a simple path of 4 distinct vertices.
            # So if the ONLY pair with equal degrees is (s, t) where s == t, then it's NOT a violation.
            
            # So:
            # For each degree value d:
            #   S_d = {s in neigh_u | deg[s] == d}
            #   T_d = {t in neigh_v | deg[t] == d}
            #   If S_d is not empty and T_d is not empty:
            #       If there exists s in S_d and t in T_d such that s != t, then INVALID.
            #       This is equivalent to:
            #       If len(S_d) + len(T_d) > 1 OR (len(S_d) == 1 and len(T_d) == 1 and S_d[0] != T_d[0])
            #       Wait, if S_d = {a} and T_d = {a}, then s=a, t=a, s==t, so no violation for this pair.
            #       If S_d = {a} and T_d = {b} with a!=b, violation.
            #       If S_d = {a, b} and T_d = {a}, then pair (b, a) has b!=a, violation.
            
            # So condition for validity for degree d:
            # The only possible equal-degree pairs must be identical vertices.
            # i.e., S_d and T_d must be subsets of each other? No.
            # If S_d = {a, b} and T_d = {a, b}, then (a, b) is a valid pair with a!=b. Violation.
            # If S_d = {a} and T_d = {a}, valid.
            # If S_d = {a, b} and T_d = {a}, then (b, a) is a violation.
            
            # So, for a given d, it is SAFE only if S_d == T_d == {x} for some x.
            # If S_d or T_d has size > 1, it's unsafe (unless the other is empty, but we are iterating d where both non-empty).
            # If S_d = {a} and T_d = {b}, safe only if a == b.
            
            # Let's re-verify.
            # Violation if exists s in S_d, t in T_d such that s != t.
            # This is false (i.e., safe) if for all s in S_d, t in T_d, s == t.
            # This implies S_d == T_d and |S_d| == 1.
            
            # So:
            # For each d:
            #   if deg_count_in_neigh_u[d] > 0 and deg_count_in_neigh_v[d] > 0:
            #       if deg_count_in_neigh_u[d] > 1 or deg_count_in_neigh_v[d] > 1:
            #           is_valid_edge = False
            #           break
            #       else:
            #           # Both counts are 1
            #           s_val = [x for x in neigh_u if deg[x] == d][0]
            #           t_val = [x for x in neigh_v if deg[x] == d][0]
            #           if s_val != t_val:
            #               is_valid_edge = False
            #               break
            
            if is_valid_edge:
                # Check the degree condition efficiently
                # Build maps of degree -> count and degree -> node (if count==1)
                # Actually, just iterate. n is small (15).
                
                # Simple check:
                for s in neigh_u:
                    ds = degrees[s]
                    for t in neigh_v:
                        if s == t:
                            continue
                        if degrees[t] == ds:
                            is_valid_edge = False
                            break
                    if not is_valid_edge:
                        break
            
            if is_valid_edge:
                # Add edge
                current_edges.add((u, v))
                current_edges.add((v, u))
                degrees[u] += 1
                degrees[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges)
            
        # If we found a very high count, we can stop early?
        # Max possible edges is 105.
        # If best_count is close to max, maybe stop.
        # But let's run all restarts within time limit.
        
    # Convert best_edges to list of lists of ints
    # Ensure no duplicates, no loops
    final_edges = []
    seen = set()
    for e in best_edges:
        u, v = e
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            final_edges.append([int(u), int(v)])
            
    return final_edges
# EVOLVE-BLOCK-END
