import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)

    def count_violations_full(n, adj):
        """
        Full verification. Returns count of violating 4-sets.
        adj is an n x n boolean matrix where adj[i][j] is 1 if i -> j.
        """
        # Precompute out-masks
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        viol_count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                viol_count += 1
        return viol_count

    def greedy_repair(n, adj, max_iters=20000):
        """
        Greedy local search with exact delta cost calculation.
        Tries to flip edges that reduce the number of violating 4-sets.
        """
        # Precompute out-masks
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        # Precompute neighbor sets for each vertex (in and out)
        # in_neighbors[v] = set of u such that u -> v
        # out_neighbors[v] = set of w such that v -> w
        in_neighbors = [set() for _ in range(n)]
        out_neighbors = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and adj[u][v]:
                    out_neighbors[u].add(v)
                    in_neighbors[v].add(u)
                    
        current_cost = count_violations_full(n, adj)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        if current_cost == 0:
            return best_adj, 0

        # Precompute all 4-sets for faster iteration
        all_quads = list(combinations(range(n), 4))
        
        # We will iterate through random edges and try to flip them if it improves cost.
        # To speed up, we can maintain a list of "potentially useful" edges or just random sample.
        # Given n <= 90, number of edges is ~4000.
        # Number of quads is ~2.5M for n=90.
        # Delta calculation for one edge flip involves checking ~4*n quads? No, more.
        
        # Let's use a simpler heuristic:
        # Randomly pick an edge (u, v).
        # Calculate delta cost.
        # If delta < 0, accept.
        # If stuck, try random acceptance (simulated annealing) or restart.
        
        # To make it faster, we precompute for each vertex v, the set of 4-sets it dominates?
        # That's too much memory.
        
        # Alternative: 
        # For a flip of edge (u, v) from u->v to v->u:
        # 1. u loses out-edge to v, gains in-edge from v.
        # 2. v gains out-edge to u, loses in-edge from u.
        # 
        # A 4-set Q is affected if it contains u or v.
        # If Q doesn't contain u or v, its dominator status is unchanged.
        # 
        # Let's identify the set of 4-sets containing u or v.
        # This is O(n^3) per flip if done naively.
        
        # Optimization:
        # Precompute for each vertex x, the list of 4-sets containing x.
        # quad_index[x] = list of indices in all_quads that contain x.
        # This uses memory: sum(len) = 4 * C(n,4). For n=90, C(90,4) ~ 2.5M. 4*2.5M = 10M integers. ~80MB. Acceptable.
        
        quad_index = [[] for _ in range(n)]
        for idx, quad in enumerate(all_quads):
            for v in quad:
                quad_index[v].append(idx)
                
        # Precompute dominator status for each quad?
        # dom_status[quad_idx] = list of vertices that dominate it?
        # Or just a boolean is_dominated[quad_idx] and count?
        # To calculate delta, we need to know if the quad WAS dominated and IS dominated.
        
        # Let's store:
        # dom_count[quad_idx] = number of dominators for this quad.
        # is_violating[quad_idx] = 1 if dom_count == 0 else 0.
        
        dom_count = np.zeros(len(all_quads), dtype=np.int32)
        is_violating = np.zeros(len(all_quads), dtype=np.int32)
        
        # Initial computation
        for idx, quad in enumerate(all_quads):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            cnt = 0
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    cnt += 1
            dom_count[idx] = cnt
            if cnt == 0:
                is_violating[idx] = 1
                current_cost += 1 # Wait, I already calculated current_cost. Let's just use this.
        
        # Re-sync current_cost
        current_cost = np.sum(is_violating)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        if current_cost == 0:
            return best_adj, 0

        # Helper to update dom_count for a flip of edge (u, v)
        # Assume adj[u][v] is 1 (u->v). We flip to 0 (v->u).
        # u loses out to v. v gains out to u.
        
        def flip_edge(u, v):
            # u -> v becomes v -> u
            # Check current state
            if adj[u][v] == 0:
                # Already v -> u, so we are flipping u <- v to u -> v?
                # The function name is flip, let's assume we want to toggle.
                # Let's define direction clearly.
                # Current: v -> u. Target: u -> v.
                # v loses out to u. u gains out to v.
                pass
            
            # Let's implement specific logic for "change direction of edge between u and v"
            # Let current_dir = 1 if u->v, 0 if v->u.
            # We flip it.
            
            # Identify affected quads: those containing u or v.
            # We need to update dom_count for these quads.
            
            # Which vertices' dominance changes?
            # Only u and v change their out-neighborhoods.
            # So for any quad Q containing u or v, the set of dominators might change 
            # ONLY if u or v were potential dominators of Q.
            # A vertex x dominates Q if x->all(v in Q).
            
            # If Q contains u:
            #   u cannot dominate Q (u is in Q, and no self-loops).
            #   Other vertices might.
            #   Specifically, v's status as a dominator of Q changes if v is not in Q.
            #   If v is in Q, v cannot dominate Q.
            
            # So, for a quad Q:
            # 1. If Q does not contain u or v: No change.
            # 2. If Q contains u but not v:
            #    - v's ability to dominate Q changes.
            #    - u's ability to dominate Q is irrelevant (u in Q).
            # 3. If Q contains v but not u:
            #    - u's ability to dominate Q changes.
            # 4. If Q contains both u and v:
            #    - Neither u nor v can dominate Q.
            #    - No change in dominators from u or v.
            
            # Therefore, we only need to update quads that contain exactly one of {u, v}.
            
            # Let's collect indices of such quads.
            # Quads containing u but not v:
            # From quad_index[u], filter out those containing v.
            # Quads containing v but not u:
            # From quad_index[v], filter out those containing u.
            
            # To avoid double counting or missing, we can just iterate through quad_index[u] and quad_index[v]
            # and check membership.
            
            affected_indices = set()
            for idx in quad_index[u]:
                if v not in all_quads[idx]:
                    affected_indices.add(idx)
            for idx in quad_index[v]:
                if u not in all_quads[idx]:
                    affected_indices.add(idx)
            
            # For each affected quad, recalculate contribution of the changing vertex.
            # Changing vertex for Q containing u (not v) is v.
            # Changing vertex for Q containing v (not u) is u.
            
            delta_cost = 0
            changes = [] # (idx, old_dom_count, new_dom_count)
            
            for idx in affected_indices:
                quad = all_quads[idx]
                a, b, c, d = quad
                quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                
                # Determine which vertex is changing
                if u in quad and v not in quad:
                    changing_v = v
                elif v in quad and u not in quad:
                    changing_v = u
                else:
                    continue # Should not happen based on filter
                
                # Check if changing_v dominated the quad BEFORE flip
                # We need the state of out_mask BEFORE flip.
                # This is tricky if we update in place.
                # Let's calculate "did it dominate before" and "does it dominate after".
                
                # Before flip:
                # If u->v (adj[u][v]=1):
                #   out_mask[u] has v. out_mask[v] does not have u.
                # If v->u (adj[u][v]=0):
                #   out_mask[v] has u. out_mask[u] does not have v.
                
                current_dir = adj[u][v] # 1 if u->v
                
                # Simulate out_mask for changing_v before and after
                # We can just check the specific edge and the rest.
                # Does changing_v point to all in quad?
                
                # Let's just recompute the full dominance check for the changing vertex.
                # It's O(4) checks per quad.
                
                def dominates(vertex, quad_mask, edge_state):
                    # edge_state: 1 if u->v, 0 if v->u
                    # We need to know if vertex -> all nodes in quad.
                    # If vertex == u:
                    #   We need u -> a, u -> b, u -> c, u -> d.
                    #   If v is in quad, we check u->v based on edge_state.
                    #   If u is in quad, return False.
                    # If vertex == v:
                    #   We need v -> a, ...
                    #   If u is in quad, we check v->u based on edge_state (inverse of u->v).
                    
                    # Since we already filtered out quads containing the changing vertex?
                    # No, we filtered out quads containing BOTH.
                    # So changing_v is NOT in quad.
                    # The other vertex (u or v) IS in quad.
                    
                    if vertex == u:
                        # u is not in quad. v is in quad.
                        # We need u -> all nodes in quad.
                        # One node is v. u->v is current_dir.
                        # Other nodes are in quad \ {v}.
                        # We need to check adj[u][w] for w in quad \ {v}.
                        for w in quad:
                            if w == v:
                                if current_dir != 1: # u->v is false
                                    return False
                            else:
                                if adj[u][w] != 1:
                                    return False
                        return True
                    else: # vertex == v
                        # v is not in quad. u is in quad.
                        # We need v -> all nodes in quad.
                        # One node is u. v->u is (1 - current_dir).
                        for w in quad:
                            if w == u:
                                if (1 - current_dir) != 1: # v->u is false
                                    return False
                            else:
                                if adj[v][w] != 1:
                                    return False
                        return True

                was_dominating = dominates(changing_v, quad_mask, current_dir)
                
                # Now simulate flip
                new_dir = 1 - current_dir
                is_dominating_after = dominates(changing_v, quad_mask, new_dir)
                
                old_count = dom_count[idx]
                new_count = old_count
                if was_dominating:
                    new_count -= 1
                if is_dominating_after:
                    new_count += 1
                    
                # Update violation status
                old_viol = 1 if old_count == 0 else 0
                new_viol = 1 if new_count == 0 else 0
                
                if old_viol != new_viol:
                    delta_cost += new_viol - old_viol
                    changes.append((idx, old_count, new_count))
                    
            return delta_cost, changes, affected_indices, current_dir

        # Main Loop
        T = 1.0
        cooling = 0.999
        no_improve_counter = 0
        
        for i in range(max_iters):
            if time_left() < 5:
                break
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            delta, changes, affected, current_dir = flip_edge(u, v)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                # Apply flip to adj
                adj[u][v] = 1 - current_dir
                adj[v][u] = 1 - adj[u][v]
                
                # Update out_mask (not strictly needed for delta calc if we use adj, but good for consistency)
                # Actually, my dominates() function uses adj directly, so out_mask is not used in the inner loop.
                # I should remove out_mask dependency in the inner loop to avoid sync issues.
                # The current implementation of flip_edge uses adj.
                
                # Apply changes to dom_count and is_violating
                for idx, old_c, new_c in changes:
                    dom_count[idx] = new_c
                    if new_c == 0:
                        is_violating[idx] = 1
                    else:
                        is_violating[idx] = 0
                        
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
                    no_improve_counter = 0
                    if best_cost == 0:
                        break
                else:
                    no_improve_counter += 1
                    
                # Simple restart logic: if stuck for too long, maybe randomize a bit?
                # Or just keep going.
                
                T *= cooling
            else:
                # Reject
                pass
                
        return best_adj, best_cost

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try Greedy Repair for smaller n
    # Start from lower bound 31
    for n in range(31, 83):
        if time_left() < 10:
            break
            
        # Try a few random seeds
        for seed in range(3):
            if time_left() < 5:
                break
                
            random.seed(seed)
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
                        
            best_adj, best_cost = greedy_repair(n, adj, max_iters=10000)
            
            if best_cost == 0:
                best_n = n
                best_orient = adj_to_orient(n, best_adj)
                break
                
        if best_n < 83:
            break
            
    return {'n': best_n, 'orientation': best_orient}
