import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Strategy 1: Multiple random greedy constructions
    for seed in range(200):
        if time.time() > deadline - 30.0:
            break
        
        edges = _greedy_construct(n, all_edges, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 2: Local search improvement on best greedy result
    if time.time() < deadline - 20.0 and best_edges:
        improved = _local_search(n, best_edges, all_edges, deadline - 15.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    # Strategy 3: Conflict-graph branch-and-bound to find better solutions
    if time.time() < deadline - 10.0 and best_edges:
        bnb_result = _bnb_conflict(n, all_edges, best_edges, deadline - 5.0)
        if len(bnb_result) > best_score:
            best_edges = bnb_result
            best_score = len(best_edges)
    
    return best_edges

def _greedy_construct(n, all_edges, seed):
    """
    Greedy construction of K2,3-free graph using bitmasks for speed.
    """
    random.seed(seed)
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    edges = []
    
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    for u, v in shuffled_edges:
        if _can_add_edge_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    return edges

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) is valid using bitmasks.
    adj is a list of ints.
    """
    # Check pair (u, v)
    if (adj[u] & adj[v]).bit_count() >= 3:
        return False
    
    mask_u = adj[u]
    mask_v = adj[v]
    
    # Iterate over all w
    for w in range(len(adj)):
        if w == u or w == v:
            continue
        
        # Check (u, w)
        common_uw = mask_u & adj[w]
        count_uw = common_uw.bit_count()
        if (adj[w] & (1 << v)):
            count_uw += 1
        if count_uw >= 3:
            return False
            
        # Check (v, w)
        common_vw = mask_v & adj[w]
        count_vw = common_vw.bit_count()
        if (adj[w] & (1 << u)):
            count_vw += 1
        if count_vw >= 3:
            return False
            
    return True

def _local_search(n, edges, all_edges, deadline):
    """
    Local search with 1-2 moves (remove 1, add up to 2).
    """
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
        
    current_edges = [list(e) for e in edges]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single edge
        for u, v in all_edges:
            if time.time() > deadline:
                return current_edges
            if (u, v) in edge_set:
                continue
            if _can_add_edge_fast(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                current_edges.append([u, v])
                improved = True
        
        if improved:
            continue # Restart adding edges as the graph changed
        
        # Phase 2: If no single edge can be added, try 1-2 moves
        for e in list(edge_set):
            if time.time() > deadline:
                return current_edges
            
            u1, v1 = e
            # Remove e
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)
            
            # Try to add two edges
            added = []
            for u2, v2 in all_edges:
                if (u2, v2) in edge_set:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    # Add first edge
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edge_set.add((u2, v2))
                    added.append((u2, v2))
                    break
            
            if len(added) == 1:
                u2, v2 = added[0]
                for u3, v3 in all_edges:
                    if (u3, v3) in edge_set:
                        continue
                    if _can_add_edge_fast(adj, u3, v3):
                        # Add second edge
                        adj[u3] |= (1 << v3)
                        adj[v3] |= (1 << u3)
                        edge_set.add((u3, v3))
                        added.append((u3, v3))
                        break
            
            if len(added) == 2:
                # Successful 1-2 move
                current_edges = [[u, v] for u, v in edge_set]
                improved = True
                break
            else:
                # Revert
                if len(added) == 1:
                    u2, v2 = added[0]
                    adj[u2] ^= (1 << v2)
                    adj[v2] ^= (1 << u2)
                    edge_set.remove((u2, v2))
                # Restore original edge
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
                
    return current_edges

def _bnb_conflict(n, all_edges, initial_edges, deadline):
    """
    Branch-and-bound on the conflict graph of edges.
    Two edges conflict if they cannot both be in a K2,3-free graph.
    We seek a maximum independent set in this conflict graph.
    """
    m = len(all_edges)
    
    # Build conflict graph: conflict[i][j] = True if edges i and j conflict
    # Edges conflict if adding both to some context creates K2,3
    # Actually, two edges conflict if they share a vertex and together with 3 common neighbors of the other endpoint... 
    # Simpler: two edges e1=(a,b) and e2=(c,d) conflict if there exist 3 vertices w1,w2,w3 
    # such that adding both edges would create K2,3.
    # Actually, the cleanest definition: two edges conflict if they cannot both appear in any K2,3-free graph.
    # For simplicity, we'll use: edges conflict if they share exactly one vertex and the two non-shared vertices have 3 common neighbors.
    # But this is not quite right either. Let me think more carefully.
    
    # Actually, the correct conflict: edges e1=(u,v) and e2=(u,w) (sharing vertex u) conflict if v and w have at least 3 common neighbors.
    # Edges that don't share a vertex never conflict in the K2,3 sense directly.
    # Wait, K2,3 has two vertices connected to three others. So if we have edges (a,x) and (b,y), these don't directly form K2,3.
    # The conflict is: if we include edges (u,v) and (u,w), then v and w must have at most 2 common neighbors.
    # So the conflict graph: nodes are all possible edges. Two edges (u,v) and (u,w) are adjacent in conflict graph iff v and w have 3+ common neighbors in the final graph.
    # But this depends on the final graph, so it's not a fixed conflict graph.
    
    # Alternative approach: Use a different B&B strategy.
    # Start from initial_edges and try to improve by swapping.
    
    adj = [0] * n
    edge_set = set()
    for u, v in initial_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    current_edges = [list(e) for e in initial_edges]
    best_edges = current_edges[:]
    best_count = len(current_edges)
    
    # Try 2-2 swaps: remove two edges, add two different edges
    if time.time() > deadline:
        return best_edges
    
    # Precompute which edges can potentially be added given current adj
    # For each non-edge, check if it can be added
    addable = []
    for u, v in all_edges:
        if (u, v) in edge_set:
            continue
        if _can_add_edge_fast(adj, u, v):
            addable.append((u, v))
    
    # If we can add edges, do it first
    for u, v in addable:
        if time.time() > deadline:
            return best_edges
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        current_edges.append([u, v])
        best_count += 1
    
    if len(current_edges) > best_count:
        best_edges = current_edges[:]
        best_count = len(current_edges)
    
    # Now try 2-2 swaps
    current_edge_list = list(edge_set)
    if len(current_edge_list) < 2:
        return best_edges
    
    swap_attempts = 0
    max_swaps = 50000
    
    while swap_attempts < max_swaps and time.time() < deadline:
        swap_attempts += 1
        
        # Pick two random edges to remove
        i, j = random.sample(range(len(current_edge_list)), 2)
        e1 = current_edge_list[i]
        e2 = current_edge_list[j]
        
        # Remove them
        u1, v1 = e1
        u2, v2 = e2
        adj[u1] ^= (1 << v1)
        adj[v1] ^= (1 << u1)
        adj[u2] ^= (1 << v2)
        adj[v2] ^= (1 << u2)
        edge_set.remove(e1)
        edge_set.remove(e2)
        
        # Try to add two different edges
        added = []
        for u, v in all_edges:
            if (u, v) in edge_set:
                continue
            if _can_add_edge_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                added.append((u, v))
                break
        
        if len(added) == 1:
            u_a, v_a = added[0]
            for u, v in all_edges:
                if (u, v) in edge_set:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    added.append((u, v))
                    break
        
        if len(added) == 2:
            # Successful swap
            current_edges = [[u, v] for u, v in edge_set]
            if len(current_edges) > best_count:
                best_edges = current_edges[:]
                best_count = len(current_edges)
            current_edge_list = list(edge_set)
        else:
            # Revert
            if len(added) == 1:
                u_a, v_a = added[0]
                adj[u_a] ^= (1 << v_a)
                adj[v_a] ^= (1 << u_a)
                edge_set.remove((u_a, v_a))
            # Restore removed edges
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            edge_set.add(e1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edge_set.add(e2)
    
    return best_edges
