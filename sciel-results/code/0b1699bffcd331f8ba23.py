import time
import random
from collections import defaultdict

def instance(n, p, seed):
    """Copy of the verifier's instance generator."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, deadline):
    """
    Decompose a graph's edges into the minimum number of edge-disjoint simple paths.
    Uses a greedy initialization followed by local search (path merging).
    """
    if not edges:
        return []
    
    # Build adjacency and edge set
    adj = defaultdict(set)
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Greedy Initialization
    # We want to cover all edges. 
    # Strategy: Find a path cover.
    # A common heuristic: 
    # - Start with all edges as separate paths.
    # - Repeatedly try to merge two paths P1 and P2 into one path if there is an edge (u,v) 
    #   where u is an endpoint of P1 and v is an endpoint of P2, and (u,v) is not used.
    #   Wait, the edges in the paths *are* the used edges. The "unused" edges are those not yet in any path.
    #   Actually, the problem is to partition the set of edges E into paths.
    #   So, initially, every edge is a path of length 1.
    #   We have a multiset of paths. We want to merge them.
    #   Two paths P1 = [v1, ..., vk] and P2 = [v_{k+1}, ..., v_m] can be merged if 
    #   there is an edge (vk, v_{k+1}) in the graph. But wait, the edge (vk, v_{k+1}) must be 
    #   available. Since we are partitioning E, the edge (vk, v_{k+1}) must be one of the edges 
    #   we haven't assigned to a path yet? No, that's not right.
    
    # Let's rethink.
    # We have a set of edges E. We want to partition E into paths.
    # A path is a sequence of vertices v0, v1, ..., vk such that (v_i, v_{i+1}) is an edge in E.
    # The paths must be edge-disjoint and cover E.
    
    # Greedy approach:
    # 1. Start with each edge as a separate path.
    # 2. We have a pool of "current paths".
    # 3. We also have a pool of "unused edges"? No, all edges are used in the initial paths.
    #    So, how can we merge?
    #    If we have path P1 = [u, v] (edge (u,v)) and path P2 = [v, w] (edge (v,w)),
    #    we can merge them into [u, v, w]. This uses edges (u,v) and (v,w).
    #    This reduces the number of paths by 1.
    #    The condition for merging P1=[..., a, b] and P2=[c, ..., d] is that there exists
    #    an edge (b, c) in the graph? No, the edge (b,c) must be part of the partition.
    #    But in the initial state, (b,c) is its own path [b, c].
    #    So, if we have paths [a, b], [b, c], [c, d], we can merge [a,b] and [b,c] into [a,b,c].
    #    Now we have [a,b,c] and [c,d]. We can merge [a,b,c] and [c,d] into [a,b,c,d].
    #    
    #    So the operation is:
    #    Find two paths P1 and P2 such that an endpoint of P1 (say u) and an endpoint of P2 (say v)
    #    are connected by an edge (u,v) that is currently a path of length 1 (i.e., just the edge (u,v)).
    #    Wait, if (u,v) is a path of length 1, it is just the edge.
    #    If we merge P1 (ending in u) and the edge-path (u,v) and P2 (starting with v), 
    #    we are effectively concatenating them.
    #    
    #    More generally, we can maintain a set of "active paths".
    #    Initially, active_paths = { [u, v] for (u, v) in E }.
    #    We want to reduce the size of active_paths.
    #    A merge is possible if we have two paths P1, P2 in active_paths such that:
    #    - P1 ends at u, P2 starts at v (or vice versa)
    #    - There is an edge (u, v) in E.
    #    - BUT, the edge (u, v) must be "free" to be used as the connector.
    #      In the partition model, every edge is in exactly one path.
    #      If (u, v) is already in P1 or P2, we can't use it as a connector unless it's the endpoint edge.
    #      
    #    Let's look at the structure of a path cover.
    #    If we have a path cover of E, we can try to improve it.
    #    A simple local search:
    #    1. Start with a random valid path cover (e.g., all edges separate).
    #    2. While time permits:
    #       a. Pick two random paths P1, P2 from the current cover.
    #       b. Try to find a way to merge them into a single path, or restructure them 
    #          to reduce the total count or set up for a better merge.
    #       c. Specifically, if P1 ends at u and P2 starts at v, and (u,v) is an edge in E 
    #          that is NOT currently in P1 or P2, we can't just merge them because (u,v) is 
    #          "occupied" by another path.
    #          
    #    Alternative View: 
    #    This is equivalent to finding a minimum path cover in a directed graph where 
    #    each undirected edge (u,v) is replaced by two directed edges (u,v) and (v,u), 
    #    with the constraint that we pick a subset of these directed edges that forms a set of 
    #    vertex-disjoint paths (in the directed sense) covering all vertices? No, covering all edges.
    #    
    #    Actually, a path decomposition of an undirected graph G into k paths is equivalent to 
    #    adding k/2 pairs of edges? No.
    #    
    #    Let's stick to the constructive merging.
    #    State: A list of paths. Each path is a list of vertices.
    #    Invariant: The set of edges used by all paths is exactly E, and no edge is used twice.
    #    Initial State: Each edge (u,v) is a path [u,v].
    #    Merge Operation:
    #    Consider two paths P1 = [v1, ..., vk] and P2 = [v_{k+1}, ..., v_m].
    #    They can be merged into [v1, ..., vk, v_{k+1}, ..., v_m] if:
    #    1. (vk, v_{k+1}) is an edge in E.
    #    2. The edge (vk, v_{k+1}) is NOT currently part of P1 or P2 (except as the connecting link, 
    #       but wait, if it's not part of P1 or P2, it must be part of some OTHER path P3).
    #       If (vk, v_{k+1}) is in P3, we can't just use it.
    #    
    #    This implies that simple merging of two paths requires the connecting edge to be "free", 
    #    which it never is in a partition.
    #    
    #    CORRECTION:
    #    The "connecting edge" is not an extra edge. The paths P1 and P2 are already formed by edges in E.
    #    If P1 ends at u and P2 starts at v, and (u,v) is an edge in E, does that mean we can merge?
    #    Only if the edge (u,v) is NOT already used in P1 or P2.
    #    But since P1 and P2 are simple paths, and (u,v) is an edge, if (u,v) is not in P1 or P2, 
    #    it must be in some other path P3.
    #    So we cannot simply merge P1 and P2 using (u,v) because (u,v) is "busy".
    #    
    #    HOWEVER, if (u,v) IS in P1 or P2, that creates a cycle or repeats a vertex, which is not allowed 
    #    unless it's the very first/last edge.
    #    
    #    Let's reconsider the "all edges as paths" start.
    #    Path 1: [u, v] (edge (u,v))
    #    Path 2: [v, w] (edge (v,w))
    #    Can we merge Path 1 and Path 2?
    #    Path 1 ends at v. Path 2 starts at v.
    #    The connection is at vertex v. We don't need a new edge (v,v). We just join them at v.
    #    Result: [u, v, w]. Edges used: (u,v) and (v,w). Both are in E.
    #    This is valid!
    #    
    #    So the merge condition is:
    #    P1 = [..., u] and P2 = [v, ...]
    #    We can merge them into [..., u, v, ...] IF:
    #    1. (u, v) is an edge in E.
    #    2. The edge (u, v) is NOT already in P1 or P2.
    #       Wait. If P1 is [u, v], the edge (u,v) is in P1.
    #       If we merge [u,v] and [v,w], we get [u,v,w].
    #       The edge (u,v) is in the new path. The edge (v,w) is in the new path.
    #       Is the edge (u,v) "used twice"? No, it's in P1, and now it's in the merged path.
    #       The key is: The set of edges in the merged path is the union of edges in P1 and P2.
    #       For this to be a valid simple path, the vertices must be distinct.
    #       In [u, v, w], u, v, w are distinct.
    #       The condition "edge (u,v) is not in P1 or P2" is WRONG.
    #       The condition is: The edge (u,v) must exist in E.
    #       And the vertices in P1 and P2 must be disjoint (except for the connection point u=v? No, u and v are distinct vertices, the edge connects them).
    #       Wait, P1 ends at u. P2 starts at v.
    #       If u == v, then P1=[..., u] and P2=[u, ...]. We merge at u.
    #       The new path is [..., u, ...]. The edge connecting the two parts is... there is no edge "between" u and u.
    #       The edges are those of P1 and those of P2.
    #       So if u == v, we can always merge?
    #       Example: P1=[a,b], P2=[b,c]. u=b, v=b. Merge -> [a,b,c].
    #       Edges: (a,b), (b,c). Valid.
    #       
    #       If u != v:
    #       P1=[..., u], P2=[v, ...].
    #       We want to form [..., u, v, ...].
    #       This requires the edge (u,v) to be in E.
    #       AND the edge (u,v) must NOT be in P1 or P2?
    #       If (u,v) is in P1, say P1=[u,v, ...], then P1 ends at some x, not u.
    #       So if P1 ends at u, the edge (u,v) cannot be in P1 unless P1 is just [u,v] (length 1) or [v,u] (length 1, but ends at u? No, [v,u] ends at u, starts at v).
    #       If P1=[v,u], it ends at u. Edge (u,v) is in P1.
    #       If P2=[v,w], starts at v.
    #       Can we merge [v,u] and [v,w]?
    #       P1 ends at u. P2 starts at v. u != v.
    #       We want [..., u, v, ...].
    #       New path: [v, u, v, w]? No, v is repeated. Not simple.
    #       So we cannot merge if it creates a repeated vertex.
    #       
    #    So, Merge Rule:
    #    Given P1 (list of vertices) and P2 (list of vertices).
    #    Let u = P1[-1], v = P2[0].
    #    If u == v:
    #       Merge P1 and P2 (remove the duplicate v) -> P1 + P2[1:].
    #       Check simplicity: All vertices in P1 + P2[1:] must be unique.
    #    If u != v:
    #       Check if (u,v) is in E.
    #       If yes, form NewP = P1 + P2.
    #       Check simplicity: All vertices in NewP must be unique.
    #       ALSO, the edge (u,v) must be "available". 
    #       Since we are partitioning E, the edge (u,v) is currently in some path P3.
    #       If P3 is P1 or P2, we have a conflict?
    #       If (u,v) is in P1, then P1 contains u and v. Since P1 is a path, it's either ...u,v... or ...v,u...
    #       If P1 ends at u, and contains v, it must be [v, ..., u] or [u, v] (if length 1).
    #       If P1 is [v, ..., u], and we merge with P2 starting with v, we get [v, ..., u, v, ...] -> v repeated. Invalid.
    #       So the simplicity check handles this.
    #       What if (u,v) is in a third path P3?
    #       Then (u,v) is "used" by P3. We cannot use it again in the merged path.
    #       So we CANNOT merge P1 and P2 using edge (u,v) if (u,v) is not in P1 or P2.
    #       
    #    Conclusion: We can merge P1 and P2 if:
    #    1. u == v (P1[-1] == P2[0]) AND vertices are disjoint (except u).
    #    2. u != v AND (u,v) in E AND (u,v) is in P1 OR (u,v) is in P2?
    #       No, if (u,v) is in P1, P1 has u and v. If P1 ends at u, P1 is [v,...,u] or [u,v].
    #       If P1 is [u,v], it ends at v, not u.
    #       So if P1 ends at u, and (u,v) is in P1, P1 must be [v, u] (length 1) or [v, ..., u].
    #       If P1 is [v, u], and P2 starts with v, merge -> [v, u, v, ...] invalid.
    #       So effectively, the only valid merge for u != v is if we DON'T use the edge (u,v) from the pool?
    #       Wait. The paths ARE the edges.
    #       If I merge P1 and P2, the new path contains all edges of P1 and all edges of P2.
    #       It does NOT contain the edge (u,v) unless (u,v) was already in P1 or P2.
    #       But the new path has the sequence ...u, v... which implies the edge (u,v) is traversed.
    #       If (u,v) was not in P1 or P2, it's in P3. Now it's in the merged path AND in P3. Conflict!
    #       
    #    THEREFORE:
    #    We can ONLY merge P1 and P2 if the connection is at a shared vertex (u==v).
    #    Because if u!=v, we need the edge (u,v), which is already assigned to some path.
    #    
    #    So the local search is:
    #    Find P1, P2 such that P1[-1] == P2[0] (or P1[0] == P2[-1], etc.) and the vertex sets are disjoint (except the join).
    #    Merge them.
    #    
    #    This is essentially finding a path cover where we maximize the number of such "joins".
    #    This is equivalent to finding a maximum set of vertex-disjoint paths? No.
    #    It's equivalent to: In the graph of "paths" (where nodes are paths, edge if they can be joined), find a maximum matching? No, it's a set packing problem.
    
    # Algorithm:
    # 1. Initialize paths = [[u, v] for u, v in edges]
    # 2. While time:
    #    - Pick a random path P1.
    #    - Find all paths P2 that can be joined to P1 (at either end).
    #    - If any, join the one that results in the "best" path (e.g., longest, or random).
    #    - Update paths list.
    
    # To make it faster, we can index paths by their endpoints.
    
    start_time = time.time()
    paths = [[u, v] for u, v in edges]
    # Index: endpoint -> list of indices of paths starting/ending there
    # Since paths can be reversed, we treat them as undirected for joining purposes, 
    # but the order matters for the vertex list.
    
    # We will maintain a list of paths.
    # To find joinable partners, we can just iterate or use a dict.
    # Given small n (20-30), iterating is fine.
    
    while time.time() - start_time < deadline:
        # Try to find a merge
        merged = False
        # Randomize the order of paths to avoid bias
        indices = list(range(len(paths)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(paths): continue
            p1 = paths[i]
            # Try to join p1 with another path p2
            for j in indices:
                if j == i or j >= len(paths): continue
                p2 = paths[j]
                
                # Check if p1 and p2 can be joined
                # Case 1: p1_end == p2_start
                if p1[-1] == p2[0]:
                    new_p = p1 + p2[1:]
                    if len(set(new_p)) == len(new_p):
                        # Valid merge
                        paths.pop(j) # Remove p2 (j > i? not necessarily, but we manage indices carefully)
                        # Actually, popping changes indices. Let's collect to remove.
                        # Better: Mark for removal.
                        pass # Logic below handles this properly
                
                # Case 2: p1_end == p2_end (join p1 with reversed p2)
                if p1[-1] == p2[-1]:
                    new_p = p1 + p2[-2::-1]
                    if len(set(new_p)) == len(new_p):
                        pass # Valid merge
                
                # Case 3: p1_start == p2_start
                if p1[0] == p2[0]:
                    new_p = p1[::-1] + p2[1:]
                    if len(set(new_p)) == len(new_p):
                        pass # Valid merge
                        
                # Case 4: p1_start == p2_end
                if p1[0] == p2[-1]:
                    new_p = p1[::-1] + p2
                    if len(set(new_p)) == len(new_p):
                        pass # Valid merge

        # The above loop is inefficient and buggy with index management.
        # Let's implement a clean "find any valid merge" step.
        
        found_merge = False
        # Shuffle to randomize
        path_indices = list(range(len(paths)))
        random.shuffle(path_indices)
        
        for i_idx in path_indices:
            if i_idx >= len(paths): continue
            p1 = paths[i_idx]
            
            for j_idx in path_indices:
                if j_idx == i_idx or j_idx >= len(paths): continue
                p2 = paths[j_idx]
                
                candidates = []
                if p1[-1] == p2[0]:
                    candidates.append(p1 + p2[1:])
                if p1[-1] == p2[-1]:
                    candidates.append(p1 + p2[-2::-1])
                if p1[0] == p2[0]:
                    candidates.append(p1[::-1] + p2[1:])
                if p1[0] == p2[-1]:
                    candidates.append(p1[::-1] + p2)
                
                for new_p in candidates:
                    if len(set(new_p)) == len(new_p):
                        # Valid merge found
                        # Remove p1 and p2, add new_p
                        # Be careful with indices
                        if i_idx < j_idx:
                            paths.pop(j_idx)
                            paths.pop(i_idx)
                        else:
                            paths.pop(i_idx)
                            paths.pop(j_idx)
                        paths.append(new_p)
                        found_merge = True
                        break
                if found_merge:
                    break
            if found_merge:
                break
        
        if not found_merge:
            # No more simple merges possible.
            # Try a "swap" or "restructure" to enable new merges?
            # Or just restart with a different random initialization?
            # For now, if no merge, we are stuck in a local optimum.
            # We can try to break a long path into two and see if they merge differently.
            # This is complex. Given the time limit, we might just accept this.
            # Or, we can try a different initial randomization.
            # Let's try to randomly reverse a path to see if it enables a merge.
            # Reversing doesn't change the set of edges, but changes the endpoints.
            # If we reverse a path, its start and end swap.
            # This might enable a merge that wasn't possible before.
            
            if paths:
                idx = random.randint(0, len(paths)-1)
                paths[idx] = paths[idx][::-1]
            # Loop continues to try merging with the new configuration.
            # To prevent infinite loop of just reversing, we can count reversals.
            # But reversing is cheap.
            # If we reverse and still no merge, we might cycle.
            # Let's add a counter.
            
            # Actually, if no merge is found, the graph of paths is "maximally joined" 
            # given the current orientations.
            # Reorienting might help.
            # We'll just let it keep trying. If it cycles, it will waste time.
            # To mitigate, we can shuffle the entire list of paths.
            random.shuffle(paths)

    return paths

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    total_deadline = 150.0
    time_per_instance = total_deadline / len(specs)
    
    start_total = time.time()
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        # Allow slightly more time for earlier instances if they finish fast?
        # No, just allocate evenly.
        current_time = time.time()
        elapsed = current_time - start_total
        remaining = total_deadline - elapsed - 5.0 # 5s safety margin
        
        if remaining <= 0:
            # Fallback: return remaining as single edges
            results.append([[u, v] for u, v in edges])
            continue
            
        # Use half the remaining time for this instance to be safe, or just time_per_instance?
        # If we use time_per_instance, we might run over if one is slow.
        # Better: use min(time_per_instance, remaining/2) ?
        # Let's just use a fixed budget per instance, but check global deadline.
        instance_deadline = time.time() + min(time_per_instance, remaining)
        
        paths = decompose_graph(edges, instance_deadline)
        results.append(paths)
        
        # Check if we are running out of time
        if time.time() - start_total > total_deadline - 5.0:
             # If we are close to deadline, fill the rest with single edges
             for spec in specs[len(results):]:
                 edges = instance(*spec)
                 results.append([[u, v] for u, v in edges])
             break

    return results

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
