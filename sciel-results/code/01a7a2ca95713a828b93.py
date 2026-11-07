import random
import time
import sys

N = 15

def _check_full(adj, deg):
    """
    Full validity check.
    adj: list of sets
    deg: list of ints
    Returns True if valid.
    """
    for u in range(N):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _try_add_edge(adj, deg, a, b):
    """
    Try to add edge (a,b).
    Returns True if valid, and updates adj/deg if so.
    """
    if a == b or b in adj[a]:
        return False
    
    # Check if adding (a,b) creates a violation
    # We simulate the addition
    deg_a_new = deg[a] + 1
    deg_b_new = deg[b] + 1
    
    # 1. Paths where (a,b) is the middle edge: s-a-b-t
    for s in adj[a]:
        if s == b:
            continue
        for t in adj[b]:
            if t == a:
                continue
            if deg[s] == deg[t]:
                return False

    # 2. Paths where a is an endpoint, and the path starts/ends at a
    #    and involves b as the first/last hop? No, we need to check ALL paths ending at a.
    #    The degree of a changes, so any path s-u-v-a must satisfy deg[s] != deg[a]_new.
    #    Also any path a-u-v-t must satisfy deg[a]_new != deg[t].
    
    # Check paths ending at a: s-u-v-a
    for v in adj[a]:
        if v == b:
            continue # (a,b) is not in the old graph, so v != b for old neighbors? 
                     # Wait, we are adding (a,b). So b is NOT in adj[a] yet.
                     # So v in adj[a] are old neighbors.
        for u in adj[v]:
            if u == a:
                continue
            # Path s=u? No, s-u-v-a. s is the other end of u.
            # Path is s-u-v-a. s is in adj[u] \ {v}.
            for s in adj[u]:
                if s == v:
                    continue
                # s, u, v, a are distinct?
                # s != u (no loops), u != v (simple graph), v != a (adjacency).
                # s != v checked.
                # s != a? If s == a, then path is a-u-v-a, not simple (cycle). 
                # But s is in adj[u]. If s==a, then a-u is an edge. 
                # Path a-u-v-a is a cycle of length 3? No, a-u-v-a is 3 edges.
                # The problem says "simple three-edge path". Usually implies distinct vertices.
                # "s != t" is explicitly stated. 
                # "simple path" usually implies distinct vertices.
                # If s == a, then endpoints are a and a? No, s is the start.
                # If s == a, then path is a-u-v-a. Endpoints are a and a. 
                # The constraint says "endpoints of equal degree". 
                # If s == t, the condition "s != t" in the prompt "s != t" applies.
                # "Reject if any simple three-edge path has endpoints of equal degree... s != t"
                # So if s == t, we don't care.
                if s == a:
                    continue
                if deg[s] == deg_a_new:
                    return False

    # Check paths starting at a: a-u-v-t
    for u in adj[a]:
        if u == b:
            continue
        for v in adj[u]:
            if v == a:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if t == a:
                    continue
                if deg_a_new == deg[t]:
                    return False

    # 3. Same for b
    # Paths ending at b: s-u-v-b
    for v in adj[b]:
        if v == a:
            continue
        for u in adj[v]:
            if u == b:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                if s == b:
                    continue
                if deg[s] == deg_b_new:
                    return False

    # Paths starting at b: b-u-v-t
    for u in adj[b]:
        if u == a:
            continue
        for v in adj[u]:
            if v == b:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if t == b:
                    continue
                if deg_b_new == deg[t]:
                    return False

    # 4. Paths where a is endpoint and b is NOT the first hop?
    #    Wait, the logic above "Paths ending at a" iterated over ALL neighbors v of a.
    #    It did NOT exclude b. But b is not in adj[a] yet.
    #    So it checked paths using OLD edges.
    #    What about paths that USE the new edge (a,b) and have a as endpoint?
    #    Path a-b-u-t. 
    #    This is covered in "Paths starting at a" if we included b in adj[a]. 
    #    But we didn't.
    #    So we need to explicitly check paths involving the new edge (a,b) where a or b is an endpoint.
    
    # Path a-b-u-t:
    for u in adj[b]:
        if u == a:
            continue
        for t in adj[u]:
            if t == b:
                continue
            if t == a:
                continue
            if deg_a_new == deg[t]:
                return False

    # Path b-a-u-t:
    for u in adj[a]:
        if u == b:
            continue
        for t in adj[u]:
            if t == a:
                continue
            if t == b:
                continue
            if deg_b_new == deg[t]:
                return False

    # If all checks pass, add the edge
    adj[a].add(b)
    adj[b].add(a)
    deg[a] = deg_a_new
    deg[b] = deg_b_new
    return True

def _remove_edge(adj, deg, a, b):
    adj[a].discard(b)
    adj[b].discard(a)
    deg[a] -= 1
    deg[b] -= 1

def _local_search(adj, deg, rng, deadline):
    """
    Local search to maximize edges.
    """
    current_edges = sum(len(s) for s in adj) // 2
    best_edges = current_edges
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add missing edges
        missing = [(i, j) for i in range(N) for j in range(i+1, N) if j not in adj[i]]
        rng.shuffle(missing)
        
        for a, b in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, deg, a, b):
                improved = True
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj]
                    best_deg = list(deg)
                    no_improve_count = 0
                # Restart scan if we added an edge? 
                # Or just continue? Restarting is safer to find more adds.
                break
        
        if improved:
            continue
            
        # 2. If no edge can be added, try swaps
        # Pick a random existing edge to remove
        existing = [(i, j) for i in range(N) for j in adj[i] if i < j]
        if not existing:
            break
        rng.shuffle(existing)
        
        swap_found = False
        for ra, rb in existing:
            if time.time() > deadline:
                break
            
            # Try to find a missing edge to add
            # We can sample a few missing edges
            missing_sample = [(i, j) for i in range(N) for j in range(i+1, N) if j not in adj[i] and not (i==ra and j==rb) and not (i==rb and j==ra)]
            rng.shuffle(missing_sample)
            
            for limit in range(20): # Try up to 20 random missing edges
                if not missing_sample:
                    break
                if time.time() > deadline:
                    break
                a, b = missing_sample.pop()
                
                # Simulate swap: remove (ra, rb), add (a, b)
                # Save state
                adj_save = [set(s) for s in adj]
                deg_save = list(deg)
                
                _remove_edge(adj, deg, ra, rb)
                
                if _try_add_edge(adj, deg, a, b):
                    swap_found = True
                    current_edges = sum(len(s) for s in adj) // 2
                    # Edge count stays same (1 removed, 1 added)
                    # But maybe it allows more adds later?
                    # For now, we accept if it's valid.
                    # We don't update best_edges unless we can add more?
                    # No, best_edges is max edges. Swap keeps count same.
                    # So we only accept swap if it leads to a state where we can add more?
                    # That's 2-step lookahead.
                    # Simpler: Accept swap if it doesn't reduce density (it doesn't) 
                    # and we hope it unlocks adds.
                    # But if we just accept random valid swaps, we might wander.
                    # Let's accept the swap.
                    no_improve_count = 0
                    break
                else:
                    # Restore state
                    adj = adj_save
                    deg = deg_save
            
            if swap_found:
                break
            else:
                # No valid swap found for this removed edge
                # We already restored state if _try_add_edge failed?
                # In the loop above, if _try_add_edge fails, we restore.
                # If we break out of inner loop without swap_found, state is restored.
                # So we are good.
                pass
        
        if not swap_found:
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            
    return best_adj, best_deg, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_edges = -1
    best_adj = None
    best_deg = None
    
    # Multi-start
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(seed)
        
        # Initialize with random sparse graph
        adj = [set() for _ in range(N)]
        deg = [0] * N
        
        # Greedy initialization
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(edges)
        
        for a, b in edges:
            if time.time() > deadline - 5.0:
                break
            _try_add_edge(adj, deg, a, b)
            
        current_edges = sum(len(s) for s in adj) // 2
        
        # Local search
        adj2, deg2, edges2 = _local_search(adj, deg, rng, deadline - 2.0)
        
        if edges2 > best_edges:
            best_edges = edges2
            best_adj = adj2
            best_deg = deg2
            
    # Final check
    if best_adj is None:
        best_adj = [set() for _ in range(N)]
        best_deg = [0] * N
        best_edges = 0
        
    # Verify best result
    if not _check_full(best_adj, best_deg):
        # Fallback to a simple valid graph (e.g. empty or star)
        # Star is valid? 
        # Path s-u-v-t. 
        # In star, center c. 
        # Path must go through c. 
        # s-c-u-t? No, u is leaf, t is leaf. 
        # s-c-u is 2 edges. 
        # Path length 3: s-c-u-v? 
        # If u is leaf, it has no other neighbors. 
        # So no path of length 3 exists in a star. 
        # So star is valid.
        best_adj = [set() for _ in range(N)]
        best_deg = [0] * N
        for i in range(1, N):
            best_adj[0].add(i)
            best_adj[i].add(0)
            best_deg[0] += 1
            best_deg[i] += 1
        best_edges = N - 1

    # Convert to list of lists
    result = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                result.append([int(i), int(j)])
                
    return result

if __name__ == "__main__":
    construct()
