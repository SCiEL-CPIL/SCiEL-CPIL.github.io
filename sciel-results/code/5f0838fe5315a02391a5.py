import time
import random
from itertools import combinations

def check_constraint(edges, n):
    """
    Check if no simple 3-edge path s-u-v-t has endpoints s, t with equal degree.
    Returns True if valid.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # Iterate over all paths s-u-v-t
    # u-v is an edge
    # s is neighbor of u, s != v
    # t is neighbor of v, t != u, t != s
    # Check deg(s) != deg(t)
    
    for u, v in edges:
        # s in adj[u], t in adj[v]
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                # Path s-u-v-t is simple if s,u,v,t distinct.
                # We ensured s!=v, t!=u, t!=s.
                # Do we need s!=v? Yes, s is neighbor of u. If s==v, path is v-u-v... not simple.
                # Do we need u!=t? Yes, t is neighbor of v. If t==u, path is s-u-v-u... not simple.
                # Do we need s!=t? Yes, checked.
                # Are s and v distinct? Yes.
                # Are u and t distinct? Yes.
                # Are s and u distinct? s is neighbor of u, no loops, so yes.
                # Are v and t distinct? t is neighbor of v, no loops, so yes.
                # What about s==v? Handled.
                # What about t==u? Handled.
                # What about s==t? Handled.
                # Is it possible that s==v? No, s!=v.
                # Is it possible that u==t? No, t!=u.
                # Is it possible that s==u? No.
                # Is it possible that v==t? No.
                # So s,u,v,t are distinct.
                
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    n = 15
    
    # Initialize with K_{7,8}
    # Vertices 0..6 in A, 7..14 in B
    edges = []
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    for i in range(7):
        for j in range(7, 15):
            edges.append((i, j))
            adj[i].add(j)
            adj[j].add(i)
            deg[i] += 1
            deg[j] += 1
            
    # Verify initial validity (should be true)
    # K_{7,8} is valid because deg(A)=8, deg(B)=7.
    # Path s-u-v-t.
    # If s,t in A: u,v in B. No edges in B. Impossible.
    # If s,t in B: u,v in A. No edges in A. Impossible.
    # If s in A, t in B: deg(s)=8, deg(t)=7. 8!=7. OK.
    # If s in B, t in A: deg(s)=7, deg(t)=8. 7!=8. OK.
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Local Search: Hill Climbing
    # We will try to add edges, then swap edges.
    
    def try_add_edges(current_edges, current_adj, current_deg):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if v in current_adj[u]:
                        continue
                    
                    # Check if adding (u,v) is valid
                    # Temporarily add
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_deg[u] += 1
                    current_deg[v] += 1
                    current_edges.append((u, v))
                    
                    if check_constraint(current_edges, n):
                        improved = True
                    else:
                        # Revert
                        current_edges.pop()
                        current_deg[u] -= 1
                        current_deg[v] -= 1
                        current_adj[u].remove(v)
                        current_adj[v].remove(u)
                        
        return current_edges, current_adj, current_deg

    def try_swaps(current_edges, current_adj, current_deg):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try swapping one edge for another
            # This is expensive: O(E * (N^2-E))
            # E ~ 56, N^2-E ~ 50. 2800 checks. Each check is O(N^3) roughly.
            # 15^3 = 3375. 2800 * 3375 ~ 9.4M ops. Should be fast in Python if optimized.
            
            # To speed up, we can just iterate.
            for i in range(len(current_edges)):
                if time.time() > deadline - 5:
                    return current_edges, current_adj, current_deg, False
                e1 = current_edges[i]
                # Try removing e1
                u1, v1 = e1
                current_adj[u1].remove(v1)
                current_adj[v1].remove(u1)
                current_deg[u1] -= 1
                current_deg[v1] -= 1
                current_edges.pop(i)
                
                # Try adding any missing edge
                found_swap = False
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if v2 in current_adj[u2]:
                            continue
                        # Also skip if (u2,v2) is (u1,v1)
                        if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                            continue
                        
                        # Add e2
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                        current_deg[u2] += 1
                        current_deg[v2] += 1
                        current_edges.append((u2, v2))
                        
                        if check_constraint(current_edges, n):
                            improved = True
                            found_swap = True
                            break
                        else:
                            # Revert e2
                            current_edges.pop()
                            current_deg[u2] -= 1
                            current_deg[v2] -= 1
                            current_adj[u2].remove(v2)
                            current_adj[v2].remove(u2)
                            
                    if found_swap:
                        break
                
                if not found_swap:
                    # Revert e1
                    current_edges.append(e1)
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                    current_deg[u1] += 1
                    current_deg[v1] += 1
                    
        return current_edges, current_adj, current_deg, improved

    # Run hill climbing
    edges, adj, deg = try_add_edges(edges, adj, deg)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = list(edges)
        
    edges, adj, deg, _ = try_swaps(edges, adj, deg)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = list(edges)
        
    # Try random restarts if time permits
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 10 and attempts < 50:
        attempts += 1
        # Start from K_{7,8} again, but maybe perturb
        # Or just re-run greedy from scratch with different order?
        # Since K_{7,8} is a very strong local optimum, random restarts from scratch might be worse.
        # Better to perturb the current best.
        
        # Perturb best_edges: remove k random edges
        perturbed = list(best_edges)
        if len(perturbed) > 10:
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        p_adj = [set() for _ in range(n)]
        p_deg = [0] * n
        for u, v in perturbed:
            p_adj[u].add(v)
            p_adj[v].add(u)
            p_deg[u] += 1
            p_deg[v] += 1
            
        # Run hill climbing on perturbed
        p_edges, p_adj, p_deg = try_add_edges(perturbed, p_adj, p_deg)
        if len(p_edges) > best_count:
            best_count = len(p_edges)
            best_edges = list(p_edges)
            
        # Try swaps on perturbed
        p_edges, p_adj, p_deg, _ = try_swaps(p_edges, p_adj, p_deg)
        if len(p_edges) > best_count:
            best_count = len(p_edges)
            best_edges = list(p_edges)

    # Final result formatting
    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
            
    return result
