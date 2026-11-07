import time
import random

N = 15
DEADLINE = 50.0  # safety margin under the 60 s wall-clock limit

def _can_add(adj, deg, u, v):
    """
    Check if adding edge (u, v) to a valid graph (represented by adj, deg)
    keeps the graph valid.
    adj: list of sets of neighbors.
    deg: list of degrees.
    u, v: vertices to connect. Assumes u != v and (u, v) not in adj.
    """
    # New paths are s-u-v-t where s in N(u)\{v}, t in N(v)\{u}
    # We need deg[s] != deg[t]
    # Note: deg[u] and deg[v] will increase, but s and t are not u or v
    # (since s in N(u) and s!=v, t in N(v) and t!=u, and s!=t implies s!=u? 
    # No, s is neighbor of u, so s!=u. t is neighbor of v, so t!=v.
    # s could be v? No, s in N(u)\{v}.
    # t could be u? No, t in N(v)\{u}.
    # So s and t are distinct from u and v.
    # Thus deg[s] and deg[t] remain unchanged by the addition of (u, v).
    
    nu = adj[u]
    nv = adj[v]
    
    for s in nu:
        if s == v:
            continue
        ds = deg[s]
        for t in nv:
            if t == u or t == s:
                continue
            if ds == deg[t]:
                return False
    return True

def _valid_full(adj, deg):
    """
    Full validity check for a graph.
    Used to verify the final result or initial states.
    """
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _greedy_fast(adj, deg, order):
    """
    Greedy add edges in given order.
    adj, deg are modified in place.
    Returns number of edges added.
    """
    added = 0
    for u, v in order:
        if u == v:
            continue
        if v in adj[u]:
            continue
        if _can_add(adj, deg, u, v):
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            added += 1
    return added

def _local_search_swap(adj, deg, all_edges, rng, max_iters=100):
    """
    Local search: try to swap an existing edge with a missing edge.
    Returns improved adj, deg or None if no improvement.
    """
    current_edges = set()
    for u in range(N):
        for v in adj[u]:
            if u < v:
                current_edges.add((u, v))
    
    missing_edges = [e for e in all_edges if e not in current_edges]
    
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    best_count = len(current_edges)
    
    for _ in range(max_iters):
        if not missing_edges or not current_edges:
            break
            
        # Pick a random missing edge to add
        u, v = rng.choice(missing_edges)
        
        # Find conflicts: pairs (s, t) such that s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]==deg[t]
        # To resolve, we can try removing an edge incident to s, t, u, or v.
        # Simpler: Try removing ANY existing edge and see if (u, v) can be added.
        # This is expensive. 
        # Better: Identify specific edges that "cause" the conflict.
        # If deg[s] == deg[t], and we remove an edge incident to s, deg[s] changes.
        # If we remove an edge incident to t, deg[t] changes.
        # If we remove an edge incident to u, s might drop out of N(u).
        # If we remove an edge incident to v, t might drop out of N(v).
        
        # Heuristic: Try removing one edge from the set of edges incident to u, v, s, or t.
        candidates_to_remove = set()
        for s in adj[u]:
            if s == v: continue
            for t in adj[v]:
                if t == u or t == s: continue
                if deg[s] == deg[t]:
                    # Conflict found. Add edges incident to s, t, u, v to candidates
                    candidates_to_remove.update(adj[s])
                    candidates_to_remove.update(adj[t])
                    candidates_to_remove.update(adj[u])
                    candidates_to_remove.update(adj[v])
        
        if not candidates_to_remove:
            # Should not happen if missing edge is invalid, but just in case
            continue
            
        # Try removing a random candidate
        # Note: candidate is a vertex. We need an edge.
        # Let's pick a random vertex from candidates, then a random neighbor.
        # Actually, let's just iterate through existing edges and try swapping.
        # To keep it fast, limit the number of swaps tried.
        
        improved = False
        # Try a few random swaps
        for _ in range(10):
            if not current_edges:
                break
            a, b = rng.choice(list(current_edges))
            
            # Try swap: remove (a, b), add (u, v)
            # 1. Remove (a, b)
            adj[a].remove(b)
            adj[b].remove(a)
            deg[a] -= 1
            deg[b] -= 1
            current_edges.remove((a, b))
            
            # 2. Try add (u, v)
            if _can_add(adj, deg, u, v):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                current_edges.add((u, v))
                missing_edges.remove((u, v))
                
                # Check if we improved edge count (it should be same, but maybe we can add more later?
                # No, swap keeps count same. 
                # However, the new configuration might allow adding MORE edges later.
                # So we accept the swap if it doesn't reduce count (it doesn't) 
                # and we hope it opens up more possibilities.
                # But we need a criterion to stop.
                # Let's accept the swap and continue.
                improved = True
                break
            else:
                # Revert
                adj[u].discard(v)
                adj[v].discard(u)
                # deg[u], deg[v] were not incremented in failed check
                # But we need to revert the removal of (a, b)
                adj[a].add(b)
                adj[b].add(a)
                deg[a] += 1
                deg[b] += 1
                current_edges.add((a, b))
        
        if not improved:
            # Try adding a different missing edge? Or break?
            # Let's try to just add a missing edge if possible without swap
            # (Already handled by greedy, but maybe after swaps?)
            pass
            
    return best_adj, best_deg, best_count

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)

    best_adj = [set() for _ in range(N)]
    best_deg = [0] * N
    best_count = 0

    def consider(adj, deg, count):
        nonlocal best_adj, best_deg, best_count
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
            best_deg = list(deg)

    # Phase 1: Randomized Greedy
    while time.time() - t0 < DEADLINE:
        order = all_edges[:]
        rng.shuffle(order)
        
        adj = [set() for _ in range(N)]
        deg = [0] * N
        
        # Start with empty graph
        added = _greedy_fast(adj, deg, order)
        consider(adj, deg, added)
        
        # Second pass with new order
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            added2 = _greedy_fast(adj, deg, order2)
            consider(adj, deg, added2)

    # Phase 2: Local Search on Best
    if time.time() - t0 < DEADLINE:
        # Copy best
        cur_adj = [set(s) for s in best_adj]
        cur_deg = list(best_deg)
        cur_count = best_count
        
        # Try to improve
        for _ in range(100): # Limit local search iterations
            if time.time() - t0 > DEADLINE:
                break
            
            # Try to add any missing edge
            improved = False
            for u, v in all_edges:
                if v in cur_adj[u]:
                    continue
                if _can_add(cur_adj, cur_deg, u, v):
                    cur_adj[u].add(v)
                    cur_adj[v].add(u)
                    cur_deg[u] += 1
                    cur_deg[v] += 1
                    cur_count += 1
                    improved = True
                    # Restart search from here? Or continue?
                    # Continue might be better to pack more.
            
            if not improved:
                # Try swaps
                # Identify missing edges
                current_edges = set()
                for u in range(N):
                    for v in cur_adj[u]:
                        if u < v:
                            current_edges.add((u, v))
                missing_edges = [e for e in all_edges if e not in current_edges]
                
                if not missing_edges:
                    break
                
                swap_done = False
                for _ in range(20): # Try few swaps
                    if not missing_edges or not current_edges:
                        break
                    u, v = rng.choice(missing_edges)
                    a, b = rng.choice(list(current_edges))
                    
                    # Save state
                    old_adj_u = set(cur_adj[u])
                    old_adj_v = set(cur_adj[v])
                    old_adj_a = set(cur_adj[a])
                    old_adj_b = set(cur_adj[b])
                    old_deg = list(cur_deg)
                    
                    # Remove (a, b)
                    cur_adj[a].remove(b)
                    cur_adj[b].remove(a)
                    cur_deg[a] -= 1
                    cur_deg[b] -= 1
                    current_edges.remove((a, b))
                    
                    # Try add (u, v)
                    if _can_add(cur_adj, cur_deg, u, v):
                        cur_adj[u].add(v)
                        cur_adj[v].add(u)
                        cur_deg[u] += 1
                        cur_deg[v] += 1
                        current_edges.add((u, v))
                        missing_edges.remove((u, v))
                        
                        # Accept swap. Count is same, but structure changed.
                        # Check if this allows adding more edges immediately?
                        # The loop above "Try to add any missing edge" will handle that in next iteration.
                        swap_done = True
                        improved = True
                        break
                    else:
                        # Revert
                        cur_adj[u] = old_adj_u
                        cur_adj[v] = old_adj_v
                        cur_adj[a] = old_adj_a
                        cur_adj[b] = old_adj_b
                        cur_deg = old_deg
                        current_edges.add((a, b))
                
                if not swap_done:
                    break
            
            consider(cur_adj, cur_deg, cur_count)

    # Final verification
    if not _valid_full(best_adj, best_deg):
        # Fallback to star if something went wrong
        star_adj = [set() for _ in range(N)]
        star_deg = [0] * N
        for v in range(1, N):
            star_adj[0].add(v)
            star_adj[v].add(0)
            star_deg[0] += 1
            star_deg[v] += 1
        best_adj = star_adj
        best_deg = star_deg
        best_count = N - 1

    result = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                result.append([int(u), int(v)])
    return result
