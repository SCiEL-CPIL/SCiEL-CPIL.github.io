import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))

    def bit(i):
        return 1 << i

    def popcount(x):
        return x.bit_count()

    def edge_count(adj):
        return sum(m.bit_count() for m in adj) // 2

    def mask_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges

    def is_k33_free(adj):
        for u, v, w in triples:
            if popcount(adj[u] & adj[v] & adj[w]) > 2:
                return False
        return True

    def can_add(u, v, adj):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj[u] | bit_v
        new_v = adj[v] | bit_u
        
        # Precompute intersection of new_u and new_v for triples containing both
        # But we can just iterate. Optimization:
        # A triple (a,b,c) is affected if u in {a,b,c} or v in {a,b,c}.
        
        for a, b, c in triples:
            # Skip if neither u nor v is in the triple
            if u not in (a, b, c) and v not in (a, b, c):
                continue
            
            ma = adj[a]
            mb = adj[b]
            mc = adj[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        return True

    def greedy_add(adj):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & bit(j)):
                        if can_add(i, j, adj):
                            adj[i] |= bit(j)
                            adj[j] |= bit(i)
                            changed = True
        return adj

    def repair(adj):
        """Remove edges to fix violations."""
        # Fast check first
        if is_k33_free(adj):
            return adj
            
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_k33_free(adj):
                break
            
            # Find a violating triple
            viol = None
            for u, v, w in triples:
                common = adj[u] & adj[v] & adj[w]
                if popcount(common) > 2:
                    viol = (u, v, w, common)
                    break
            
            if viol is None:
                break
            
            u, v, w, common = viol
            # Remove a random edge incident to a common neighbor
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                # Try removing (u,c), (v,c), or (w,c)
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj[a] & bit(b):
                        adj[a] &= ~bit(b)
                        adj[b] &= ~bit(a)
                        if is_k33_free(adj):
                            removed = True
                            break
                        else:
                            # Restore
                            adj[a] |= bit(b)
                            adj[b] |= bit(a)
                if removed:
                    break
            if not removed:
                # Force remove one
                if common_list:
                    c = common_list[0]
                    a, b = u, c
                    adj[a] &= ~bit(b)
                    adj[b] &= ~bit(a)
        return adj

    def local_search(adj, time_limit):
        """
        Local search:
        1. Repair
        2. Greedy Add
        3. Hill Climbing with Swaps / Moves
        """
        best_adj = list(adj)
        best_count = edge_count(best_adj)
        cur_adj = list(adj)
        cur_count = edge_count(cur_adj)
        
        t_end = time.time() + time_limit
        
        while time.time() < t_end:
            # Repair if invalid
            if not is_k33_free(cur_adj):
                cur_adj = repair(cur_adj)
                cur_count = edge_count(cur_adj)
                
            # Greedy add
            cur_adj = greedy_add(cur_adj)
            cur_count = edge_count(cur_adj)
            
            if cur_count > best_count:
                best_count = cur_count
                best_adj = list(cur_adj)
                
            # Try to find a move
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (cur_adj[i] & bit(j))]
            if not missing:
                break
                
            random.shuffle(missing)
            improved = False
            
            # Try adding a single missing edge
            for u, v in missing:
                if time.time() > t_end:
                    break
                if can_add(u, v, cur_adj):
                    cur_adj[u] |= bit(v)
                    cur_adj[v] |= bit(u)
                    cur_count += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no direct addition works, try a swap
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if cur_adj[i] & bit(j)]
            if not all_edges:
                break
                
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try a few swaps
            for i in range(min(20, len(all_edges), len(missing))):
                if time.time() > t_end:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove r_u-r_v
                cur_adj[r_u] &= ~bit(r_v)
                cur_adj[r_v] &= ~bit(r_u)
                # Add a_u-a_v
                cur_adj[a_u] |= bit(a_v)
                cur_adj[a_v] |= bit(a_u)
                
                if is_k33_free(cur_adj):
                    # Check if this swap allows adding more edges
                    cur_adj = greedy_add(cur_adj)
                    cur_count = edge_count(cur_adj)
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(cur_adj)
                    improved = True
                    break
                else:
                    # Revert
                    cur_adj[r_u] |= bit(r_v)
                    cur_adj[r_v] |= bit(r_u)
                    cur_adj[a_u] &= ~bit(a_v)
                    cur_adj[a_v] &= ~bit(a_u)
            
            if not improved:
                # Escape local optimum: remove a random edge and greedy add
                if all_edges:
                    r_u, r_v = all_edges[0]
                    cur_adj[r_u] &= ~bit(r_v)
                    cur_adj[r_v] &= ~bit(r_u)
                    cur_adj = greedy_add(cur_adj)
                    cur_count = edge_count(cur_adj)
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = list(cur_adj)
        
        return best_adj, best_count

    best_adj = None
    best_count = -1
    rng = random.Random(42)

    # Multiple restarts
    # Use a mix of random densities and structured starts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        density = 0.3 + rng.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and Optimize
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        # Shorter time for later restarts
        ls_time = min(time_left * 0.8, 5.0)
        adj, cnt = local_search(adj, ls_time)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
            
        # If we have a very good result, do a longer polish
        if best_count > 150 and restart % 5 == 0 and time.time() < deadline - 10:
            polish_time = min(deadline - time.time() - 5, 10.0)
            adj2, cnt2 = local_search(best_adj, polish_time)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = list(adj2)

    if best_adj is None:
        # Fallback: Empty graph
        best_adj = [0] * n
        best_count = 0

    # Final polish with remaining time
    if time.time() < deadline - 2:
        final_time = deadline - time.time() - 2
        if final_time > 1:
            adj_final, cnt_final = local_search(best_adj, final_time)
            if cnt_final > best_count:
                best_count = cnt_final
                best_adj = list(adj_final)

    edges = mask_to_edges(best_adj)
    return edges

