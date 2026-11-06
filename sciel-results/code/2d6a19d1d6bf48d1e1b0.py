"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    
    # Fast adjacency using integer bitmasks
    # adj[i] is an integer where bit j is set if edge (i,j) exists
    
    def make_adj(n):
        return [0] * n
    
    def has_edge(adj, n, u, v):
        if u > v:
            u, v = v, u
        return (adj[u] >> v) & 1
    
    def set_edge(adj, n, u, v, val):
        if u > v:
            u, v = v, u
        if val:
            adj[u] |= (1 << v)
        else:
            adj[u] &= ~(1 << v)
    
    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def find_k4(adj, n):
        """Find one K4 quickly. Returns (i,j,k,l) or None."""
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j+1, n):
                    if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                        continue
                    for l in range(k+1, n):
                        if ((adj[i] >> l) & 1 and (adj[j] >> l) & 1 and (adj[k] >> l) & 1):
                            return (i, j, k, l)
        return None
    
    def find_indep6(adj, n):
        """Find one independent 6-set. Returns list of 6 vertices or None."""
        def backtrack(remaining, chosen):
            if len(chosen) == 6:
                return chosen
            if len(remaining) < 6 - len(chosen):
                return None
            for idx, v in enumerate(remaining):
                new_rem = []
                for u in remaining[idx+1:]:
                    if not ((adj[v] >> u) & 1):
                        new_rem.append(u)
                res = backtrack(new_rem, chosen + [v])
                if res is not None:
                    return res
            return None
        return backtrack(list(range(n)), [])
    
    def is_valid(adj, n):
        return find_k4(adj, n) is None and find_indep6(adj, n) is None
    
    def count_k4_fast(adj, n):
        """Count K4s using bitmask intersection."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                common = adj[i] & adj[j]
                # Only consider vertices > j
                common &= ~((1 << (j+1)) - 1)
                # For each pair (k,l) in common with k < l
                k_list = []
                for k in range(j+1, n):
                    if (common >> k) & 1:
                        k_list.append(k)
                for a in range(len(k_list)):
                    k = k_list[a]
                    common2 = common & adj[k]
                    common2 &= ~((1 << (k+1)) - 1)
                    count += bin(common2).count('1')
        return count
    
    def count_indep6_fast(adj, n):
        """Count independent 6-sets using backtracking with pruning."""
        count = 0
        def backtrack(remaining, chosen_size):
            nonlocal count
            if chosen_size == 6:
                count += 1
                return
            if len(remaining) < 6 - chosen_size:
                return
            for idx, v in enumerate(remaining):
                new_remaining = []
                for u in remaining[idx+1:]:
                    if not ((adj[v] >> u) & 1):
                        new_remaining.append(u)
                backtrack(new_remaining, chosen_size + 1)
        backtrack(list(range(n)), 0)
        return count
    
    def energy(adj, n):
        """Energy = #K4 * 1000 + #indep6. K4s are much worse."""
        k4 = count_k4_fast(adj, n)
        if k4 > 0:
            return k4 * 1000
        indep = count_indep6_fast(adj, n)
        return indep
    
    # ---- Known 35-vertex product graph ----
    def build_product_35():
        n = 35
        D5 = {1, 4}
        D7 = {1, 2, 4}
        adj = make_adj(n)
        for i in range(n):
            a1, b1 = divmod(i, 7)
            for j in range(i+1, n):
                a2, b2 = divmod(j, 7)
                da = (a2 - a1) % 5
                db = (b2 - b1) % 7
                if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                    set_edge(adj, n, i, j, True)
        return adj
    
    # Try the known 35-vertex graph
    adj35 = build_product_35()
    if is_valid(adj35, 35):
        return {'n': 35, 'edges': extract_edges(adj35, 35)}
    
    # ---- Local search with efficient repair ----
    def local_search(adj, n, max_iters, deadline):
        """Local search: find violations, fix them with targeted edge flips."""
        best_adj = adj[:]
        best_energy = energy(adj, n)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            e = energy(adj, n)
            if e == 0:
                return adj, 0
            
            if e < best_energy:
                best_energy = e
                best_adj = adj[:]
            
            # Find and fix a K4 first (higher priority)
            k4 = find_k4(adj, n)
            if k4 is not None:
                i, j, k, l = k4
                # Try removing each edge of the K4 and pick the one that reduces energy most
                best_flip = None
                best_new_e = e
                for u, v in [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]:
                    set_edge(adj, n, u, v, False)
                    new_e = energy(adj, n)
                    set_edge(adj, n, u, v, True)  # restore
                    if new_e < best_new_e:
                        best_new_e = new_e
                        best_flip = (u, v, False)
                if best_flip is not None:
                    u, v, val = best_flip
                    set_edge(adj, n, u, v, val)
                    continue
            
            # Find and fix an independent 6-set
            indep6 = find_indep6(adj, n)
            if indep6 is not None:
                # Try adding each edge within the independent set
                best_flip = None
                best_new_e = e
                for a in range(6):
                    for b in range(a+1, 6):
                        u, v = indep6[a], indep6[b]
                        set_edge(adj, n, u, v, True)
                        new_e = energy(adj, n)
                        set_edge(adj, n, u, v, False)  # restore
                        if new_e < best_new_e:
                            best_new_e = new_e
                            best_flip = (u, v, True)
                if best_flip is not None:
                    u, v, val = best_flip
                    set_edge(adj, n, u, v, val)
                    continue
            
            # No improvement possible with single flips; try random perturbation
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            set_edge(adj, n, u, v, not has_edge(adj, n, u, v))
        
        return best_adj, best_energy
    
    # ---- Try to find valid graphs at various sizes ----
    best_result = None
    best_n = 0
    
    for target_n in [35, 34, 33, 32, 31, 30, 29, 28]:
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple restarts
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            if found:
                break
            
            if target_n == 35 and restart == 0:
                adj = build_product_35()
            else:
                # Random initialization with density tuned for K4-free
                adj = make_adj(target_n)
                density = 0.42
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < density:
                            set_edge(adj, target_n, i, j, True)
            
            # Local search
            adj, e = local_search(adj, target_n, 2000, deadline - 15)
            
            if e == 0:
                found = True
                if target_n > best_n:
                    best_n = target_n
                    best_result = {'n': target_n, 'edges': extract_edges(adj, target_n)}
        
        if found and target_n == 35:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the best we can find
    # Try a few more random attempts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        n = 30
        adj = make_adj(n)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    set_edge(adj, n, i, j, True)
        adj, e = local_search(adj, n, 1000, deadline - 5)
        if e == 0 and n > best_n:
            best_n = n
            best_result = {'n': n, 'edges': extract_edges(adj, n)}
    
    if best_result is not None:
        return best_result
    
    # Last resort: small valid graph
    # C5 is K4-free and has alpha=2
    adj5 = make_adj(5)
    for i in range(5):
        set_edge(adj5, 5, i, (i+1)%5, True)
    return {'n': 5, 'edges': extract_edges(adj5, 5)}
# EVOLVE-BLOCK-END
