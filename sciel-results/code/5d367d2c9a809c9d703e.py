import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences (nonzero quadratic residues mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: adj[i] = set of j such that (i-j) % m in forbidden
    # i.e., j = (i + d) % m for d in forbidden
    adj = [set() for _ in range(m)]
    adj_list = [[] for _ in range(m)]
    for i in range(m):
        for d in forbidden:
            j = (i + d) % m
            if j != i:
                adj[i].add(j)
                adj_list[i].append(j)
    # Make symmetric: if j in adj[i], then i in adj[j]
    # Actually (i-j) mod m in F means (j-i) mod m = -d mod m. 
    # Is -d also in F? For d = x^2, -d = (-x)^2 mod m only if -1 is QR mod m.
    # -1 mod 5 = 4 = 2^2, so -1 is QR mod 5. -1 mod 41: 41 mod 4 = 1, so -1 is QR mod 41.
    # So -1 is QR mod 205, meaning F is symmetric. So adj is symmetric.
    
    # Verify symmetry and build bitmask adjacency for fast operations
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj_list[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute degree
    degree = [len(adj_list[i]) for i in range(m)]
    
    def calc_conflicts(S):
        """Count total conflicts (pairs with forbidden difference) in S."""
        c = 0
        temp = S
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            c += (S & adj_mask[i]).bit_count()
            temp ^= lsb
        return c // 2
    
    def greedy_init(order_seed=None):
        """Greedy construction with given random seed for ordering."""
        if order_seed is not None:
            rng = random.Random(order_seed)
        else:
            rng = random
        S = 0
        nodes = list(range(m))
        rng.shuffle(nodes)
        for n in nodes:
            if (S & adj_mask[n]) == 0:
                S |= (1 << n)
        return S
    
    def local_search(S, max_iters=100000, rng=None):
        """Local search with conflict-aware single flips and pair exchanges."""
        if rng is None:
            rng = random
        
        in_set = [(S >> i) & 1 for i in range(m)]
        # conflict_count[v] = number of neighbors of v that are in S
        conflict_count = [0] * m
        for v in range(m):
            if in_set[v]:
                for u in adj_list[v]:
                    if in_set[u]:
                        conflict_count[v] += 1
        
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try all possible single-bit flips, pick best
            best_delta = 0
            best_move = None
            best_type = 0  # 0=none, 1=add, 2=remove
            
            for v in range(m):
                if in_set[v]:
                    # Removing v: delta_conf = -conflict_count[v], delta_size = -1
                    dc = -conflict_count[v]
                    ds = -1
                    de = -ds + 20 * dc  # energy = -size + penalty*conf
                    if de < best_delta:
                        best_delta = de
                        best_move = v
                        best_type = 2
                else:
                    # Adding v: delta_conf = conflict_count[v], delta_size = +1
                    dc = conflict_count[v]
                    ds = 1
                    de = -ds + 20 * dc
                    if de < best_delta:
                        best_delta = de
                        best_move = v
                        best_type = 1
            
            if best_move is not None:
                v = best_move
                if best_type == 1:  # add
                    in_set[v] = 1
                    S |= (1 << v)
                    current_size += 1
                    current_conf += conflict_count[v]
                    # Update conflict counts for neighbors
                    for u in adj_list[v]:
                        if in_set[u]:
                            conflict_count[u] += 1
                        conflict_count[v] += 0  # v's own count already correct
                    # v's conflict count is conflict_count[v] (neighbors in S)
                    # After adding v, v's conflict count = number of neighbors in S = conflict_count[v] (was already that)
                    # But neighbors' conflict counts increase by 1
                else:  # remove
                    in_set[v] = 0
                    S &= ~(1 << v)
                    current_size -= 1
                    current_conf -= conflict_count[v]
                    # Update conflict counts for neighbors
                    for u in adj_list[v]:
                        if in_set[u]:
                            conflict_count[u] -= 1
                    conflict_count[v] = 0
                improved = True
        
        # Now try pair exchanges: remove one, add one (size stays same, may reduce conflicts)
        improved2 = True
        while improved2:
            improved2 = False
            # Find best pair exchange
            best_delta_conf = 0
            best_remove = -1
            best_add = -1
            
            for v in range(m):
                if not in_set[v]:
                    continue
                for u in range(m):
                    if in_set[u]:
                        continue
                    # Remove v, add u
                    # New conflict for u after removing v:
                    # conflict_count[u] - (1 if v in adj[u] else 0)
                    # Delta conf = (conflict_count[u] - (1 if v in adj[u] else 0)) - conflict_count[v]
                    # Wait, need to be careful.
                    # Before: conf_v = conflict_count[v], conf_u = conflict_count[u]
                    # After removing v: conf_u' = conf_u - (1 if u in adj[v] else 0) = conf_u - (1 if v in adj[u] else 0)
                    # After adding u: new conflicts for u = conf_u'
                    # Delta conf = conf_u' - conf_v
                    is_adj = 1 if (adj_mask[v] >> u) & 1 else 0
                    conf_u_after = conflict_count[u] - is_adj
                    delta_conf = conf_u_after - conflict_count[v]
                    if delta_conf < best_delta_conf:
                        best_delta_conf = delta_conf
                        best_remove = v
                        best_add = u
            
            if best_remove >= 0 and best_add >= 0:
                v = best_remove
                u = best_add
                # Remove v
                in_set[v] = 0
                S &= ~(1 << v)
                current_conf -= conflict_count[v]
                for w in adj_list[v]:
                    if in_set[w]:
                        conflict_count[w] -= 1
                conflict_count[v] = 0
                # Add u
                in_set[u] = 1
                S |= (1 << u)
                current_conf += conflict_count[u]
                for w in adj_list[u]:
                    if in_set[w]:
                        conflict_count[w] += 1
                improved2 = True
        
        return S, current_size, current_conf
    
    def perturb(S, strength=3):
        """Perturb by randomly removing and adding nodes."""
        nodes_in = [i for i in range(m) if (S >> i) & 1]
        nodes_out = [i for i in range(m) if not ((S >> i) & 1)]
        
        S_new = S
        # Remove some random nodes
        k_remove = min(strength, len(nodes_in))
        for _ in range(k_remove):
            v = random.choice(nodes_in)
            S_new &= ~(1 << v)
            nodes_in.remove(v)
            nodes_out.append(v)
        # Add some random nodes
        k_add = min(strength, len(nodes_out))
        for _ in range(k_add):
            u = random.choice(nodes_out)
            S_new |= (1 << u)
            nodes_out.remove(u)
            nodes_in.append(u)
        return S_new
    
    # Main search loop
    start_time = time.time()
    deadline = 620  # safety margin
    
    best_set = 0
    best_size = 0
    best_conf = 0
    
    def update_best(S):
        nonlocal best_set, best_size, best_conf
        conf = calc_conflicts(S)
        size = S.bit_count()
        if conf == 0 and size > best_size:
            best_size = size
            best_set = S
            best_conf = 0
        elif conf == 0 and size == best_size:
            pass  # tie, keep existing
    
    # Initial greedy solutions
    for seed in range(20):
        S = greedy_init(order_seed=seed)
        S, size, conf = local_search(S, max_iters=50000)
        update_best(S)
        if time.time() - start_time > deadline:
            break
    
    # Iterated local search
    rng = random
    while time.time() - start_time < deadline:
        # Start from best or random
        if rng.random() < 0.7:
            S = best_set
        else:
            S = greedy_init()
        
        # Perturb
        S = perturb(S, strength=3)
        
        # Local search
        S, size, conf = local_search(S, max_iters=80000)
        update_best(S)
        
        if time.time() - start_time > deadline:
            break
    
    # Final polish: greedy fill on best_set
    changed = True
    while changed:
        changed = False
        for n in range(m):
            if not (best_set & (1 << n)) and (best_set & adj_mask[n]) == 0:
                best_set |= (1 << n)
                best_size += 1
                changed = True
                break
    
    # Try more local search rounds on the final best
    for _ in range(10):
        if time.time() - start_time > deadline:
            break
        S, size, conf = local_search(best_set, max_iters=50000)
        if conf == 0 and size > best_size:
            best_size = size
            best_set = S
    
    # Convert to array
    res = []
    for i in range(m):
        if best_set & (1 << i):
            res.append(i)
    
    return np.array(res, dtype=np.int64)
