import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _check_valid(adj, n):
    """
    Check if the graph is valid.
    Valid: No simple path s-u-v-t has d(s) == d(t).
    Uses bitmasks for adjacency for speed.
    """
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj[i]).count('1')
        
    # Group vertices by degree
    deg_groups = defaultdict(list)
    for i in range(n):
        deg_groups[deg[i]].append(i)
        
    for d_val, vertices in deg_groups.items():
        if len(vertices) < 2:
            continue
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                s = vertices[i]
                t = vertices[j]
                
                N_s = adj[s]
                N_t = adj[t]
                
                # Iterate over neighbors of s
                # u must be in N_s, u != t
                u_mask = N_s & ~(1 << t)
                
                while u_mask:
                    # Extract lowest set bit
                    u_bit = u_mask & -u_mask
                    u = u_bit.bit_length() - 1
                    u_mask ^= u_bit
                    
                    # v must be in N_t, v != s, v != u
                    # Also v must be neighbor of u
                    N_u = adj[u]
                    v_mask = N_t & N_u
                    v_mask &= ~(1 << s)
                    v_mask &= ~(1 << u)
                    
                    if v_mask:
                        return False
    return True

def _count_edges(adj, n):
    return sum(bin(adj[i]).count('1') for i in range(n)) // 2

def _construct_bipartite(n):
    """Construct a complete bipartite graph K_{k, n-k} with max edges."""
    best_adj = 0
    best_count = -1
    best_k = 0
    
    for k in range(1, n):
        adj = [0] * n
        # Partition: 0..k-1 and k..n-1
        for i in range(k):
            for j in range(k, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        cnt = _count_edges(adj, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
            best_k = k
            
    return best_adj, best_count

def _local_search(adj, n, rng, deadline):
    """
    Local search to maximize edges while maintaining validity.
    adj is a list of bitmasks.
    """
    cur_adj = adj[:]
    cur_count = _count_edges(cur_adj, n)
    best_adj = cur_adj[:]
    best_count = cur_count
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    no_improve = 0
    max_no_improve = 100 # Increased patience
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        # Try to add a random missing edge
        missing = [(i, j) for i, j in all_edges if not (cur_adj[i] & (1 << j))]
        if not missing:
            break
            
        rng.shuffle(missing)
        improved = False
        
        for (i, j) in missing:
            if time.time() > deadline:
                return best_adj, best_count
                
            # Try adding (i, j)
            cur_adj[i] |= (1 << j)
            cur_adj[j] |= (1 << i)
            
            if _check_valid(cur_adj, n):
                cur_count += 1
                improved = True
                if cur_count > best_count:
                    best_count = cur_count
                    best_adj = cur_adj[:]
                    no_improve = 0
                break
            else:
                # Revert
                cur_adj[i] &= ~(1 << j)
                cur_adj[j] &= ~(1 << i)
        
        if not improved:
            no_improve += 1
            # Try a swap move: remove one edge, try to add another
            existing = [(i, j) for i, j in all_edges if (cur_adj[i] & (1 << j))]
            if existing:
                rng.shuffle(existing)
                swap_attempted = 0
                for (a, b) in existing:
                    if swap_attempted > 20: # Limit swap attempts per cycle
                        break
                    swap_attempted += 1
                    if time.time() > deadline:
                        return best_adj, best_count
                        
                    cur_adj[a] &= ~(1 << b)
                    cur_adj[b] &= ~(1 << a)
                    temp_count = cur_count - 1
                    
                    # Try to add a different edge
                    missing_after_removal = [(i, j) for i, j in all_edges if not (cur_adj[i] & (1 << j))]
                    rng.shuffle(missing_after_removal)
                    swap_improved = False
                    for (c, d) in missing_after_removal:
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        cur_adj[c] |= (1 << d)
                        cur_adj[d] |= (1 << c)
                        if _check_valid(cur_adj, n):
                            # Keep the swap
                            cur_count = temp_count + 1
                            swap_improved = True
                            if cur_count > best_count:
                                best_count = cur_count
                                best_adj = cur_adj[:]
                                no_improve = 0
                            break
                        else:
                            cur_adj[c] &= ~(1 << d)
                            cur_adj[d] &= ~(1 << c)
                    
                    if not swap_improved:
                        # Revert removal
                        cur_adj[a] |= (1 << b)
                        cur_adj[b] |= (1 << a)
                        cur_count = temp_count + 1
                    else:
                        improved = True
                        break
                        
        if improved:
            no_improve = 0

    return best_adj, best_count

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_adj = None
    best_count = -1
    
    # 1. Start with best bipartite graph
    bip_adj, bip_count = _construct_bipartite(n)
    best_adj = bip_adj
    best_count = bip_count
    
    # 2. Local search from bipartite
    ls_adj, ls_count = _local_search(bip_adj, n, rng, deadline - 10.0)
    if ls_count > best_count:
        best_count = ls_count
        best_adj = ls_adj
        
    # 3. Random restarts with greedy construction
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 50:
        attempts += 1
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for i, j in edges:
            if time.time() > deadline - 5.0:
                break
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            if not _check_valid(adj, n):
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                
        g_count = _count_edges(adj, n)
        if g_count > best_count:
            best_count = g_count
            best_adj = adj[:]
            
        # Refine
        ls_adj2, ls_count2 = _local_search(adj, n, rng, deadline - 2.0)
        if ls_count2 > best_count:
            best_count = ls_count2
            best_adj = ls_adj2

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(n):
            if i < j and (best_adj[i] & (1 << j)):
                edges.append([int(i), int(j)])
                
    return edges

# EVOLVE-BLOCK-END
