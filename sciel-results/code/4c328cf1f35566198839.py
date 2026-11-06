import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0

def _score_from_bitmasks(adj, deg):
    """
    adj: list of 16 ints (bitmasks)
    deg: list of 16 ints
    Returns max triangle degree sum.
    """
    best = 0
    for i in range(N):
        # Get neighbors of i
        # We only need to check triangles where i is the smallest vertex to avoid duplicates?
        # Actually, we can just iterate all pairs (i, j) with i < j and edge exists.
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                # Find common neighbors k > j
                common = adj[i] & adj[j]
                # Mask out i and j
                common &= ~((1 << i) | (1 << j))
                # Iterate through set bits in common
                # k must be > j to avoid re-counting? 
                # If we just check all k, we might count the triangle multiple times.
                # But we just need the MAX sum. Duplicates don't hurt the max.
                # However, to be efficient, we can just iterate all k in common.
                m = common
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
                    m ^= lsb
    return best

def _get_edges_and_nonedges(adj):
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
            else:
                nonedges.append((i, j))
    return edges, nonedges

def _apply_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    # Remove u-v
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1
    # Add x-y
    adj[x] |= (1 << y)
    adj[y] |= (1 << x)
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    # Undo add x-y (remove it)
    adj[x] &= ~(1 << y)
    adj[y] &= ~(1 << x)
    deg[x] -= 1
    deg[y] -= 1
    # Undo remove u-v (add it back)
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    num_restarts = 50
    max_iters_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE:
            break
            
        # Initialize: K_{8,8} + 1 edge
        adj = [0] * N
        for i in range(8):
            for j in range(8, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Add 1 random internal edge
        part = rng.choice([range(8), range(8, 16)])
        u, v = rng.sample(list(part), 2)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        deg = [bin(a).count('1') for a in adj]
        
        cur_score = _score_from_bitmasks(adj, deg)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = adj[:]
            best_deg = deg[:]
            
        edges, nonedges = _get_edges_and_nonedges(adj)
        
        # Simulated Annealing
        start_temp = 2.0
        end_temp = 0.01
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters_per_restart - 1))
        temp = start_temp
        
        for it in range(max_iters_per_restart):
            if time.time() - t0 > DEADLINE:
                break
                
            if not edges or not nonedges:
                break
                
            e_remove = edges[rng.randrange(len(edges))]
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            if (u == x and v == y) or (u == y and v == x):
                continue
                
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score_from_bitmasks(adj, deg)
            
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.3
            else:
                delta = new_score - cur_score
                if delta > 5:
                    accept = False
                elif temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    best_deg = deg[:]
                
                # Update lists
                # Remove e_remove from edges, add to nonedges
                # Remove e_add from nonedges, add to edges
                # To avoid O(N) list operations, we can just rebuild if needed, 
                # but for 16 nodes, lists are small.
                # However, doing list.remove is O(N). 
                # We can maintain a set or just swap indices.
                # For simplicity and speed, we'll just update the lists.
                # Optimization: Instead of remove/append, we can just pick random from lists 
                # and if the swap is accepted, we know the new state.
                # But the lists become stale if we don't update them.
                
                # Fast update:
                # Find index of e_remove in edges
                try:
                    idx_rm = edges.index(e_remove)
                except ValueError:
                    idx_rm = -1
                if idx_rm != -1:
                    edges[idx_rm] = e_add
                    # e_add is now in edges, remove from nonedges
                    try:
                        idx_add = nonedges.index(e_add)
                    except ValueError:
                        idx_add = -1
                    if idx_add != -1:
                        nonedges[idx_add] = e_remove
                else:
                    # Should not happen if state is consistent
                    pass
            else:
                _undo_swap(adj, deg, e_remove, e_add)
                
            temp *= cooling_rate

    if best_adj is None:
        # Fallback
        adj = [0] * N
        for i in range(8):
            for j in range(8, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    # Convert bitmask adj to edge list
    edges_out = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_adj[i] >> j) & 1:
                edges_out.append([i, j])
                
    return {'n': N, 'edges': edges_out}
