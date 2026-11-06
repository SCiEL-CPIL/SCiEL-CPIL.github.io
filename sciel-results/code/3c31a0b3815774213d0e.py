import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0

def _score_from_bitmasks(adj, deg):
    best = 0
    for i in range(N):
        mask_i = adj[i]
        for j in range(i + 1, N):
            if (mask_i >> j) & 1:
                common = mask_i & adj[j]
                # Exclude i and j
                common &= ~((1 << i) | (1 << j))
                m = common
                di = deg[i]
                dj = deg[j]
                while m:
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    s = di + dj + deg[k]
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
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1
    adj[x] |= (1 << y)
    adj[y] |= (1 << x)
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[x] &= ~(1 << y)
    adj[y] &= ~(1 << x)
    deg[x] -= 1
    deg[y] -= 1
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1

def construct():
    t0 = time.time()
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    num_restarts = 30 # Fewer restarts, but deeper search per restart
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE:
            break
        
        # Initialize
        adj = [0] * N
        for i in range(8):
            for j in range(8, 16):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # Random internal edge
        rng = random.Random(42 + r)
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
        
        # SA Parameters
        start_temp = 3.0
        end_temp = 0.05
        max_iters = 5000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        tabu_list = [] # List of (e_remove, e_add)
        TABU_SIZE = 50
        stagnation = 0
        STAGNATION_LIMIT = 50
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE:
                break
                
            if not edges or not nonedges:
                break
                
            if stagnation > STAGNATION_LIMIT:
                # Polish: Find best move
                best_move_score = float('inf')
                best_move_rm = None
                best_move_add = None
                
                # Sample a subset of moves if full evaluation is too slow, 
                # but 65*55 is small. Let's try full.
                for e_rm in edges:
                    for e_add in nonedges:
                        u, v = e_rm
                        x, y = e_add
                        if (u == x and v == y) or (u == y and v == x):
                            continue
                        _apply_swap(adj, deg, e_rm, e_add)
                        s = _score_from_bitmasks(adj, deg)
                        _undo_swap(adj, deg, e_rm, e_add)
                        if s < best_move_score:
                            best_move_score = s
                            best_move_rm = e_rm
                            best_move_add = e_add
                
                if best_move_rm is not None:
                    # Apply best move
                    _apply_swap(adj, deg, best_move_rm, best_move_add)
                    cur_score = best_move_score
                    
                    # Update lists
                    edges.remove(best_move_rm)
                    nonedges.append(best_move_rm)
                    nonedges.remove(best_move_add)
                    edges.append(best_move_add)
                    
                    # Update tabu
                    tabu_list.append((best_move_rm, best_move_add))
                    if len(tabu_list) > TABU_SIZE:
                        tabu_list.pop(0)
                    
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj = adj[:]
                        best_deg = deg[:]
                    
                    stagnation = 0
                    temp = max(temp * 0.5, end_temp) # Cool down after polish
                    continue # Skip random move this iteration
            
            # Random Move
            e_remove = edges[rng.randrange(len(edges))]
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            if (u == x and v == y) or (u == y and v == x):
                continue
            
            # Tabu check
            is_tabu = (e_remove, e_add) in tabu_list
            
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score_from_bitmasks(adj, deg)
            
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.3
            else:
                delta = new_score - cur_score
                if is_tabu and delta > 0:
                    # Aspiration criterion: accept if it improves global best
                    if new_score < best_score:
                        accept = True
                    else:
                        accept = False
                elif delta > 6:
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
                    stagnation = 0
                else:
                    stagnation += 1
                    
                # Update lists
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
                
                # Update tabu
                tabu_list.append((e_remove, e_add))
                if len(tabu_list) > TABU_SIZE:
                    tabu_list.pop(0)
            else:
                _undo_swap(adj, deg, e_remove, e_add)
                stagnation += 1
                
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

    edges_out = []
    for i in range(N):
        for j in range(i + 1, N):
            if (best_adj[i] >> j) & 1:
                edges_out.append([i, j])
                
    return {'n': N, 'edges': edges_out}
