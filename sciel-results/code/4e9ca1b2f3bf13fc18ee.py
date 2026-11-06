import time
import random
import math

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    
    A K3,3 containing (u,v) must have u and v in opposite parts.
    Let parts be A and B. WLOG u in A, v in B.
    We need 2 other vertices in A, say x, y, and 2 in B, say b1, b2.
    Conditions:
    1. x, y in N(u) and x, y in N(v)  => x, y in N(u) & N(v)
    2. b1, b2 in N(v) and b1, b2 in N(u) => b1, b2 in N(u) & N(v)
    3. x, y connected to b1, b2 => b1, b2 in N(x) & N(y)
    
    So, let M = N(u) & N(v).
    We need to check if there exist x, y in M and b1, b2 in M
    such that b1, b2 in N(x) & N(y).
    This is equivalent to:
    For any pair x, y in M, check if |N(x) & N(y) & M| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    m = nu & nv
    
    # Extract vertices in M
    verts = []
    temp = m
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        if idx != u and idx != v:
            verts.append(idx)
        temp -= lsb
        
    for i in range(len(verts)):
        x = verts[i]
        ax = adj[x]
        for j in range(i+1, len(verts)):
            y = verts[j]
            ay = adj[y]
            # Check if there are at least 2 common neighbors in M
            common = ax & ay & m
            if common.bit_count() >= 2:
                return False
    return True

def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_adj = None
    best_count = 0
    best_adj_ref = [None]
    best_count_ref = [0]
    
    # Helper to update best
    def update_best(adj):
        c = count_edges(adj)
        if c > best_count_ref[0]:
            best_count_ref[0] = c
            best_adj_ref[0] = adj[:]
            
    # Phase 1: Structured Initialization
    # Try a few random structured starts
    adj = [0] * N
    # Random 5-regular-ish
    for _ in range(50):
        if time.time() > deadline - 10: break
        cand = [0] * N
        # Connect i to 5 random
        for i in range(N):
            candidates = [j for j in range(N) if j != i]
            random.shuffle(candidates)
            for j in candidates[:5]:
                cand[i] |= (1 << j)
                cand[j] |= (1 << i)
        # Verify and repair? Or just start search from here.
        # Greedy completion
        missing = [(i, j) for i in range(N) for j in range(i+1, N) if not (cand[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if is_valid_add(u, v, cand):
                cand[u] |= (1 << v)
                cand[v] |= (1 << u)
        update_best(cand)
        if best_adj_ref[0]:
            adj = best_adj_ref[0][:]

    # Phase 2: Tabu Search
    def tabu_search(adj, deadline):
        tabu_list = {}
        tabu_tenure = 10
        iteration = 0
        no_improve = 0
        current_count = count_edges(adj)
        
        while time.time() < deadline:
            iteration += 1
            no_improve += 1
            
            if no_improve > 50:
                tabu_tenure = max(5, tabu_tenure // 2)
                no_improve = 0
                
            moves = []
            missing = [(i, j) for i in range(N) for j in range(i+1, N) if not (adj[i] & (1 << j))]
            
            # Add moves
            for u, v in missing:
                if is_valid_add(u, v, adj):
                    moves.append(('add', u, v, 1))
                    
            # 1-swap moves: remove 1, add 2
            current_edges = [(i, j) for i in range(N) for j in range(i+1, N) if (adj[i] & (1 << j))]
            for u1, v1 in current_edges:
                # Remove u1-v1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try to add 2
                for u2, v2 in missing:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1): continue
                    if is_valid_add(u2, v2, adj):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        
                        for u3, v3 in missing:
                            if (u3, v3) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]: continue
                            if is_valid_add(u3, v3, adj):
                                moves.append(('swap1', u1, v1, u2, v2, u3, v3, 1))
                                break
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                
                # Restore u1-v1
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
            
            if not moves:
                break
                
            # Select best non-tabu move
            best_move = None
            best_score = -1
            
            for move in moves:
                # Check tabu
                is_tabu = False
                if move[0] == 'add':
                    _, u, v, _ = move
                    key = (u, v) if u < v else (v, u)
                    if key in tabu_list and tabu_list[key] > iteration:
                        is_tabu = True
                elif move[0] == 'swap1':
                    _, u1, v1, u2, v2, u3, v3, _ = move
                    k1 = (u1, v1) if u1 < v1 else (v1, u1)
                    k2 = (u2, v2) if u2 < v2 else (v2, u2)
                    k3 = (u3, v3) if u3 < v3 else (v3, u3)
                    if (k1 in tabu_list and tabu_list[k1] > iteration) or \
                       (k2 in tabu_list and tabu_list[k2] > iteration) or \
                       (k3 in tabu_list and tabu_list[k3] > iteration):
                        is_tabu = True
                       
                if is_tabu: continue
                if 1 > best_score:
                    best_score = 1
                    best_move = move
                    
            if best_move is None:
                # Aspiration
                for move in moves:
                    if move[0] == 'add':
                        if current_count + 1 > best_count_ref[0]:
                            best_move = move
                            break
                    elif move[0] == 'swap1':
                        if current_count + 1 > best_count_ref[0]:
                            best_move = move
                            break
                            
            if best_move is None:
                break
                
            # Apply
            if best_move[0] == 'add':
                _, u, v, _ = best_move
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                tabu_list[(u, v) if u < v else (v, u)] = iteration + tabu_tenure
            elif best_move[0] == 'swap1':
                _, u1, v1, u2, v2, u3, v3, _ = best_move
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                adj[u3] |= (1 << v3)
                adj[v3] |= (1 << u3)
                tabu_list[(u1, v1) if u1 < v1 else (v1, u1)] = iteration + tabu_tenure
                tabu_list[(u2, v2) if u2 < v2 else (v2, u2)] = iteration + tabu_tenure
                tabu_list[(u3, v3) if u3 < v3 else (v3, u3)] = iteration + tabu_tenure
                
            current_count = count_edges(adj)
            if current_count > best_count_ref[0]:
                best_count_ref[0] = current_count
                best_adj_ref[0] = adj[:]
                no_improve = 0
                
        return best_adj_ref[0]

    if best_adj_ref[0]:
        tabu_search(best_adj_ref[0][:], deadline)
        
    if best_adj_ref[0] is None:
        return []
        
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj_ref[0][u] >> v) & 1:
                edges.append([u, v])
                
    return edges
