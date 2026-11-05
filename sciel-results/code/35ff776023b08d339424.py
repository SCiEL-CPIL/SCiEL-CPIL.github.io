import time
import sys
import random
import math

sys.setrecursionlimit(20000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj[i]).count('1')
        
    ALL = (1 << n) - 1
    deadline = time.time() + 85.0
    
    # --- 1. Find Connected Components ---
    comp_of = [-1] * n
    comps = []
    for s in range(n):
        if comp_of[s] != -1:
            continue
        cid = len(comps)
        mask = 0
        stack = [s]
        comp_of[s] = cid
        mask |= 1 << s
        while stack:
            x = stack.pop()
            nb = adj[x]
            while nb:
                lsb = nb & (-nb)
                nb ^= lsb
                y = lsb.bit_length() - 1
                if comp_of[y] == -1:
                    comp_of[y] = cid
                    mask |= lsb
                    stack.append(y)
        comps.append(mask)
    
    comps.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    best_len = 0
    best_path = []
    
    # --- 2. Initial Heuristic: Greedy with Restart ---
    def greedy_path(start):
        path = [start]
        seen = 1 << start
        while True:
            cand = adj[path[-1]] & ~seen
            if not cand:
                break
            best_u = -1
            best_score = float('inf')
            m = cand
            while m:
                lsb = m & (-m)
                m ^= lsb
                u = lsb.bit_length() - 1
                score = bin(adj[u] & ~seen & ~(1 << u)).count('1')
                if score < best_score:
                    best_score = score
                    best_u = u
            path.append(best_u)
            seen |= 1 << best_u
        return path

    if comps:
        main_comp = comps[0]
        verts_in_comp = []
        m = main_comp
        while m:
            lsb = m & (-m)
            m ^= lsb
            verts_in_comp.append(lsb.bit_length() - 1)
        
        # Try greedy from all vertices, sorted by degree
        verts_in_comp.sort(key=lambda x: deg[x])
        
        for s in verts_in_comp:
            if time.time() > deadline:
                break
            p = greedy_path(s)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # --- 3. Simulated Annealing / Local Search ---
    if not best_path:
        best_path = [verts_in_comp[0]]
        best_len = 1
        
    # State: path, seen_mask
    current_path = list(best_path)
    current_seen = 0
    for v in current_path:
        current_seen |= 1 << v
    current_len = len(current_path)
    
    T = 10.0
    T_end = 0.01
    alpha = 0.995
    
    def try_extend_end(path, seen_mask, end_idx):
        end_v = path[end_idx]
        cand = adj[end_v] & ~seen_mask
        if not cand:
            return None, None, False
        
        m = cand
        count = bin(m).count('1')
        r = random.randint(0, count - 1)
        target = m
        for _ in range(r):
            target &= target - 1
        lsb = target & (-target)
        u = lsb.bit_length() - 1
        
        new_path = list(path)
        if end_idx == 0:
            new_path.insert(0, u)
        else:
            new_path.append(u)
        new_seen = seen_mask | (1 << u)
        return new_path, new_seen, True

    def try_shrink_end(path, seen_mask, end_idx):
        if len(path) <= 1:
            return None, None, False
        if end_idx == 0:
            v = path[0]
            new_path = path[1:]
        else:
            v = path[-1]
            new_path = path[:-1]
        new_seen = seen_mask & ~(1 << v)
        return new_path, new_seen, True

    def try_insert(path, seen_mask):
        if len(path) < 2:
            return None, None, False
            
        i = random.randint(0, len(path) - 2)
        u = path[i]
        v = path[i+1]
        
        cand = adj[u] & adj[v] & ~seen_mask
        if not cand:
            return None, None, False
            
        m = cand
        count = bin(m).count('1')
        r = random.randint(0, count - 1)
        target = m
        for _ in range(r):
            target &= target - 1
        lsb = target & (-target)
        w = lsb.bit_length() - 1
        
        new_path = list(path)
        new_path.insert(i+1, w)
        new_seen = seen_mask | (1 << w)
        return new_path, new_seen, True

    while time.time() < deadline:
        move_type = random.randint(0, 3)
        
        new_path = None
        new_seen = None
        
        if move_type == 0:
            new_path, new_seen, ok = try_extend_end(current_path, current_seen, 0)
        elif move_type == 1:
            new_path, new_seen, ok = try_extend_end(current_path, current_seen, -1)
        elif move_type == 2:
            end_idx = random.choice([0, -1])
            new_path, new_seen, ok = try_shrink_end(current_path, current_seen, end_idx)
        else:
            new_path, new_seen, ok = try_insert(current_path, current_seen)
            
        if not ok:
            T *= alpha
            if T < T_end:
                T = T_start * 0.5
            continue
            
        new_len = len(new_path)
        delta = new_len - current_len
        
        if delta > 0:
            current_path = new_path
            current_seen = new_seen
            current_len = new_len
            
            if current_len > best_len:
                best_len = current_len
                best_path = current_path
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_path = new_path
                current_seen = new_seen
                current_len = new_len
                
        T *= alpha
        if T < T_end:
            T = T_start * 0.5
            
        if random.random() < 0.01:
             improved = True
             while improved and time.time() < deadline:
                 improved = False
                 
                 end_v = current_path[0]
                 cand = adj[end_v] & ~current_seen
                 if cand:
                     m = cand
                     best_u = -1
                     best_score = float('inf')
                     while m:
                         lsb = m & (-m)
                         m ^= lsb
                         u = lsb.bit_length() - 1
                         score = bin(adj[u] & ~current_seen & ~(1 << u)).count('1')
                         if score < best_score:
                             best_score = score
                             best_u = u
                     if best_u != -1:
                         current_path.insert(0, best_u)
                         current_seen |= 1 << best_u
                         current_len += 1
                         improved = True
                         
                 if improved:
                     continue
                     
                 end_v = current_path[-1]
                 cand = adj[end_v] & ~current_seen
                 if cand:
                     m = cand
                     best_u = -1
                     best_score = float('inf')
                     while m:
                         lsb = m & (-m)
                         m ^= lsb
                         u = lsb.bit_length() - 1
                         score = bin(adj[u] & ~current_seen & ~(1 << u)).count('1')
                         if score < best_score:
                             best_score = score
                             best_u = u
                     if best_u != -1:
                         current_path.append(best_u)
                         current_seen |= 1 << best_u
                         current_len += 1
                         improved = True
                         
                 if improved:
                     continue
                     
                 if len(current_path) >= 2:
                     for i in range(len(current_path) - 1):
                         u = current_path[i]
                         v = current_path[i+1]
                         cand = adj[u] & adj[v] & ~current_seen
                         if cand:
                             lsb = cand & (-cand)
                             w = lsb.bit_length() - 1
                             current_path.insert(i+1, w)
                             current_seen |= 1 << w
                             current_len += 1
                             improved = True
                             break
                             
             if current_len > best_len:
                 best_len = current_len
                 best_path = current_path

    if not best_path:
        best_path = [0]
        
    return best_path
