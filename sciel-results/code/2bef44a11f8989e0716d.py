"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start_time = time.time()
    DEADLINE = 85.0
    
    def time_up():
        return time.time() - start_time > DEADLINE
    
    def pair_key(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return (c1, c2)
    
    best_result = None
    
    for max_colors in [3, 4, 5, 6]:
        if time_up():
            break
        
        color = [-1] * n
        pair_edges = {}
        
        def would_create_cycle(v, c, u, cu):
            key = pair_key(c, cu)
            elist = pair_edges.get(key)
            if not elist:
                return False
            adj_local = {}
            for a, b in elist:
                adj_local.setdefault(a, []).append(b)
                adj_local.setdefault(b, []).append(a)
            if v not in adj_local:
                return False
            visited = {v}
            queue = [v]
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                if cur == u:
                    return True
                for nb in adj_local.get(cur, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            return False
        
        def get_available(v):
            avail = []
            for c in range(max_colors):
                conflict = False
                for nb in adj[v]:
                    if color[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                cycle_conf = False
                for nb in adj[v]:
                    cn = color[nb]
                    if cn != -1 and cn != c:
                        if would_create_cycle(v, c, nb, cn):
                            cycle_conf = True
                            break
                if not cycle_conf:
                    avail.append(c)
            return avail
        
        def solve():
            if time_up():
                return False
            
            best_v = -1
            best_avail = None
            best_cnt = max_colors + 1
            
            for v in range(n):
                if color[v] == -1:
                    avail = get_available(v)
                    cnt = len(avail)
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best_v = v
                        best_avail = avail
                        if cnt == 0:
                            return False
                        if cnt == 1:
                            break
            
            if best_v == -1:
                return True
            
            for c in best_avail:
                color[best_v] = c
                added = []
                for nb in adj[best_v]:
                    cn = color[nb]
                    if cn != -1 and cn != c:
                        key = pair_key(c, cn)
                        if key not in pair_edges:
                            pair_edges[key] = []
                        pair_edges[key].append((best_v, nb))
                        added.append((nb, key))
                
                if solve():
                    return True
                
                for nb, key in added:
                    pair_edges[key].remove((best_v, nb))
                color[best_v] = -1
            
            return False
        
        if solve():
            used = sorted(set(color))
            mapping = {c: i for i, c in enumerate(used)}
            result = [mapping[color[v]] for v in range(n)]
            best_result = result
            break
    
    if best_result is None:
        color = [-1] * n
        for v in range(n):
            for c in range(50):
                if all(color[nb] != c for nb in adj[v] if color[nb] != -1):
                    color[v] = c
                    break
            if color[v] == -1:
                color[v] = 0
        best_result = color
    
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
