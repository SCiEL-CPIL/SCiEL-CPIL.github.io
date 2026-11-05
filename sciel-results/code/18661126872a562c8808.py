"""Search for triangle-free graphs with alpha < 10, maximizing n."""
import random
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(13)
    
    def alpha_greedy(adj, n):
        """Greedy upper bound on alpha."""
        colored = [False] * n
        colors = [0] * n
        cc = 0
        for v in range(n):
            used = set()
            for u in range(n):
                if u == v or not colored[u]:
                    continue
                if not (adj[v] & (1 << u)):
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            colored[v] = True
            cc = max(cc, c + 1)
        return cc
    
    def exact_alpha(adj, n, limit=10):
        if n == 0:
            return 0
        best = [0]
        all_mask = (1 << n) - 1
        def dfs(cands, sz):
            if cands == 0:
                return
            if sz + bin(cands).count('1') <= best[0]:
                return
            if sz >= limit:
                best[0] = sz
                return
            v = -1
            md = n + 1
            c = cands
            while c:
                lsb = c & (-c)
                i = lsb.bit_length() - 1
                d = bin(cands & adj[i]).count('1')
                if d < md:
                    md = d
                    v = i
                    if d == 0:
                        break
                c ^= lsb
            if v < 0:
                return
            dfs(cands & ~adj[v] & ~(1 << v), sz + 1)
            dfs(cands & ~(1 << v), sz)
        dfs(all_mask, 0)
        return best[0]
    
    def is_tri_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    if adj[i] & adj[j] & ~((1 << i) | (1 << j)):
                        return False
        return True
    
    def edges_from_adj(adj, n):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
    
    best_n = 26
    best_edges = None
    
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        
        # Start from Andrasfai
        n = 26
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(15):
            if time.time() > deadline - 40:
                break
            added = False
            for attempt in range(30):
                sz = rng.randint(1, 4)
                S = set()
                for _ in range(sz):
                    v = rng.randint(0, n - 1)
                    S.add(v)
                Sv = list(S)
                valid = True
                for i in range(len(Sv)):
                    for j in range(i + 1, len(Sv)):
                        if adj[Sv[i]] & (1 << Sv[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                new_adj = adj + [0]
                for v in S:
                    new_adj[v] |= (1 << n)
                    new_adj[n] |= (1 << v)
                
                if not is_tri_free(new_adj, n + 1):
                    continue
                
                # Fast check: greedy alpha bound
                if alpha_greedy(new_adj, n + 1) >= 10:
                    continue
                
                # Exact check
                if exact_alpha(new_adj, n + 1, 10) < 10:
                    adj = new_adj
                    n += 1
                    added = True
                    break
            
            if not added:
                break
        
        if n > best_n:
            if is_tri_free(adj, n) and exact_alpha(adj, n, 10) < 10:
                best_n = n
                best_edges = edges_from_adj(adj, n)
    
    if best_edges is None:
        n = 26
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_edges = edges_from_adj(adj, n)
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
