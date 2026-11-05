import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of bitmasks, each representing a maximal clique.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n):
        cliques = enumerate_maximal_cliques(adj, n)
        
        # Filter cliques of size >= 2
        cliques = [c for c in cliques if c.bit_count() >= 2]
        
        if not cliques:
            return []

        k = len(cliques)
        
        # If k is small, use exact BFS/DP on the state space of covered cliques
        # State: bitmask of covered cliques (k bits)
        # Transition: add a vertex v, new state = state | cover_mask[v]
        # Cost: 1 vertex
        if k <= 20:
            full_mask = (1 << k) - 1
            # cover_mask[v] is a bitmask of length k
            cover_mask = [0] * n
            
            for v in range(n):
                v_bit = 1 << v
                mask = 0
                for i, c in enumerate(cliques):
                    if c & v_bit:
                        mask |= (1 << i)
                cover_mask[v] = mask
            
            # BFS
            # dist[state] = min vertices
            dist = [-1] * (1 << k)
            parent = [None] * (1 << k) # (prev_state, vertex)
            
            dist[0] = 0
            q = deque([0])
            
            while q:
                curr = q.popleft()
                if curr == full_mask:
                    break
                for v in range(n):
                    next_state = curr | cover_mask[v]
                    if dist[next_state] == -1:
                        dist[next_state] = dist[curr] + 1
                        parent[next_state] = (curr, v)
                        q.append(next_state)
            
            if dist[full_mask] != -1:
                # Reconstruct path
                selected = []
                state = full_mask
                while state != 0:
                    prev_state, v = parent[state]
                    selected.append(v)
                    state = prev_state
                return sorted(selected)
            else:
                # Should not happen if graph is connected? 
                # Actually, if some clique is empty? No, cliques have size >= 2.
                # If no vertex covers a clique? Impossible, clique vertices cover it.
                # So full_mask is always reachable.
                pass

        # Fallback: Greedy Set Cover
        remaining_cliques = set(cliques)
        selected = []
        
        while remaining_cliques and time.time() < deadline:
            best_vertex = -1
            max_hits = 0
            
            for v in range(n):
                v_mask = 1 << v
                hits = 0
                for c in remaining_cliques:
                    if c & v_mask:
                        hits += 1
                
                if hits > max_hits:
                    max_hits = hits
                    best_vertex = v
            
            if best_vertex == -1 or max_hits == 0:
                break
            
            selected.append(best_vertex)
            best_mask = 1 << best_vertex
            
            new_remaining = set()
            for c in remaining_cliques:
                if not (c & best_mask):
                    new_remaining.add(c)
            remaining_cliques = new_remaining
            
            if not remaining_cliques:
                break
        
        if remaining_cliques:
            for c in remaining_cliques:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if v not in selected:
                    selected.append(v)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            adj = random_graph(n, p, seed)
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        transversal = solve_instance(adj, n)
        results.append(transversal)

    return results
