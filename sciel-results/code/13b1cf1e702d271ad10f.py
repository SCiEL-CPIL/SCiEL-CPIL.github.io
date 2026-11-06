import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda u: len(adj[u] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P = P - {v}
                X = X | {v}
        
        bk(frozenset(), set(range(n)), frozenset())
        return cliques

    def min_hitting_set(cliques, n):
        if not cliques:
            return []
        
        m = len(cliques)
        
        # For each vertex, bitmask of cliques it hits
        vertex_cliques_mask = [0] * n
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques_mask[v] |= (1 << i)
        
        all_cliques_mask = (1 << m) - 1
        
        # Greedy upper bound
        remaining = all_cliques_mask
        greedy_cover = []
        
        while remaining:
            best_v = -1
            best_count = -1
            for v in range(n):
                count = bin(remaining & vertex_cliques_mask[v]).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v < 0 or best_count == 0:
                break
            greedy_cover.append(best_v)
            remaining &= ~vertex_cliques_mask[best_v]
        
        best_cover = list(greedy_cover)
        best_size = len(best_cover)
        
        if best_size == 0:
            return []
        
        # Local search
        def try_improve(cover, max_iter=30):
            improved = True
            iterations = 0
            while improved and iterations < max_iter:
                improved = False
                iterations += 1
                cover_set = set(cover)
                
                for v in list(cover_set):
                    new_cover = cover_set - {v}
                    new_covered = 0
                    for cv in new_cover:
                        new_covered |= vertex_cliques_mask[cv]
                    uncovered = all_cliques_mask & ~new_covered
                    if uncovered == 0:
                        cover_set = new_cover
                        improved = True
                        break
                    best_add = -1
                    best_add_count = -1
                    for av in range(n):
                        if av in new_cover:
                            continue
                        count = bin(uncovered & vertex_cliques_mask[av]).count('1')
                        if count > best_add_count:
                            best_add_count = count
                            best_add = av
                    if best_add >= 0 and best_add_count > 0:
                        new_cover2 = new_cover | {best_add}
                        if len(new_cover2) < len(cover_set):
                            cover_set = new_cover2
                            improved = True
                            break
            return list(cover_set)
        
        for _ in range(20):
            improved_cover = try_improve(best_cover, max_iter=30)
            if len(improved_cover) < len(best_cover):
                best_cover = improved_cover
                best_size = len(best_cover)
            else:
                break
        
        # B&B with bitmask
        def lower_bound(remaining_mask):
            if remaining_mask == 0:
                return 0
            rem = remaining_mask
            lb = 0
            while rem:
                ci = (rem & (-rem)).bit_length() - 1
                lb += 1
                clique_ci = cliques[ci]
                shared_mask = 0
                for v in clique_ci:
                    shared_mask |= vertex_cliques_mask[v]
                rem &= ~shared_mask
            return lb
        
        def bbs(remaining_mask, current_size, current_cover):
            nonlocal best_size, best_cover
            
            if time.time() > deadline:
                return
            
            if remaining_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = list(current_cover)
                return
            
            lb = current_size + lower_bound(remaining_mask)
            if lb >= best_size:
                return
            
            ci = (remaining_mask & (-remaining_mask)).bit_length() - 1
            branch_clique = cliques[ci]
            
            vertex_scores = []
            for v in branch_clique:
                score = bin(remaining_mask & vertex_cliques_mask[v]).count('1')
                vertex_scores.append((score, v))
            vertex_scores.sort(reverse=True)
            
            for score, v in vertex_scores:
                if time.time() > deadline:
                    return
                new_remaining = remaining_mask & ~vertex_cliques_mask[v]
                bbs(new_remaining, current_size + 1, current_cover + [v])
        
        bbs(all_cliques_mask, 0, [])
        
        return best_cover

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = instance(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cover = min_hitting_set(cliques, n)
        out.append(sorted(cover))
        
    return out
