import time
import random as _rnd
import math

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s kill

    def generate_edges(n, p, seed):
        rng = _rnd.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """Return list of maximal cliques (each as frozenset) using pivot-based BK with bitmasks."""
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            # Pivot: choose vertex in P|X with max degree in P
            PX = P | X
            if PX:
                # Pick pivot with max |adj[p] & P|
                best_p = -1
                best_cnt = -1
                pp = PX
                while pp:
                    lb = pp & (-pp)
                    p = lb.bit_length() - 1
                    cnt = (adj[p] & P).bit_count()
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_p = p
                    pp ^= lb
                # Candidates: P \ adj[pivot]
                candidates = P & ~(adj[best_p])
            else:
                candidates = P

            cp = candidates
            while cp:
                lb = cp & (-cp)
                u = lb.bit_length() - 1
                cp ^= lb
                bk(R | lb, P & adj[u], X & adj[u])
                P ^= lb
                X |= lb

        bk(0, all_mask, 0)
        return cliques

    def filter_cliques(cliques, min_size=2):
        return [c for c in cliques if len(c) >= min_size]

    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = set(range(len(cliques)))
        vertex_cliques = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(ci)

        hit = set()
        while uncovered:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                cnt = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_cnt <= 0:
                break
            hit.add(best_v)
            for ci in vertex_cliques[best_v]:
                uncovered.discard(ci)
        return hit

    def max_disjoint_cliques_bitmask(clique_masks, n):
        """Fast greedy max disjoint cliques using bitmasks for intersection testing."""
        if not clique_masks:
            return 0
        # Sort by size (smaller first) for better packing
        indexed = list(enumerate(clique_masks))
        indexed.sort(key=lambda x: x[1].bit_count())
        count = 0
        used = [False] * len(clique_masks)
        for i, mask_i in indexed:
            if used[i]:
                continue
            count += 1
            # Mark all cliques that intersect with this one
            for j in range(len(clique_masks)):
                if not used[j] and (clique_masks[i] & clique_masks[j]):
                    used[j] = True
        return count

    def branch_and_bound(cliques, n, time_limit):
        """
        Exact solver using Branch and Bound with bitmask-based lower bound.
        Returns the best hitting set found within time_limit.
        """
        t0 = time.time()
        # Initial solution via greedy
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        # Pre-compute clique masks for fast intersection testing
        clique_masks = [0] * len(cliques)
        for i, c in enumerate(cliques):
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks[i] = mask
        
        # Pre-compute for each vertex, the bitmask of cliques it covers
        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)
        
        all_clique_mask = (1 << len(cliques)) - 1
        
        # Count of cliques covered by each vertex
        vertex_clique_count = [0] * n
        for v in range(n):
            vertex_clique_count[v] = vertex_clique_mask[v].bit_count()
        
        # Lower bound: greedy max disjoint cliques using bitmask
        def compute_lb(remaining_clique_mask):
            if not remaining_clique_mask:
                return 0
            # Greedy: pick clique, mark intersecting ones
            count = 0
            remaining = remaining_clique_mask
            while remaining:
                # Find clique with smallest size among remaining
                best_i = -1
                best_size_clique = n + 1
                rm = remaining
                while rm:
                    lb = rm & (-rm)
                    i = lb.bit_length() - 1
                    sz = clique_masks[i].bit_count()
                    if sz < best_size_clique:
                        best_size_clique = sz
                        best_i = i
                    rm ^= lb
                if best_i < 0:
                    break
                count += 1
                # Remove all cliques that intersect with best_i
                intersecting_mask = 0
                rm2 = remaining
                while rm2:
                    lb2 = rm2 & (-rm2)
                    j = lb2.bit_length() - 1
                    if clique_masks[best_i] & clique_masks[j]:
                        intersecting_mask |= lb2
                    rm2 ^= lb2
                remaining &= ~intersecting_mask
            return count
        
        # Branch and bound solver
        def solve(remaining_clique_mask, current_hit, depth):
            nonlocal best_hit, best_size, t0
            
            if time.time() - t0 > time_limit:
                return False
            
            if not remaining_clique_mask:
                if len(current_hit) < best_size:
                    best_size = len(current_hit)
                    best_hit = set(current_hit)
                return True
            
            # Pruning
            if depth >= best_size:
                return True
            
            # Lower bound
            lb_val = compute_lb(remaining_clique_mask)
            if depth + lb_val >= best_size:
                return True

            # Select a clique to branch on: smallest size, then by vertex rarity
            best_clique_i = -1
            best_clique_size = n + 1
            rm = remaining_clique_mask
            while rm:
                lb = rm & (-rm)
                i = lb.bit_length() - 1
                sz = clique_masks[i].bit_count()
                if sz < best_clique_size:
                    best_clique_size = sz
                    best_clique_i = i
                rm ^= lb
            
            if best_clique_i < 0:
                return True
            
            # Get vertices in the selected clique
            clique_mask = clique_masks[best_clique_i]
            verts = []
            cm = clique_mask
            while cm:
                lb = cm & (-cm)
                v = lb.bit_length() - 1
                verts.append(v)
                cm ^= lb
            
            # Order vertices by descending frequency in remaining cliques
            # Higher frequency = covers more remaining cliques = better to try first
            verts.sort(key=lambda v: -vertex_clique_mask[v].bit_count())
            
            for v in verts:
                # New state: add v to hit, remove all cliques containing v
                new_hit = current_hit | {v}
                # Remove cliques that contain vertex v
                new_remaining = remaining_clique_mask & ~vertex_clique_mask[v]
                
                if not solve(new_remaining, new_hit, depth + 1):
                    return False
            
            return True

        # Start BnB
        solve(all_clique_mask, set(), 0)
        
        return best_hit

    def local_search(hit, cliques, n, time_limit):
        """Try deletions and 2-exchange moves to reduce size."""
        hit = set(hit)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # Try deletions
            for v in list(hit):
                trial = hit - {v}
                valid = True
                for c in cliques:
                    if not any(v2 in trial for v2 in c):
                        valid = False
                        break
                if valid:
                    hit = trial
                    improved = True
                    break
            # Try 2-exchange: remove two vertices, add one (net -1)
            if not improved and len(hit) >= 2:
                hit_list = list(hit)
                found = False
                for i in range(len(hit_list)):
                    if time.time() - t0 > time_limit:
                        break
                    v1 = hit_list[i]
                    for j in range(i + 1, len(hit_list)):
                        if time.time() - t0 > time_limit:
                            break
                        v2 = hit_list[j]
                        trial = hit - {v1, v2}
                        # Find a single vertex w that covers all cliques not covered by trial
                        # and is not already in trial
                        uncovered_cliques = []
                        for c in cliques:
                            if not any(v in trial for v in c):
                                uncovered_cliques.append(c)
                        if not uncovered_cliques:
                            # Can just remove both
                            hit = trial
                            improved = True
                            found = True
                            break
                        # Try adding each candidate w
                        for w in range(n):
                            if w in trial:
                                continue
                            # Check if w covers all uncovered cliques
                            covers_all = True
                            for uc in uncovered_cliques:
                                if w not in uc:
                                    covers_all = False
                                    break
                            if covers_all:
                                hit = trial | {w}
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            # Try 1-exchange: remove v, add w where w covers more than v
            if not improved:
                for v in list(hit):
                    if time.time() - t0 > time_limit:
                        break
                    trial = hit - {v}
                    # Find w not in hit that when added makes trial valid
                    # and w is "better" than v (covers cliques v didn't)
                    best_w = -1
                    best_gain = 0
                    for w in range(n):
                        if w in hit:
                            continue
                        # Check validity
                        valid = True
                        for c in cliques:
                            if not any(v2 in (trial | {w}) for v2 in c):
                                valid = False
                                break
                        if valid:
                            # Count how many cliques w covers that v doesn't
                            gain = 0
                            for c in cliques:
                                if w in c and v not in c:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_w = w
                    if best_w >= 0:
                        hit = trial | {best_w}
                        improved = True
                        break
        return hit

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Not enough time; return all vertices as fallback
            results.append(list(range(n)))
            continue

        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            results.append([])
            continue

        # 1. Greedy Initial Solution
        hit = greedy_hitting_set(cliques, n)
        
        # 2. Branch and Bound with time limit
        # Give BnB a significant chunk of time, e.g., 65% of remaining
        bnb_time = min(remaining * 0.65, 35.0)
        hit_bnb = branch_and_bound(cliques, n, bnb_time)
        
        # BnB might return a smaller set. Use it.
        if len(hit_bnb) < len(hit):
            hit = hit_bnb
            
        # 3. Local Search to polish
        ls_time = remaining * 0.30
        hit = local_search(hit, cliques, n, ls_time)

        # Verify validity
        valid = True
        for c in cliques:
            if not any(v in hit for v in c):
                valid = False
                break
        
        if not valid:
            # Fallback: all vertices in any clique
            all_v = set()
            for c in cliques:
                all_v |= c
            hit = all_v

        results.append(sorted(hit))

    return results
