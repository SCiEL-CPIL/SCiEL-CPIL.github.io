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

    def max_disjoint_cliques(cliques):
        """Greedy approximation of the maximum number of disjoint cliques.
        This provides a lower bound on the hitting set size."""
        if not cliques:
            return 0
        # Sort cliques by size (smaller first) to greedily pack them
        sorted_cliques = sorted(cliques, key=len)
        count = 0
        remaining = sorted_cliques[:]
        while remaining:
            # Pick the first clique
            c = remaining[0]
            count += 1
            # Remove all cliques that intersect with c
            new_remaining = []
            for rc in remaining[1:]:
                if c.isdisjoint(rc):
                    new_remaining.append(rc)
            remaining = new_remaining
        return count

    def branch_and_bound(cliques, n, time_limit):
        """
        Exact solver using Branch and Bound.
        Returns the best hitting set found within time_limit.
        """
        t0 = time.time()
        # Initial solution via greedy
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        # Pre-compute for speed: list of cliques as frozensets
        cliques_list = list(cliques)
        
        # Lower bound calculation function
        def lb(cliques_subset):
            if not cliques_subset:
                return 0
            # Use a fast greedy disjoint set count
            # To make it faster, we can sample or use a simpler bound
            # For now, use the greedy max disjoint
            return max_disjoint_cliques(cliques_subset)

        # Branching heuristic: pick the clique that has the smallest "flexibility"
        # i.e., smallest size, or vertices that appear in fewest other cliques.
        # Let's stick to smallest size for simplicity and speed.
        
        def solve(cliques_subset, current_hit, depth):
            nonlocal best_hit, best_size, t0
            
            if time.time() - t0 > time_limit:
                return False # Timeout signal
            
            if not cliques_subset:
                if len(current_hit) < best_size:
                    best_size = len(current_hit)
                    best_hit = set(current_hit)
                return True
            
            # Pruning
            if depth >= best_size:
                return True
            
            # Lower bound
            lb_val = lb(cliques_subset)
            if depth + lb_val >= best_size:
                return True

            # Select a clique to branch on
            # Heuristic: Pick the clique with the minimum size
            # If ties, pick one with vertices that are in the fewest other cliques
            min_size_clique = min(cliques_subset, key=len)
            
            # Order vertices in the clique by frequency (least frequent first?)
            # Actually, branching on a vertex that covers many cliques is better.
            # So we should try vertices that cover the most cliques first.
            freq = {}
            for c in cliques_subset:
                for v in c:
                    freq[v] = freq.get(v, 0) + 1
            
            # Sort vertices in min_size_clique by descending frequency
            verts = sorted(min_size_clique, key=lambda v: -freq.get(v, 0))
            
            for v in verts:
                # New state: add v to hit, remove all cliques containing v
                new_hit = current_hit | {v}
                new_cliques = [c for c in cliques_subset if v not in c]
                
                # Recursive call
                # We don't need to pass current_hit as a list, set is fine
                if not solve(new_cliques, new_hit, depth + 1):
                    return False # Timeout propagated
            
            return True

        # Start BnB
        # We pass the full list of cliques
        # To avoid recursion depth issues, we assume n is small enough (40)
        # but the depth of the tree is the size of the hitting set.
        solve(cliques_list, set(), 0)
        
        return best_hit

    def local_search(hit, cliques, n, time_limit):
        """Try deletions and 2-opt swaps to reduce size."""
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
            # Try 2-opt swaps: remove v, add w (w not in hit)
            if not improved:
                for v in list(hit):
                    if time.time() > t0 + time_limit:
                        break
                    trial = hit - {v}
                    for w in range(n):
                        if w in hit:
                            continue
                        trial2 = trial | {w}
                        if len(trial2) < len(hit):
                            valid = True
                            for c in cliques:
                                if not any(v2 in trial2 for v2 in c):
                                    valid = False
                                    break
                            if valid:
                                hit = trial2
                                improved = True
                                break
                    if improved:
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
        # Give BnB a significant chunk of time, e.g., 60% of remaining
        bnb_time = min(remaining * 0.6, 30.0)
        hit_bnb = branch_and_bound(cliques, n, bnb_time)
        
        # BnB might return a smaller set. Use it.
        if len(hit_bnb) < len(hit):
            hit = hit_bnb
            
        # 3. Local Search to polish
        ls_time = remaining * 0.3
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
