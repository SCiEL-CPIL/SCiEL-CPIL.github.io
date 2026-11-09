import time
import random as _rnd
import math
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 115.0

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
        cliques = []
        all_mask = (1 << n) - 1

        def bk(R, P, X):
            if not P and not X:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            PX = P | X
            if PX:
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

    def max_disjoint_cliques_fast(cliques):
        if not cliques:
            return 0
        sorted_cliques = sorted(cliques, key=len)
        count = 0
        used = [False] * len(sorted_cliques)
        for i in range(len(sorted_cliques)):
            if used[i]:
                continue
            count += 1
            ci = sorted_cliques[i]
            for j in range(i + 1, len(sorted_cliques)):
                if not used[j] and not ci.isdisjoint(sorted_cliques[j]):
                    used[j] = True
        return count

    def lp_lower_bound(cliques, n):
        """
        Compute a lower bound via a fractional set cover relaxation.
        For each clique, at least one vertex must be selected.
        We solve a simplified LP relaxation using greedy dual fitting.
        """
        m = len(cliques)
        if m == 0:
            return 0
        
        # Greedy dual: assign weights to cliques
        # y_c >= 0, sum_{c: v in c} y_c <= 1 for all v
        # Maximize sum_c y_c
        # This is a packing problem; we use a greedy approach
        
        y = [0.0] * m
        remaining_capacity = [1.0] * n
        
        for _ in range(50):  # Iterate a few times
            improved = False
            for c_idx, c in enumerate(cliques):
                min_cap = min(remaining_capacity[v] for v in c)
                if min_cap > 0:
                    y[c_idx] += min_cap
                    for v in c:
                        remaining_capacity[v] -= min_cap
                    improved = True
            if not improved:
                break
        
        return sum(y)

    def branch_and_bound_mwis(cliques, n, time_limit):
        """
        Branch and bound for minimum hitting set using MWIS-style formulation.
        Key insight: The hitting set problem can be viewed as finding a minimum
        vertex cover in the hypergraph. We use a branch-and-bound that branches
        on vertices with strong lower bounds from LP relaxation.
        """
        t0 = time.time()
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        num_cliques = len(cliques)
        if num_cliques == 0:
            return set()
        
        # Precompute: for each vertex, which cliques it covers
        vertex_clique_list = [[] for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_clique_list[v].append(ci)
        
        # Sort cliques by size (smaller first for better branching)
        clique_order = sorted(range(num_cliques), key=lambda i: len(cliques[i]))
        
        # State: set of uncovered cliques
        # Use a frozenset of clique indices for state representation
        
        def get_lower_bound(uncovered_set):
            """Get a lower bound on additional vertices needed."""
            if not uncovered_set:
                return 0
            
            # Simple bound: max number of disjoint cliques in uncovered
            disjoint_count = 0
            remaining = set(uncovered_set)
            while remaining:
                # Pick smallest clique
                min_ci = min(remaining, key=lambda i: len(cliques[i]))
                disjoint_count += 1
                # Remove all cliques that intersect with this one
                min_clique = cliques[min_ci]
                new_remaining = set()
                for ci in remaining:
                    if cliques[ci].isdisjoint(min_clique):
                        new_remaining.add(ci)
                remaining = new_remaining
            
            # Also use LP bound
            lb_lp = lp_lower_bound([cliques[i] for i in uncovered_set], n)
            
            return max(disjoint_count, int(math.ceil(lb_lp)) - 1)  # -1 because we're counting additional
        
        def solve(uncovered_set, current_hit_size, current_hit):
            nonlocal best_hit, best_size, t0
            if time.time() - t0 > time_limit:
                return False
            
            if not uncovered_set:
                if current_hit_size < best_size:
                    best_size = current_hit_size
                    best_hit = set(current_hit)
                return True
            
            # Lower bound pruning
            lb = get_lower_bound(uncovered_set)
            if current_hit_size + lb >= best_size:
                return True
            
            # Choose clique to branch on: smallest uncovered clique
            target_ci = min(uncovered_set, key=lambda i: len(cliques[i]))
            target_clique = cliques[target_ci]
            
            # Order vertices by coverage (most covered first)
            verts = list(target_clique)
            verts.sort(key=lambda v: sum(1 for ci in uncovered_set if v in cliques[ci]), reverse=True)
            
            for v in verts:
                if time.time() - t0 > time_limit:
                    return False
                # If we add v, which cliques become covered?
                new_uncovered = set()
                for ci in uncovered_set:
                    if v not in cliques[ci]:
                        new_uncovered.add(ci)
                
                if not solve(new_uncovered, current_hit_size + 1, current_hit | {v}):
                    return False
            return True

        solve(set(range(num_cliques)), 0, set())
        return best_hit

    def local_search_2swap(hit, cliques, n, time_limit):
        """
        Local search with 1-swap (remove) and 2-swap (remove+add) moves.
        """
        hit = set(hit)
        t0 = time.time()
        improved = True
        
        while improved and (time.time() - t0) < time_limit:
            improved = False
            
            # 1-swap: try removing each vertex
            for v in list(hit):
                if time.time() - t0 > time_limit:
                    break
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
            
            # 2-swap: try removing one and adding another
            if not improved:
                for v in list(hit):
                    if time.time() - t0 > time_limit:
                        break
                    trial = hit - {v}
                    # Find which cliques are now uncovered
                    uncovered_cliques = []
                    for c in cliques:
                        if not any(v2 in trial for v2 in c):
                            uncovered_cliques.append(c)
                    
                    if not uncovered_cliques:
                        continue
                    
                    # Try adding each non-hit vertex
                    for w in range(n):
                        if w in hit:
                            continue
                        if time.time() - t0 > time_limit:
                            break
                        # Check if w covers all uncovered cliques
                        covers_all = all(w in c for c in uncovered_cliques)
                        if covers_all:
                            hit = trial | {w}
                            improved = True
                            break
                    if improved:
                        break
        return hit

    def simulated_annealing(hit, cliques, n, time_limit):
        """
        Simulated annealing to escape local minima.
        """
        t0 = time.time()
        hit = set(hit)
        best_hit = set(hit)
        best_size = len(hit)
        current_temp = 1.0
        cooling = 0.995
        
        while time.time() - t0 < time_limit:
            # Generate a random neighbor
            neighbor = set(hit)
            move_type = _rnd.random()
            
            if move_type < 0.4:
                # Remove a random vertex
                if hit:
                    v = _rnd.choice(list(hit))
                    neighbor.discard(v)
            elif move_type < 0.8:
                # Add a random vertex
                v = _rnd.randrange(n)
                neighbor.add(v)
            else:
                # Swap
                if hit:
                    v = _rnd.choice(list(hit))
                    w = _rnd.randrange(n)
                    neighbor.discard(v)
                    neighbor.add(w)
            
            # Check validity
            valid = True
            for c in cliques:
                if not any(v2 in neighbor for v2 in c):
                    valid = False
                    break
            
            if not valid:
                # Penalty for invalid
                delta = len(neighbor) - len(hit) + 10  # Large penalty
            else:
                delta = len(neighbor) - len(hit)
            
            # Accept or reject
            if delta < 0 or _rnd.random() < math.exp(-delta / current_temp):
                hit = neighbor
                if len(hit) < best_size:
                    best_size = len(hit)
                    best_hit = set(hit)
            
            current_temp *= cooling
            if current_temp < 0.01:
                current_temp = 1.0  # Restart
        
        return best_hit

    def iterative_deepening_bnb(cliques, n, time_limit):
        """
        Iterative deepening: try to find solutions of size k, k+1, ...
        """
        t0 = time.time()
        best_hit = greedy_hitting_set(cliques, n)
        best_size = len(best_hit)
        
        num_cliques = len(cliques)
        if num_cliques == 0:
            return set()
        
        def dfs(uncovered_set, current_size, limit, current_hit):
            nonlocal best_hit, best_size, t0
            if time.time() - t0 > time_limit:
                return False
            if current_size > limit:
                return False
            if not uncovered_set:
                if current_size < best_size:
                    best_size = current_size
                    best_hit = set(current_hit)
                return True
            
            # Lower bound
            if current_size + max(1, len(uncovered_set) // max(1, max(len(c) for c in [cliques[i] for i in uncovered_set]))) > limit:
                return False
            
            target_ci = min(uncovered_set, key=lambda i: len(cliques[i]))
            target_clique = cliques[target_ci]
            verts = list(target_clique)
            verts.sort(key=lambda v: sum(1 for ci in uncovered_set if v in cliques[ci]), reverse=True)
            
            for v in verts:
                if time.time() - t0 > time_limit:
                    return False
                new_uncovered = set(ci for ci in uncovered_set if v not in cliques[ci])
                if dfs(new_uncovered, current_size + 1, limit, current_hit | {v}):
                    return False
            return True
        
        # Start from current best size - 1 and go down
        for limit in range(best_size - 1, 0, -1):
            if time.time() - t0 > time_limit:
                break
            if dfs(set(range(num_cliques)), 0, limit, set()):
                break
        
        return best_hit

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining = deadline - time.time()
        if remaining < 10:
            results.append(list(range(n)))
            continue

        edges = generate_edges(n, p, seed)
        adj = build_adjacency(n, edges)
        cliques = bron_kerbosch(adj, n)
        cliques = filter_cliques(cliques, min_size=2)

        if not cliques:
            results.append([])
            continue

        # Start with greedy
        hit = greedy_hitting_set(cliques, n)
        
        # Phase 1: Branch and bound with MWIS-style formulation
        bnb_time = min(remaining * 0.4, 25.0)
        hit_bnb = branch_and_bound_mwis(cliques, n, bnb_time)
        if len(hit_bnb) < len(hit):
            hit = hit_bnb
        
        # Phase 2: Local search with 2-swap
        ls_time = min(remaining * 0.25, 15.0)
        hit = local_search_2swap(hit, cliques, n, ls_time)
        
        # Phase 3: Simulated annealing to escape local minima
        sa_time = min(remaining * 0.2, 12.0)
        hit_sa = simulated_annealing(hit, cliques, n, sa_time)
        if len(hit_sa) < len(hit):
            hit = hit_sa
        
        # Phase 4: Final local search
        final_ls_time = min(remaining * 0.1, 5.0)
        hit = local_search_2swap(hit, cliques, n, final_ls_time)

        # Validate
        valid = True
        for c in cliques:
            if not any(v in hit for v in c):
                valid = False
                break
        
        if not valid:
            all_v = set()
            for c in cliques:
                all_v |= c
            hit = all_v

        results.append(sorted(hit))

    return results
