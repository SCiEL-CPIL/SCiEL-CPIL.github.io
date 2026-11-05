"""Solve minimum transversal for maximal cliques of three fixed random graphs."""
import time
import random
import sys

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def bron_kerbosch(edges, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot: vertex in P∪X with max |P ∩ adj[u]|
            candidates = P | X
            pivot = max(candidates, key=lambda u: len(P & adj[u]))
            # Iterate over P \ adj[pivot]
            for v in list(P - adj[pivot]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques
    
    def greedy_hitting_set(cliques, n):
        """Greedy: repeatedly pick vertex covering most uncovered cliques."""
        # Map each clique to its vertex set
        uncovered = set(range(len(cliques)))
        # Count how many cliques each vertex covers
        vert_clique_count = [0] * n
        vert_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_clique_count[v] += 1
                vert_cliques[v].add(i)
        
        selected = []
        remaining_cliques = set(range(len(cliques)))
        
        while remaining_cliques:
            # Find vertex covering most remaining cliques
            best_v = -1
            best_count = 0
            for v in range(n):
                count = len(vert_cliques[v] & remaining_cliques)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            remaining_cliques -= vert_cliques[best_v]
        
        return selected
    
    def local_search_improve(selected, cliques, n, max_iters=5000):
        """Try to improve by removing redundant vertices and 2-for-1 swaps."""
        selected = list(selected)
        # Build mapping: for each selected vertex, which cliques does it uniquely cover?
        
        def get_covered_cliques(vert_set):
            """Return set of clique indices covered by the given vertex set."""
            covered = set()
            for v in vert_set:
                for i, clique in enumerate(cliques):
                    if v in clique:
                        covered.add(i)
            return covered
        
        # Precompute: for each vertex, which cliques it belongs to
        vert_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_to_cliques[v].add(i)
        
        # Remove redundant vertices
        changed = True
        while changed:
            changed = False
            for i, v in enumerate(selected):
                # Check if removing v still covers all cliques
                remaining = set(selected)
                remaining.discard(v)
                # Check all cliques are still covered
                all_covered = True
                for clique in cliques:
                    if not (clique & remaining):
                        all_covered = False
                        break
                if all_covered:
                    selected = [x for x in selected if x != v]
                    changed = True
                    break
        
        # 2-for-1 swap: replace two vertices with one if possible
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    v1, v2 = selected[i], selected[j]
                    # Cliques that v1 covers
                    c1 = vert_to_cliques[v1]
                    c2 = vert_to_cliques[v2]
                    # Cliques covered by both
                    both = c1 & c2
                    # Cliques covered only by v1 (among selected)
                    only_v1 = c1 - c2
                    only_v2 = c2 - c1
                    # Check if all cliques in only_v1 are also covered by other selected vertices
                    other_selected = set(selected) - {v1, v2}
                    v1_redundant = True
                    for c_idx in only_v1:
                        if not (cliques[c_idx] & other_selected):
                            v1_redundant = False
                            break
                    v2_redundant = True
                    for c_idx in only_v2:
                        if not (cliques[c_idx] & other_selected):
                            v2_redundant = False
                            break
                    
                    if v1_redundant and v2_redundant:
                        # Both can be removed, but we want 2-for-1
                        # Find a vertex w that covers all cliques in only_v1 ∪ only_v2 ∪ both
                        needed = only_v1 | only_v2 | both
                        best_w = -1
                        best_gain = 0
                        for w in range(n):
                            if w == v1 or w == v2:
                                continue
                            covers = vert_to_cliques[w]
                            gain = len(needed & covers)
                            if gain > best_gain:
                                best_gain = gain
                                best_w = w
                        if best_gain == len(needed):
                            # w covers everything v1 and v2 cover that isn't covered by others
                            new_selected = [x for x in selected if x != v1 and x != v2]
                            new_selected.append(best_w)
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
        
        # Remove redundant again
        changed = True
        while changed:
            changed = False
            for i, v in enumerate(selected):
                remaining = set(selected)
                remaining.discard(v)
                all_covered = True
                for clique in cliques:
                    if not (clique & remaining):
                        all_covered = False
                        break
                if all_covered:
                    selected = [x for x in selected if x != v]
                    changed = True
                    break
        
        return selected
    
    def random_restart_improve(selected, cliques, n, budget=5.0):
        """Try random perturbations to escape local optima."""
        start_time = time.time()
        best = list(selected)
        best_size = len(best)
        
        vert_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vert_to_cliques[v].add(i)
        
        def is_valid(vert_set):
            for clique in cliques:
                if not (clique & vert_set):
                    return False
            return True
        
        def remove_redundant(vert_set):
            vert_set = list(vert_set)
            changed = True
            while changed:
                changed = False
                for i, v in enumerate(vert_set):
                    remaining = set(vert_set)
                    remaining.discard(v)
                    if is_valid(remaining):
                        vert_set = [x for x in vert_set if x != v]
                        changed = True
                        break
            return vert_set
        
        while time.time() - start_time < budget:
            # Random perturbation: remove 1-3 random vertices, add 1-3 random vertices
            current = list(best)
            num_remove = random.randint(1, min(3, len(current)))
            num_add = random.randint(1, 3)
            
            to_remove = random.sample(current, num_remove)
            candidate = [v for v in current if v not in to_remove]
            
            # Add random vertices
            for _ in range(num_add):
                v = random.randint(0, n - 1)
                if v not in candidate:
                    candidate.append(v)
            
            # Try to repair: add vertices to cover uncovered cliques
            candidate_set = set(candidate)
            for clique in cliques:
                if not (clique & candidate_set):
                    # Add a random vertex from this clique
                    v = random.choice(list(clique))
                    candidate_set.add(v)
            
            # Remove redundant
            candidate = remove_redundant(list(candidate_set))
            
            if len(candidate) < best_size:
                best = candidate
                best_size = len(candidate)
        
        return best
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Emergency: return greedy solutions for remaining
            for remaining_n, remaining_p, remaining_seed in instances[instances.index((n, p, seed)):]:
                edges = generate_graph(remaining_n, remaining_p, remaining_seed)
                cliques = bron_kerbosch(edges, remaining_n)
                sol = greedy_hitting_set(cliques, remaining_n)
                results.append(sorted(sol))
            break
        
        edges = generate_graph(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        
        # Initial greedy solution
        sol = greedy_hitting_set(cliques, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            sol = local_search_improve(sol, cliques, n)
        
        # Random restart improvement if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 15:
            budget = min(remaining_time - 5, 10.0)
            sol = random_restart_improve(sol, cliques, n, budget)
        
        results.append(sorted(sol))
    
    return results
