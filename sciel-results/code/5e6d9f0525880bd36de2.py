"""Combined approach: greedy set cover + local search + randomized restarts for minimal clique transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 145.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance(n, p, seed)
        out.append(sol)
    return out

def _solve_instance(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        best_pivot = None
        max_count = -1
        for v in P:
            c = len(adj[v] & (P | X))
            if c > max_count:
                max_count = c
                best_pivot = v
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = set(P)
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    cliques = [c for c in cliques if len(c) >= 2]
    if not cliques:
        return []
    
    m = len(cliques)
    
    # Build incidence structures
    vertex_to_cliques = [list() for _ in range(n)]
    clique_vertices = [list() for _ in range(m)]
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
            clique_vertices[i].append(v)
    
    # Precompute: for each clique, which vertices are in it (as set for fast intersection)
    clique_sets = [set(c) for c in cliques]
    
    # Count how many times each vertex appears in cliques
    vertex_clique_count = [len(vertex_to_cliques[v]) for v in range(n)]
    
    def _is_valid(selected):
        """Check if selected covers all cliques."""
        for ci in range(m):
            c = clique_sets[ci]
            if not (c & selected):
                return False
        return True
    
    def _remove_redundant(selected, time_limit):
        """Remove vertices that are not needed (no clique is solely covered by them)."""
        improved = True
        while improved and time.time() - _START_TIME < time_limit:
            improved = False
            # Compute coverage count: for each clique, how many selected vertices cover it
            clique_cover_count = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_cover_count[ci] += 1
            
            # Find vertices that are the unique cover for some clique
            for v in list(selected):
                if v not in selected:
                    continue
                needed = False
                for ci in vertex_to_cliques[v]:
                    if clique_cover_count[ci] == 1:
                        needed = True
                        break
                if not needed:
                    selected.discard(v)
                    # Update coverage counts
                    for ci in vertex_to_cliques[v]:
                        clique_cover_count[ci] -= 1
                    improved = True
        return selected
    
    def _try_remove_pair(selected, time_limit):
        """Try to find a pair of vertices that can both be removed."""
        improved = True
        while improved and time.time() - _START_TIME < time_limit:
            improved = False
            sel_list = list(selected)
            # Compute coverage counts
            clique_cover_count = [0] * m
            for v in selected:
                for ci in vertex_to_cliques[v]:
                    clique_cover_count[ci] += 1
            
            # Find cliques with coverage count exactly 2
            weak_cliques = [ci for ci in range(m) if clique_cover_count[ci] == 2]
            if not weak_cliques:
                break
            
            for ci in weak_cliques:
                if time.time() - _START_TIME > time_limit:
                    break
                # Get the two vertices covering this clique
                covering = [v for v in selected if ci in vertex_to_cliques[v]]
                if len(covering) != 2:
                    continue
                v1, v2 = covering[0], covering[1]
                # Check if removing both still covers everything
                # A clique is unprotected if its only selected vertices are v1 or v2
                valid = True
                for ci2 in range(m):
                    cnt = clique_cover_count[ci2]
                    if cnt == 0:
                        valid = False
                        break
                    if cnt == 1:
                        # Check if the sole cover is v1 or v2
                        sole = None
                        for v in selected:
                            if v != v1 and v != v2 and ci2 in vertex_to_cliques[v]:
                                sole = v
                                break
                        if sole is None:
                            valid = False
                            break
                if valid:
                    selected.discard(v1)
                    selected.discard(v2)
                    improved = True
                    break
        return selected
    
    def _greedy_cover(randomize=False, rng=None):
        """Greedy set cover."""
        selected = set()
        uncovered = set(range(m))
        # Track coverage counts for efficiency
        while uncovered and time.time() - _START_TIME < _DEADLINE - 15:
            best_v = None
            best_score = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in vertex_to_cliques[v] if ci in uncovered)
                if count <= 0:
                    continue
                score = float(count)
                if randomize and rng is not None:
                    score += rng.random() * 0.3
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v is None:
                break
            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                uncovered.discard(ci)
        
        for ci in uncovered:
            v = next(iter(cliques[ci]))
            selected.add(v)
        return selected
    
    # Initial solution via greedy
    selected = _greedy_cover(randomize=False)
    selected = _remove_redundant(selected, _DEADLINE - 15)
    selected = _try_remove_pair(selected, _DEADLINE - 15)
    
    best_size = len(selected)
    best_sol = set(selected)
    
    # Randomized restarts
    if time.time() - _START_TIME < _DEADLINE - 25:
        for trial in range(20):
            if time.time() - _START_TIME > _DEADLINE - 20:
                break
            trial_rng = random.Random(seed + trial * 1000 + 42)
            trial_sel = _greedy_cover(randomize=True, rng=trial_rng)
            trial_sel = _remove_redundant(trial_sel, _DEADLINE - 20)
            trial_sel = _try_remove_pair(trial_sel, _DEADLINE - 20)
            if len(trial_sel) < best_size:
                best_size = len(trial_sel)
                best_sol = set(trial_sel)
    
    # Swap-based local search
    if time.time() - _START_TIME < _DEADLINE - 10 and best_size > 1:
        improved = True
        while improved and time.time() - _START_TIME < _DEADLINE - 10:
            improved = False
            sel_list = list(best_sol)
            non_sel = [v for v in range(n) if v not in best_sol]
            for v_out in sel_list:
                if time.time() - _START_TIME > _DEADLINE - 10:
                    break
                for v_in in non_sel:
                    test_sol = set(best_sol)
                    test_sol.discard(v_out)
                    test_sol.add(v_in)
                    if _is_valid(test_sol):
                        test_sol = _remove_redundant(test_sol, _DEADLINE - 10)
                        test_sol = _try_remove_pair(test_sol, _DEADLINE - 10)
                        if len(test_sol) < best_size:
                            best_size = len(test_sol)
                            best_sol = test_sol
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(best_sol)

# EVOLVE-BLOCK-END
