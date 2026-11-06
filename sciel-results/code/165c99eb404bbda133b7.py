"""Combined program: Bron-Kerbosch clique enumeration + greedy set cover + aggressive local search with bitmask acceleration."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _build_bitmasks(n, cliques):
    """Build bitmask representation for fast validity checking."""
    # vertex_to_cliques_mask[v] = bitmask of cliques containing v
    m = len(cliques)
    vertex_cliques_mask = [0] * n
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques_mask[v] |= (1 << i)
    return vertex_cliques_mask, m

def _is_valid_fast(selected, vertex_cliques_mask, m):
    """Fast validity check using bitmasks."""
    if m == 0:
        return True
    covered = 0
    for v in selected:
        covered |= vertex_cliques_mask[v]
    # Check if all cliques are covered
    full_mask = (1 << m) - 1
    return (covered & full_mask) == full_mask

def _validate_solution(adj, selected, cliques):
    """Check if selected vertices meet every maximal clique of size >= 2."""
    selected_set = set(selected)
    for clique in cliques:
        if not (selected_set & clique):
            return False
    return True

def _greedy_set_cover(cliques):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    n = max(max(c) for c in cliques) + 1
    # Build vertex -> cliques mapping
    vertex_to_cliques = defaultdict(list)
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].append(i)
    
    covered = [False] * len(cliques)
    selected = []
    
    while any(not c for c in covered):
        best_v = None
        best_count = -1
        for v, clist in vertex_to_cliques.items():
            count = sum(1 for idx in clist if not covered[idx])
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.append(best_v)
        for idx in vertex_to_cliques[best_v]:
            covered[idx] = True
    
    return sorted(selected)

def _local_search_fast(adj, initial_solution, cliques, vertex_cliques_mask, m, max_iters=5000, time_budget=5.0):
    """Fast local search using bitmask validity checking."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if m == 0:
        return sorted(current)
    
    full_mask = (1 << m) - 1
    
    def is_valid(sol):
        covered = 0
        for v in sol:
            covered |= vertex_cliques_mask[v]
        return (covered & full_mask) == full_mask
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        
        # Try removing each vertex
        for i, v in enumerate(list(current)):
            trial = current[:i] + current[i+1:]
            if is_valid(trial):
                current = trial
                improved = True
                break
        if improved:
            continue
        
        # Try 1-out-1-in swaps
        current_set = set(current)
        all_vertices = set(range(n))
        for v in list(current):
            trial_without = current_set - {v}
            for u in all_vertices - current_set:
                trial = trial_without | {u}
                if is_valid(trial):
                    current = list(trial)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(current)

def _polish_search(adj, initial_solution, cliques, vertex_cliques_mask, m, time_budget=2.0):
    """Final polish: exhaustive search over small neighborhoods."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if m == 0:
        return sorted(current)
    
    full_mask = (1 << m) - 1
    
    def is_valid(sol):
        covered = 0
        for v in sol:
            covered |= vertex_cliques_mask[v]
        return (covered & full_mask) == full_mask
    
    # Try all pairs of removals
    current_set = set(current)
    for i in range(len(current)):
        if time.time() - start > time_budget:
            break
        for j in range(i + 1, len(current)):
            trial = list(current)
            # Remove j first (higher index), then i
            trial.pop(j)
            trial.pop(i)
            if is_valid(trial):
                current = trial
                current_set = set(current)
                break
        else:
            continue
        break
    
    # Try adding one vertex and removing two
    for u in range(n):
        if time.time() - start > time_budget:
            break
        if u in current_set:
            continue
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                trial = list(current)
                trial.pop(j)
                trial.pop(i)
                trial.append(u)
                if is_valid(trial):
                    current = trial
                    current_set = set(current)
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    return sorted(current)

def _random_local_search(adj, cliques, vertex_cliques_mask, m, time_budget=5.0):
    """Generate random candidate solutions and improve via local search."""
    start = time.time()
    n = len(adj)
    rng = random.Random(42)
    
    if m == 0:
        return []
    
    full_mask = (1 << m) - 1
    
    def is_valid(sol):
        covered = 0
        for v in sol:
            covered |= vertex_cliques_mask[v]
        return (covered & full_mask) == full_mask
    
    best = None
    best_size = float('inf')
    
    # First try greedy
    greedy_sol = _greedy_set_cover(cliques)
    if is_valid(greedy_sol):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Random restarts
    for _ in range(50):
        if time.time() - start > time_budget:
            break
        
        # Start with a random subset
        subset_size = rng.randint(0, min(n, best_size + 2 if best else 5))
        candidate = set(rng.sample(range(n), subset_size))
        
        # Greedily add vertices to cover all cliques
        while not is_valid(candidate):
            # Find uncovered cliques
            covered_mask = 0
            for v in candidate:
                covered_mask |= vertex_cliques_mask[v]
            uncovered_mask = full_mask & ~covered_mask
            
            if uncovered_mask == 0:
                break
            
            # Pick vertex that covers most uncovered cliques
            best_v = None
            best_gain = 0
            for v in range(n):
                if v in candidate:
                    continue
                gain = vertex_cliques_mask[v] & uncovered_mask
                gain_count = bin(gain).count('1')
                if gain_count > best_gain:
                    best_gain = gain_count
                    best_v = v
            if best_v is None or best_gain == 0:
                break
            candidate.add(best_v)
        
        if not is_valid(candidate):
            continue
        
        if len(candidate) < best_size:
            # Local search to improve
            improved = _local_search_fast(adj, list(candidate), cliques, vertex_cliques_mask, m, max_iters=300, time_budget=1.5)
            if is_valid(improved) and len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 15:
            # Fallback: use a simple heuristic
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        # Build bitmasks for fast validity checking
        vertex_cliques_mask, m = _build_bitmasks(n, cliques)
        
        # Try greedy first
        greedy_sol = _greedy_set_cover(cliques)
        if not _is_valid_fast(greedy_sol, vertex_cliques_mask, m):
            # Should not happen, but fallback
            touched = set()
            for clique in cliques:
                touched.update(clique)
            greedy_sol = sorted(touched)
        
        # Try local search / random restarts with remaining time
        remaining = deadline - time.time() - 10
        if remaining > 2:
            better = _random_local_search(adj, cliques, vertex_cliques_mask, m, time_budget=min(remaining, 30))
            if _is_valid_fast(better, vertex_cliques_mask, m) and len(better) < len(greedy_sol):
                greedy_sol = better
        
        # Final polish pass
        remaining = deadline - time.time() - 5
        if remaining > 1 and len(greedy_sol) > 0:
            polished = _polish_search(adj, greedy_sol, cliques, vertex_cliques_mask, m, time_budget=min(remaining, 3.0))
            if _is_valid_fast(polished, vertex_cliques_mask, m) and len(polished) < len(greedy_sol):
                greedy_sol = polished
        
        results.append(greedy_sol)
    
    return results

# EVOLVE-BLOCK-END
