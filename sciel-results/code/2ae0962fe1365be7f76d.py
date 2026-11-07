# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_n = P['n']
_edges = P['edges']
edge_set = set()
adj = defaultdict(set)
for u, v in _edges:
    key = (min(u, v), max(u, v))
    edge_set.add(key)
    adj[u].add(v)
    adj[v].add(u)

all_edges = list(edge_set)


def _maximal_bicliques():
    """Find all maximal bicliques in the graph using common-neighborhood closure."""
    bicliques = []
    seen = set()
    for (u, v) in all_edges:
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # Build biclique: L = {u} ∪ (common neighbors connected to v), R = {v} ∪ (common neighbors connected to u)
        # Actually, for a biclique with u in L and v in R:
        # L = {u} ∪ {w : w connected to all of R}
        # R = {v} ∪ {w : w connected to all of L}
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L: vertices connected to all of R
            new_L = set(L)
            for rv in R:
                new_L |= adj[rv]
            L_new = {x for x in new_L if all(y in adj[x] for y in R)}
            if L_new != L:
                L = L_new
                changed = True
            # Expand R: vertices connected to all of L
            new_R = set(R)
            for lu in L:
                new_R |= adj[lu]
            R_new = {x for x in new_R if all(y in adj[x] for y in L)}
            if R_new != R:
                R = R_new
                changed = True
        
        # Verify all cross edges exist
        valid = True
        for lu in L:
            for rv in R:
                key = (min(lu, rv), max(lu, rv))
                if key not in edge_set:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or not L or not R:
            continue
        
        # Normalize for dedup
        L_frozenset = frozenset(L)
        R_frozenset = frozenset(R)
        # Also consider the swap (L,R) vs (R,L) - they represent the same biclique
        key1 = (L_frozenset, R_frozenset)
        key2 = (R_frozenset, L_frozenset)
        if key1 not in seen and key2 not in seen:
            seen.add(key1)
            seen.add(key2)
            bicliques.append((sorted(L), sorted(R)))
    
    return bicliques


def _edge_set_of_biclique(L, R):
    """Return the set of edges covered by biclique (L, R)."""
    edges = set()
    for u in L:
        for v in R:
            key = (min(u, v), max(u, v))
            edges.add(key)
    return edges


def _greedy_set_cover(candidates, target_edges):
    """Greedy set cover: pick the biclique that covers the most uncovered edges."""
    uncovered = set(target_edges)
    selected = []
    
    while uncovered and candidates:
        best_idx = -1
        best_count = 0
        for i, (L, R) in enumerate(candidates):
            if i in selected:
                continue
            covered = _edge_set_of_biclique(L, R)
            new_cover = covered & uncovered
            if len(new_cover) > best_count:
                best_count = len(new_cover)
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            break
        
        L, R = candidates[best_idx]
        selected.append(best_idx)
        covered = _edge_set_of_biclique(L, R)
        uncovered -= covered
    
    return selected, uncovered


def _local_search_refine(selected_indices, candidates, target_edges, deadline):
    """Try to improve the set cover by swapping or replacing bicliques."""
    selected = list(selected_indices)
    
    def _is_valid(sel):
        covered = set()
        for idx in sel:
            L, R = candidates[idx]
            covered |= _edge_set_of_biclique(L, R)
        return covered == set(target_edges)
    
    # Check initial validity
    if not _is_valid(selected):
        return selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each selected biclique and see if still valid
        for i in range(len(selected)):
            trial = selected[:i] + selected[i+1:]
            if _is_valid(trial):
                selected = trial
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing one biclique with another
        for i in range(len(selected)):
            for j in range(len(candidates)):
                if j in selected:
                    continue
                trial = selected[:i] + [j] + selected[i+1:]
                if _is_valid(trial):
                    # Only accept if it reduces size or same size but different (to explore)
                    if len(trial) < len(selected):
                        selected = trial
                        improved = True
                        break
            if improved:
                break
    
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s buffer
    
    # Fallback: star decomposition
    fallback = []
    for u in range(_n):
        neighbors = [v for (a, b) in edge_set if a == u or b == u]
        if neighbors:
            # Ensure u is not in neighbors for the biclique
            fallback.append([[int(u)], [int(v) for v in neighbors if v != u]])
    
    # Find all maximal bicliques
    bicliques = _maximal_bicliques()
    
    if not bicliques:
        return fallback
    
    # Greedy set cover
    selected, uncovered = _greedy_set_cover(bicliques, set(all_edges))
    
    if uncovered:
        # Some edges not covered; add them as single-edge bicliques
        for (u, v) in uncovered:
            bicliques.append([int(u), int(v)])
            selected.append(len(bicliques) - 1)
    
    # Local search refinement
    selected = _local_search_refine(selected, bicliques, set(all_edges), deadline - 5)
    
    # Build result
    result = []
    for idx in selected:
        L, R = bicliques[idx]
        result.append([[int(x) for x in L], [int(y) for y in R]])
    
    # Final validation
    if result:
        seen = set()
        valid = True
        for L, R in result:
            if not L or not R:
                valid = False
                break
            for u in L:
                for v in R:
                    key = (min(u, v), max(u, v))
                    if key not in edge_set:
                        valid = False
                        break
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
            if not valid:
                break
        
        if valid and seen == set(all_edges):
            return result
    
    # Fallback
    return fallback
# EVOLVE-BLOCK-END
