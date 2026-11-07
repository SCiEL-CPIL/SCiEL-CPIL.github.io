"""Minimum set cover over maximal bicliques with exact backtracking."""
import time
import sys
from itertools import combinations

sys.setrecursionlimit(100000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _validate(bicliques, adj, all_edges_set):
    n = P['n']
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            if u < 0 or u >= n or not isinstance(u, int):
                return False
            for v in right:
                if v < 0 or v >= n or not isinstance(v, int):
                    return False
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True


def _biclique_edge_set(left, right):
    s = set()
    for u in left:
        for v in right:
            s.add((min(u, v), max(u, v)))
    return s


def _enumerate_maximal_bicliques(adj, n, deadline):
    """Enumerate maximal bicliques using Bron-Kerbosch on the bipartite conflict graph.
    
    A biclique (L, R) is maximal if no vertex can be added to L or R while maintaining
    the complete bipartite property. We use a recursive approach:
    - Start with all possible (u, v) pairs that are edges
    - For each, try to expand to maximal
    - Use a visited set to avoid duplicates
    """
    maximal = []
    seen = set()  # frozenset of (frozen L, frozen R) to avoid duplicates
    
    # For each edge, try to expand to a maximal biclique
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # Start with {u} on left, {v} on right
            L = {u}
            R = {v}
            
            # Expand iteratively
            changed = True
            while changed:
                changed = False
                # Add to L: vertices connected to all in R
                for x in range(n):
                    if x in L or x in R:
                        continue
                    if R.issubset(adj[x]):
                        L.add(x)
                        changed = True
                # Add to R: vertices connected to all in L
                for x in range(n):
                    if x in L or x in R:
                        continue
                    if L.issubset(adj[x]):
                        R.add(x)
                        changed = True
            
            # Ensure disjoint
            R -= L
            L -= R
            if not L or not R:
                continue
            
            # Check if this is maximal (no further expansion possible)
            # It should be by construction, but verify
            is_maximal = True
            for x in range(n):
                if x in L or x in R:
                    continue
                if R.issubset(adj[x]):
                    is_maximal = False
                    break
                if L.issubset(adj[x]):
                    is_maximal = False
                    break
            
            if not is_maximal:
                continue
            
            # Normalize: sort to create canonical form
            L_frozen = frozenset(L)
            R_frozen = frozenset(R)
            # Two representations: (L, R) and (R, L) are the same biclique
            key = min((L_frozen, R_frozen), (R_frozen, L_frozen))
            if key in seen:
                continue
            seen.add(key)
            
            # Store with L as the smaller side for consistency
            if len(L) <= len(R):
                maximal.append((L, R))
            else:
                maximal.append((R, L))
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return maximal


def _greedy_set_cover(bicliques, edge_to_bicliques, all_edges_set, deadline):
    """Greedy set cover: repeatedly pick the biclique covering the most uncovered edges."""
    remaining = set(all_edges_set)
    chosen = []
    
    # Precompute edge counts for each biclique
    biclique_edges = []
    for L, R in bicliques:
        biclique_edges.append(_biclique_edge_set(L, R))
    
    while remaining:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = 0
        for i in range(len(bicliques)):
            if not biclique_edges[i]:
                continue
            count = len(biclique_edges[i] & remaining)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            # Fallback: single edge
            e = remaining.pop()
            chosen.append((frozenset([e[0]]), frozenset([e[1]])))
            continue
        
        chosen.append((bicliques[best_idx][0], bicliques[best_idx][1]))
        remaining -= biclique_edges[best_idx]
    
    return chosen


def _backtracking_set_cover(bicliques, edge_to_bicliques, all_edges_set, deadline, current_best):
    """Exact backtracking search for minimum set cover.
    
    Branches on the uncovered edge with the fewest candidate bicliques.
    Uses memoization to prune redundant states.
    """
    best = [current_best]
    best_solution = [None]
    memo = {}
    start_time = time.time()
    
    def search(remaining, chosen_count, chosen_indices):
        if time.time() > deadline:
            return
        if chosen_count >= best[0]:
            return
        
        if not remaining:
            if chosen_count < best[0]:
                best[0] = chosen_count
                best_solution[0] = list(chosen_indices)
            return
        
        # Memoization: if we've seen this state with fewer or equal chosen, skip
        state_key = frozenset(remaining)
        if state_key in memo:
            if memo[state_key] <= chosen_count:
                return
        else:
            memo[state_key] = chosen_count
        
        # Pick the edge with fewest candidate bicliques
        min_edge = None
        min_candidates = None
        min_count = float('inf')
        for e in remaining:
            candidates = edge_to_bicliques[e]
            # Only consider bicliques that cover e and are still "available"
            # (in this formulation, all bicliques are always available)
            count = len(candidates)
            if count < min_count:
                min_count = count
                min_edge = e
                min_candidates = candidates
        
        if min_count == 0:
            return  # Dead end
        
        # Branch on each candidate biclique
        for bi in min_candidates:
            L, R = bicliques[bi]
            new_remaining = remaining - _biclique_edge_set(L, R)
            search(new_remaining, chosen_count + 1, chosen_indices + [bi])
    
    search(set(all_edges_set), 0, tuple())
    
    if best_solution[0] is not None:
        return [bicliques[i] for i in best_solution[0]]
    return None


def _hybrid_search(bicliques, edge_to_bicliques, all_edges_set, deadline):
    """Combine greedy with limited backtracking to find a good solution."""
    # First, get a greedy solution as upper bound
    greedy_result = _greedy_set_cover(bicliques, edge_to_bicliques, all_edges_set, deadline)
    if greedy_result is None:
        return None
    upper_bound = len(greedy_result)
    
    # Try backtracking with the greedy upper bound
    bt_result = _backtracking_set_cover(bicliques, edge_to_bicliques, all_edges_set, deadline, upper_bound)
    if bt_result is not None and len(bt_result) < upper_bound:
        return bt_result
    return greedy_result


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Phase 1: Enumerate all maximal bicliques
    maximal_bicliques = _enumerate_maximal_bicliques(adj, n, deadline)
    
    if not maximal_bicliques:
        # Fallback: single edges
        return [[u, v] for u, v in P['edges']]
    
    # Phase 2: Build edge-to-biclique mapping
    edge_to_bicliques = {}
    for e in all_edges_set:
        edge_to_bicliques[e] = []
    
    for i, (L, R) in enumerate(maximal_bicliques):
        edges = _biclique_edge_set(L, R)
        for e in edges:
            if e in all_edges_set:
                edge_to_bicliques[e].append(i)
    
    # Phase 3: Solve set cover
    result = _hybrid_search(maximal_bicliques, edge_to_bicliques, all_edges_set, deadline)
    
    if result is None or not result:
        # Fallback: greedy
        result = _greedy_set_cover(maximal_bicliques, edge_to_bicliques, all_edges_set, deadline)
    
    if result is None or not result:
        # Final fallback: single edges
        result = [(frozenset([u]), frozenset([v])) for u, v in P['edges']]
    
    # Convert to proper types: list of [list(left), list(right)]
    final_result = []
    for L, R in result:
        final_result.append([list(L), list(R)])
    
    return final_result
