import time
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 256
    vals = list(range(2, N + 1))
    n = len(vals)
    weights = [1.0 / v for v in vals]
    
    # Build compatibility: comp[i][j] = True iff lcm(vals[i], vals[j]) > 256
    comp = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(vals[i], vals[j]) > N:
                comp[i][j] = True
                comp[j][i] = True
    
    # For each vertex, precompute the list of compatible vertices
    compat_list = []
    for i in range(n):
        compat_list.append([j for j in range(n) if j != i and comp[i][j]])
    
    # Greedy initial solution (by descending weight = ascending value)
    greedy_set = set()
    for i in range(n):
        if all(comp[i][j] for j in greedy_set):
            greedy_set.add(i)
    
    best_score = sum(weights[i] for i in greedy_set)
    best_set = set(greedy_set)
    
    # Exact MWIS via Bron-Kerbosch with pivot + coloring bound on compatibility graph
    # We seek max-weight clique in the compatibility graph.
    
    # Ordering vertices by descending weight for better coloring bounds
    # Vertex i has weight 1/(i+2), so higher index = lower weight.
    # We process in order of descending weight: indices 0, 1, 2, ..., n-1
    # (since vals = [2,3,4,...,256], index 0 = val 2 = highest weight)
    
    # For coloring bound: assign colors greedily.
    # In a clique, all vertices must be mutually adjacent.
    # A proper coloring of the candidate subgraph gives an upper bound on clique size,
    # but for weighted case we need a weighted coloring bound.
    
    # Weighted coloring bound: order candidates, assign colors such that
    # vertices of same color are non-adjacent. Then max weight in each color class
    # sums to an upper bound on the max-weight clique.
    
    # For efficiency, we use the standard BK with pivot and a simple upper bound.
    
    best = {'score': best_score, 'set': best_set}
    
    def bk(R, P, X):
        """Bron-Kerbosch with pivot selection for max-weight clique.
        R: current clique (set of vertex indices)
        P: candidates that can be added (set)
        X: excluded vertices (set)
        """
        if time.time() > deadline:
            return
        
        if not P and not X:
            # Found a maximal clique
            score = sum(weights[i] for i in R)
            if score > best['score']:
                best['score'] = score
                best['set'] = set(R)
            return
        
        if not P:
            return
        
        # Upper bound: current weight + max possible additional weight
        # Simple bound: sum of all weights in P (optimistic)
        # Better: use coloring-based bound
        # For now, use a simple but effective bound:
        # Sort P by descending weight, take prefix sums
        p_sorted = sorted(P, key=lambda v: -weights[v])
        # Weighted coloring bound
        # Assign colors greedily: two vertices get same color if not adjacent
        color = {}
        color_classes = []  # list of lists
        for v in p_sorted:
            placed = False
            for ci, cls in enumerate(color_classes):
                # v can go in class ci if v is not adjacent to any vertex in cls
                if all(not comp[v][u] for u in cls):
                    cls.append(v)
                    color[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                color[v] = len(color_classes) - 1
        
        # Upper bound: for each color class, the max weight vertex can be in the clique
        # (since at most one vertex per color class can be in a clique)
        ub = sum(weights[i] for i in R)
        for cls in color_classes:
            ub += max(weights[v] for v in cls)
        
        if ub <= best['score']:
            return
        
        # Pivot selection: choose u from P ∪ X that maximizes |P ∩ N(u)|
        pivot_candidates = P | X
        if pivot_candidates:
            # Pick pivot with max degree into P
            u = max(pivot_candidates, key=lambda x: sum(1 for y in P if comp[x][y]))
            # Candidates to branch on: P \ N(u)
            candidates = [v for v in P if not comp[v][u]]
        else:
            candidates = list(P)
        
        # Process candidates in order of descending weight
        candidates.sort(key=lambda v: -weights[v])
        
        for v in candidates:
            if time.time() > deadline:
                return
            R.add(v)
            P_new = P & set(compat_list[v])
            X_new = X & set(compat_list[v])
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    # Initial call
    bk(set(), set(range(n)), set())
    
    if time.time() > deadline - 2.0:
        # Safety: return what we have
        return [vals[i] for i in best['set']]
    
    # Verify and return
    result_set = best['set']
    if not result_set:
        result_set = greedy_set
    
    # Final verification
    valid = True
    result_list = list(result_set)
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = vals[result_list[i]], vals[result_list[j]]
            if lcm(a, b) <= N:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        result_list = list(greedy_set)
    
    return [vals[i] for i in result_list]
