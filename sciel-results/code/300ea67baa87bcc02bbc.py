"""Rewritten program: targeted search for minimal-color valid coloring."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Precompute for fast bicolored-cycle check
# For a given coloring, check if any pair of colors forms a cycle
def has_bicolored_cycle(coloring, k):
    """Check if any pair of colors induces a subgraph with a cycle using union-find."""
    # Group vertices by color
    color_to_verts = defaultdict(list)
    for i in range(N):
        color_to_verts[coloring[i]].append(i)
    
    for ci in range(k):
        verts_i = color_to_verts.get(ci, [])
        if len(verts_i) < 3:
            continue
        for cj in range(ci + 1, k):
            verts_j = color_to_verts.get(cj, [])
            if len(verts_j) < 3:
                continue
            # Build subgraph on verts_i ∪ verts_j
            all_verts = set(verts_i) | set(verts_j)
            if len(all_verts) < 3:
                continue
            # Union-find on this subgraph
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False  # cycle detected
                parent[ra] = rb
                return True
            # Process edges within the subgraph
            for v in all_verts:
                find(v)
            for u in all_verts:
                for w in ADJ[u]:
                    if w in all_verts and u < w:
                        if not union(u, w):
                            return True
    return False

def is_valid(coloring, k):
    """Check proper coloring and no bicolored cycles."""
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    return not has_bicolored_cycle(coloring, k)

def count_distinct(coloring):
    return len(set(coloring))

# Backtracking search with forward checking
def backtrack_search(k, deadline):
    """Try to find a valid k-coloring via backtracking."""
    coloring = [-1] * N
    # Order vertices by degree (all degree 4 here, so by index)
    order = list(range(N))
    
    def can_assign(v, c):
        # Check proper coloring
        for u in ADJ[v]:
            if coloring[u] == c:
                return False
        return True
    
    def search(idx):
        if time.time() > deadline:
            return None
        if idx == N:
            if is_valid(coloring, k):
                return coloring[:]
            return None
        v = order[idx]
        for c in range(k):
            if can_assign(v, c):
                coloring[v] = c
                result = search(idx + 1)
                if result is not None:
                    return result
                coloring[v] = -1
        return None
    
    return search(0)

# Local search / simulated annealing
def local_search(k, deadline, seed=42):
    """Simulated annealing to find valid k-coloring."""
    rng = random.Random(seed)
    # Initial random proper coloring attempt
    best = None
    best_cost = float('inf')
    
    def cost(coloring):
        c = 0
        for u, v in EDGES:
            if coloring[u] == coloring[v]:
                c += 1000
        if has_bicolored_cycle(coloring, k):
            c += 10000
        return c
    
    coloring = [rng.randint(0, k - 1) for _ in range(N)]
    # Try to make it proper first via greedy repair
    for _ in range(100):
        improved = False
        for v in range(N):
            bad = False
            for u in ADJ[v]:
                if coloring[u] == coloring[v]:
                    bad = True
                    break
            if bad:
                for c in range(k):
                    if all(coloring[u] != c for u in ADJ[v]):
                        coloring[v] = c
                        improved = True
                        break
        if not improved:
            break
    
    cur_cost = cost(coloring)
    if cur_cost == 0:
        return coloring
    
    T = 100.0
    cooling = 0.9995
    for step in range(200000):
        if time.time() > deadline:
            break
        if cur_cost == 0:
            return coloring[:]
        if cur_cost < best_cost:
            best_cost = cur_cost
            best = coloring[:]
        
        v = rng.randint(0, N - 1)
        old_c = coloring[v]
        new_c = rng.randint(0, k - 1)
        if new_c == old_c:
            continue
        coloring[v] = new_c
        new_cost = cost(coloring)
        delta = new_cost - cur_cost
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.1)):
            cur_cost = new_cost
        else:
            coloring[v] = old_c
        T *= cooling
    
    if best is not None and best_cost == 0:
        return best
    return None

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s
    
    # Try k from 2 upward
    for k in range(2, 10):
        if time.time() > deadline:
            break
        
        # Try backtracking first (fast for small k if solvable)
        result = backtrack_search(k, deadline)
        if result is not None:
            return [int(x) for x in result]
        
        # Try local search
        for seed in [42, 123, 777, 2024, 999]:
            if time.time() > deadline:
                break
            result = local_search(k, deadline, seed=seed)
            if result is not None:
                # Verify
                if is_valid(result, k):
                    return [int(x) for x in result]
    
    # Fallback: use all distinct colors (always valid)
    return [int(i) for i in range(N)]
# EVOLVE-BLOCK-END
