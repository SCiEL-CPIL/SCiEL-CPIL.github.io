import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80
    
    LIMIT = 512
    N = LIMIT
    
    # Precompute compatibility
    compat = [0] * (N + 1)
    neighbors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
                if i != j:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
    
    # Precompute neighbor sets for O(1) lookup
    neighbor_sets = [set(neighbors[i]) for i in range(N + 1)]
    degrees = [len(neighbors[i]) for i in range(N + 1)]
    
    # Greedy initial solution with multiple orderings
    def greedy_clique(order):
        S = []
        for v in order:
            ok = True
            for c in S:
                if v not in neighbor_sets[c]:
                    ok = False
                    break
            if ok:
                S.append(v)
        return S
    
    best_clique = []
    best_size = 0
    
    # Try multiple orderings
    order1 = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    order2 = sorted(range(1, N + 1), key=lambda x: x)
    order3 = sorted(range(1, N + 1), key=lambda x: -degrees[x] / (x + 1))
    
    for order in [order1, order2, order3]:
        S = greedy_clique(order)
        if len(S) > best_size:
            best_clique = S
            best_size = len(S)
    
    sys.setrecursionlimit(10000)
    
    def expand(P, C, depth):
        nonlocal best_clique, best_size
        
        if time.time() > DEADLINE:
            return
        
        if not P:
            if depth > best_size:
                best_clique = C[:]
                best_size = depth
            return
        
        if depth + len(P) <= best_size:
            return
        
        # Greedy coloring - optimized: track colored vertices in order
        P_set = set(P)
        P_sorted = sorted(P, key=lambda v: -len(neighbor_sets[v] & P_set))
        
        color_of = {}
        color_list = []
        colored = []
        
        for v in P_sorted:
            # Find smallest color not used by any already-colored neighbor of v
            used_colors = set()
            for u in colored:
                if u in neighbor_sets[v]:
                    used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c >= len(color_list):
                color_list.append([])
            color_list[c].append(v)
            colored.append(v)
        
        max_color = len(color_list)
        
        if depth + max_color <= best_size:
            return
        
        # Branch on vertices in reverse color order
        for k in range(max_color - 1, -1, -1):
            for v in color_list[k]:
                if v not in P_set:
                    continue
                
                C.append(v)
                new_P = [u for u in P_set if u != v and u in neighbor_sets[v]]
                expand(new_P, C, depth + 1)
                C.pop()
                
                P_set.remove(v)
                
                if time.time() > DEADLINE:
                    return
    
    # Try multiple initial orderings for the branch-and-bound
    order1 = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    order2 = sorted(range(1, N + 1), key=lambda x: -degrees[x] / (x + 1))
    
    for initial_P in [order1, order2]:
        if time.time() > DEADLINE - 5:
            break
        expand(initial_P, [], 0)
    
    if not best_clique:
        return [1]
    
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
