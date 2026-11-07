import time
import random
import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

edge_list = edges

def check_validity(colors, K):
    """
    Returns True if coloring is proper and all bicolored subgraphs are forests.
    """
    for u, v in edge_list:
        if colors[u] == colors[v]:
            return False
    
    used_colors = set(colors)
    if len(used_colors) < 2:
        return True
        
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        return False
    return True

def get_cost(colors, K):
    """
    Returns cost. 0 is valid. Uses fast bicolored-cycle detection.
    """
    # Check properness
    cost = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            cost += 100
    
    if cost > 0:
        return cost
        
    used_colors = set(colors)
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        cost += 10
                        break
    return cost

def backtracking_search(K, time_limit):
    """
    Backtracking search with forward checking for the no-bicolored-cycle constraint.
    We track, for each color pair (a,b), a UnionFind structure over vertices colored a or b.
    When we assign vertex v color c, for each neighbor u with color d != c, we check
    if adding edge (u,v) to the (c,d) pair's UF would create a cycle.
    """
    start_time = time.time()
    
    # Precompute neighbor lists
    neighbors = [adj[i] for i in range(N)]
    
    # Sort vertices by degree (descending) for better pruning
    vertex_order = sorted(range(N), key=lambda x: -len(neighbors[x]))
    
    # Map from vertex to its position in the ordering
    pos_in_order = [0] * N
    for idx, v in enumerate(vertex_order):
        pos_in_order[v] = idx
    
    # For each color pair (a,b), maintain a UnionFind
    # We'll use a dict: (min(a,b), max(a,b)) -> UnionFind
    # But for speed, let's pre-allocate for K colors
    # Actually, let's just use a dict and create UFs as needed
    
    class PairUF:
        def __init__(self):
            self.parent = {}
            self.rank = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    # colors[v] = -1 means unassigned
    colors = [-1] * N
    # pair_ufs[(a,b)] where a < b
    pair_ufs = {}
    
    def get_pair_uf(a, b):
        key = (min(a,b), max(a,b))
        if key not in pair_ufs:
            pair_ufs[key] = PairUF()
        return pair_ufs[key]
    
    def is_valid_assignment(v, c):
        """Check if assigning color c to vertex v is valid given current assignments."""
        # Check proper coloring: no neighbor has color c
        for u in neighbors[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle constraint
        for u in neighbors[v]:
            cu = colors[u]
            if cu == -1:
                continue
            if cu == c:
                # Already checked above
                return False
            # Check if adding (v, u) to pair (c, cu) creates a cycle
            uf = get_pair_uf(c, cu)
            if not uf.union(v, u):
                return False
        return True
    
    def backtrack(idx):
        """Assign color to vertex_order[idx]. Returns True if success."""
        if time.time() - start_time >= time_limit:
            return -1  # timeout
        
        if idx == N:
            return 1  # success
        
        v = vertex_order[idx]
        
        # Try each color
        for c in range(K):
            if is_valid_assignment(v, c):
                # Assign
                colors[v] = c
                # Record UF states for undo
                uf_changes = []
                for u in neighbors[v]:
                    cu = colors[u]
                    if cu == -1 or cu == c:
                        continue
                    key = (min(c, cu), max(c, cu))
                    uf = pair_ufs[key]
                    # Save state before union
                    rx = uf.find(v)
                    ry = uf.find(u)
                    if rx != ry:
                        uf_changes.append((key, v, u, rx, ry, uf.parent.get(ry), uf.rank.get(rx), uf.rank.get(ry)))
                
                result = backtrack(idx + 1)
                if result == 1:
                    return 1
                elif result == -1:
                    # Undo before returning timeout
                    for (key, vv, uu, rx, ry, parent_ry, rank_rx, rank_ry) in uf_changes:
                        uf = pair_ufs[key]
                        if parent_ry is not None:
                            uf.parent[ry] = parent_ry
                        if rank_rx is not None:
                            uf.rank[rx] = rank_rx
                        if rank_ry is not None:
                            uf.rank[ry] = rank_ry
                    colors[v] = -1
                    return -1
                
                # Undo
                for (key, vv, uu, rx, ry, parent_ry, rank_rx, rank_ry) in reversed(uf_changes):
                    uf = pair_ufs[key]
                    if parent_ry is not None:
                        uf.parent[ry] = parent_ry
                    if rank_rx is not None:
                        uf.rank[rx] = rank_rx
                    if rank_ry is not None:
                        uf.rank[ry] = rank_ry
                colors[v] = -1
        
        return 0
    
    result = backtrack(0)
    if result == 1:
        return list(colors)
    return None

def local_search(K, time_limit):
    start_time = time.time()
    best_cost = float('inf')
    best_coloring = None
    n_restarts = 0
    max_restarts = 5000
    
    # Generate structured initializations based on modular patterns
    def get_structured_init(K):
        """Try various structured colorings that exploit the circulant structure."""
        inits = []
        # Modular patterns: color[i] = (a*i + b) % K
        for a in range(1, K):
            for b in range(K):
                colors = [(a * i + b) % K for i in range(N)]
                inits.append(colors)
        # Alternating patterns
        for period in range(2, K+1):
            for offset in range(period):
                colors = [(i % period + offset) % K for i in range(N)]
                inits.append(colors)
        # Random
        inits.append([random.randint(0, K - 1) for _ in range(N)])
        return inits
    
    structured_inits = get_structured_init(K)
    
    # First try structured inits
    for init_colors in structured_inits:
        if time.time() - start_time >= time_limit:
            break
        current_cost = get_cost(init_colors, K)
        if current_cost == 0:
            return list(init_colors)
        if current_cost < best_cost:
            best_cost = current_cost
            best_coloring = list(init_colors)
    
    # Now do local search from best structured init or random
    if best_coloring is not None:
        colors = list(best_coloring)
        current_cost = best_cost
    else:
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_cost = get_cost(colors, K)
    
    while time.time() - start_time < time_limit and n_restarts < max_restarts:
        n_restarts += 1
        
        if current_cost == 0:
            return colors
            
        no_improve = 0
        max_no_improve = 200
        
        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            if current_cost == 0:
                return colors
                
            # Pick random vertex
            v = random.randint(0, N - 1)
            old_c = colors[v]
            
            best_c = old_c
            min_c = current_cost
            
            for c in range(K):
                if c == old_c: continue
                colors[v] = c
                c_cost = get_cost(colors, K)
                if c_cost < min_c:
                    min_c = c_cost
                    best_c = c
            colors[v] = best_c
            current_cost = min_c
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
                
            if current_cost == 0:
                return colors
                
            if current_cost == min_c and best_c == old_c:
                no_improve += 1
            else:
                no_improve = 0
        
        # Restart with new random or perturbed init
        if random.random() < 0.3 and best_coloring is not None:
            # Perturb best coloring
            colors = list(best_coloring)
            for _ in range(random.randint(1, 5)):
                v = random.randint(0, N - 1)
                colors[v] = random.randint(0, K - 1)
        else:
            colors = [random.randint(0, K - 1) for _ in range(N)]
        current_cost = get_cost(colors, K)
        
        if current_cost == 0:
            return colors
    
    if best_cost == 0 and best_coloring:
        return best_coloring
    return None

def construct():
    deadline = time.time() + 55.0
    
    # Try backtracking first for K=3, 4, 5 as it's more systematic
    for K in [3, 4, 5, 6]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        t_alloc = min(15.0, remaining - 3.0)
        res = backtracking_search(K, t_alloc)
        if res is not None:
            return res
    
    # Fall back to local search
    remaining = deadline - time.time()
    if remaining > 10:
        res = local_search(4, min(20.0, remaining - 5.0))
        if res is not None:
            return res
    
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(5, min(10.0, remaining - 3.0))
        if res is not None:
            return res
    
    # Fallback: use a valid 4-coloring pattern if found, else identity
    colors4 = [i % 4 for i in range(N)]
    if check_validity(colors4, 4):
        return colors4
    
    return list(range(N))

# EVOLVE-BLOCK-END
