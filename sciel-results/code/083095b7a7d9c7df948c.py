import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def construct():
    deadline = time.time() + 85.0
    
    # Try K=3 first with backtracking + forward checking
    for K in [3, 4, 5, 6]:
        if time.time() > deadline - 5.0:
            break
        result = _solve_backtrack(K, deadline)
        if result is not None:
            return result
    
    # Fallback: greedy with many colors
    return list(range(N))

def _solve_backtrack(K, deadline):
    """
    Backtracking search with forward checking for bicolored cycle constraint.
    We maintain:
      - colors: list of assigned colors (or -1)
      - For each color pair (a,b), a Union-Find structure tracking connectivity
        in the subgraph induced by vertices colored a or b.
    
    Forward checking: when assigning vertex v color c, check all neighbors.
    For each neighbor u with color d != c, the edge (v,u) is in the bicolored
    subgraph for pair (c,d). We must ensure this doesn't create a cycle.
    """
    # Order vertices by degree (descending) for better pruning
    vertex_order = sorted(range(N), key=lambda v: len(adj[v]), reverse=True)
    
    # For each color pair, maintain a DSU
    # color_pair_dsu[(a,b)] = parent array for DSU (only for used pairs)
    # We'll use a dict mapping (min(a,b), max(a,b)) -> list parent
    
    colors = [-1] * N
    # DSU per color pair: key is (a,b) with a < b
    pair_dsu = {}  # (a,b) -> list of parents (size N)
    
    def dsu_find(dsu, x):
        if dsu[x] != x:
            dsu[x] = dsu_find(dsu, dsu[x])
        return dsu[x]
    
    def dsu_union(dsu, x, y):
        rx = dsu_find(dsu, x)
        ry = dsu_find(dsu, y)
        if rx == ry:
            return False  # would create cycle
        dsu[ry] = rx
        return True
    
    def get_or_init_dsu(a, b):
        if a > b:
            a, b = b, a
        key = (a, b)
        if key not in pair_dsu:
            pair_dsu[key] = list(range(N))
        return pair_dsu[key]
    
    def check_and_assign(v, c):
        """
        Check if assigning color c to vertex v is valid.
        Returns True if valid, and updates DSUs.
        If invalid, returns False.
        We need to be careful: we check all edges from v to already-colored neighbors.
        """
        # Check proper coloring: no neighbor has same color
        for u in adj[v]:
            if colors[u] == c:
                return None  # invalid
        
        # For each colored neighbor u, check bicolored cycle constraint
        # Edge (v, u) with colors (c, colors[u]) must not create a cycle
        checks = []
        for u in adj[v]:
            cu = colors[u]
            if cu == -1 or cu == c:
                continue
            a, b = min(c, cu), max(c, cu)
            dsu = get_or_init_dsu(a, b)
            # Check if v and u are already connected in this DSU
            rv = dsu_find(dsu, v)
            ru = dsu_find(dsu, u)
            if rv == ru:
                # Would create a cycle - invalid
                return None
            checks.append((dsu, v, u))
        
        # All checks passed, now actually union
        for dsu, v2, u2 in checks:
            dsu_union(dsu, v2, u2)
        
        return True
    
    def unassign(v, c):
        """
        Undo the assignment of color c to vertex v.
        We need to revert the DSU unions. This is tricky with DSU.
        Instead, we'll use a different approach: copy-on-write for small DSUs,
        or we track the unions and revert them.
        
        Actually, reverting DSU unions is complex. Let's use a different strategy:
        We'll recompute the DSU state from scratch for affected pairs when backtracking.
        But that's expensive too.
        
        Alternative: use a persistent/functional DSU or just recompute.
        For K=3,4,5 and 50 vertices, let's just recompute affected DSUs on backtrack.
        
        Actually, the simplest correct approach: when we backtrack, we recompute
        the DSU for each affected color pair from the current coloring.
        """
        pass  # Handled by recomputation below
    
    # Better approach: use a full recompute strategy with memoization
    # Or: track all unions made and revert them in reverse order
    
    # Let's use the approach of tracking unions for reverting
    union_log = []  # list of (dsu, x, y) for each union performed
    
    def assign(v, c):
        """Assign color c to v, with forward checking. Returns True/False."""
        # Check proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycles
        to_union = []
        for u in adj[v]:
            cu = colors[u]
            if cu == -1 or cu == c:
                continue
            a, b = min(c, cu), max(c, cu)
            dsu = get_or_init_dsu(a, b)
            rv = dsu_find(dsu, v)
            ru = dsu_find(dsu, u)
            if rv == ru:
                return False
            to_union.append((dsu, v, u))
        
        # Perform unions
        for dsu, x, y in to_union:
            dsu_union(dsu, x, y)
            union_log.append((dsu, x, y))
        
        colors[v] = c
        return True
    
    def unassign(v, c):
        """Undo assignment of color c to v."""
        colors[v] = -1
        # Revert unions in reverse order - but we need to know which unions were for this vertex
        # This is complex. Let's just recompute affected DSUs.
        # Find all color pairs affected by v's assignment
        affected_pairs = set()
        for u in adj[v]:
            cu = colors[u]
            if cu == -1:
                continue
            a, b = min(c, cu), max(c, cu)
            affected_pairs.add((a, b))
        
        # Recompute DSUs for affected pairs
        for (a, b) in affected_pairs:
            if (a, b) in pair_dsu:
                del pair_dsu[(a, b)]
            # Rebuild from current coloring
            new_dsu = list(range(N))
            for u2, v2 in edges:
                cu2, cv2 = colors[u2], colors[v2]
                if cu2 == -1 or cv2 == -1:
                    continue
                if {cu2, cv2} == {a, b}:
                    dsu_union(new_dsu, u2, v2)
            pair_dsu[(a, b)] = new_dsu
        
        # Clear union log entries for this assignment (approximate - just clear all)
        # Actually this is getting complicated. Let me use a cleaner approach.
        pass
    
    # The above revert approach is messy. Let me use a cleaner backtracking with
    # full state snapshots for the DSU structures when K is small.
    
    # Actually, for K <= 6 and N=50, the number of color pairs is at most C(6,2)=15.
    # Each DSU is size 50. So a full snapshot is 15*50 = 750 ints. That's fine.
    
    def snapshot():
        return {key: list(dsu) for key, dsu in pair_dsu.items()}
    
    def restore(snap):
        pair_dsu.clear()
        for key, dsu in snap.items():
            pair_dsu[key] = list(dsu)
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = vertex_order[idx]
        
        # Try each color
        for c in range(K):
            # Quick check: can we assign c to v?
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored cycle constraints
            snap = snapshot()
            
            to_union = []
            for u in adj[v]:
                cu = colors[u]
                if cu == -1 or cu == c:
                    continue
                a, b = min(c, cu), max(c, cu)
                dsu = get_or_init_dsu(a, b)
                rv = dsu_find(dsu, v)
                ru = dsu_find(dsu, u)
                if rv == ru:
                    valid = False
                    break
                to_union.append((dsu, v, u))
            
            if not valid:
                restore(snap)
                continue
            
            # Perform unions
            for dsu, x, y in to_union:
                dsu_union(dsu, x, y)
            
            colors[v] = c
            
            if backtrack(idx + 1):
                return True
            
            # Backtrack
            colors[v] = -1
            restore(snap)
        
        return False
    
    if backtrack(0):
        return colors
    return None

# EVOLVE-BLOCK-END
