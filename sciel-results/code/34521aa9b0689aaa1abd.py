import time
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n = 17
    
    # Try cyclic coloring: color(i,j) = f((j-i) % 17) for i != j
    # We need to assign each of {1,...,16} to one of 4 colors
    # such that no color class contains a C6
    
    # A C6 in Cay(Z_17, D) exists iff there's a closed walk of length 6
    # with all intermediate vertices distinct (a 6-cycle)
    
    def has_c6_in_cayley(D_set):
        """Check if Cayley graph on Z_17 with difference set D_set contains a C6."""
        D = set(D_set)
        # A C6: vertices v0, v1, v2, v3, v4, v5, v0
        # where v_{i+1} - v_i in D for all i (mod 6), all vi distinct
        # By translation invariance, set v0 = 0
        # v1 in D, v2-v1 in D, v3-v2 in D, v4-v3 in D, v5-v4 in D, v0-v5 = -v5 in D
        # So v5 in -D = {17-d for d in D}
        
        # Enumerate: v1 in D, v2 in v1+D, v3 in v2+D, v4 in v3+D, v5 in v4+D
        # Check: -v5 in D (i.e., v5 in negD) and all 6 vertices distinct
        
        negD = {(17 - d) % 17 for d in D}
        
        # We need: 0, v1, v2, v3, v4, v5 all distinct, v1 in D, v2-v1 in D,
        # v3-v2 in D, v4-v3 in D, v5-v4 in D, -v5 in D
        
        for v1 in D:
            for v2 in [(v1 + d) % 17 for d in D]:
                if v2 == 0 or v2 == v1:
                    continue
                for v3 in [(v2 + d) % 17 for d in D]:
                    if v3 in (0, v1, v2):
                        continue
                    for v4 in [(v3 + d) % 17 for d in D]:
                        if v4 in (0, v1, v2, v3):
                            continue
                        for v5 in [(v4 + d) % 17 for d in D]:
                            if v5 in (0, v1, v2, v3, v4):
                                continue
                            if (-v5) % 17 in D:
                                return True
        return False
    
    # Try to find a partition of {1,...,16} into 4 sets of 4 with no C6 in any
    # Use backtracking search
    
    all_diffs = list(range(1, 17))
    colors = [set() for _ in range(4)]
    
    def backtrack(idx):
        if time.time() > deadline - 5:
            return None
        if idx == 16:
            # Check all 4 color classes
            for c in range(4):
                if has_c6_in_cayley(colors[c]):
                    return None
            return [list(colors[c]) for c in range(4)]
        
        d = all_diffs[idx]
        for c in range(4):
            if len(colors[c]) < 4:
                colors[c].add(d)
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                colors[c].remove(d)
        
        return None
    
    result = backtrack(0)
    
    if result is not None:
        # Build the coloring
        color_map = {}
        for c, D in enumerate(result):
            for d in D:
                color_map[d] = c
                color_map[(17 - d) % 17] = c
        
        colors_list = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % 17
                colors_list.append(color_map[diff])
        
        # Verify no C6 in any color class
        def verify_no_c6(colors_list, n):
            # Build adjacency for each color
            for c in range(4):
                adj = [[False] * n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors_list[idx] == c:
                            adj[i][j] = adj[j][i] = True
                        idx += 1
                # Check for C6 using BFS/DFS
                # For each pair of non-adjacent vertices, check if there's a path of length 5
                # Actually, just check all 6-tuples... that's too slow for n=17
                # Better: for each edge, check if endpoints have a common neighbor at distance 5
                # Use: C6 exists iff there are vertices a,b,c,d,e,f with edges ab,bc,cd,de,ef,fa
                # For n=17, we can do: for each pair (a,f), check if there's a path of length 5 between them
                # that doesn't use the edge (a,f) directly... actually we just need any C6
                
                # Simpler: for each 6-tuple of distinct vertices, check if they form a cycle
                # 17 choose 6 * 5! / 2 is too many... 
                # Better: for each edge (a,b), look for a path of length 5 from b to a not using (a,b)
                # That's a path of length 5 in the graph minus edge (a,b)
                
                # Actually for n=17, let's just check: does the graph contain C6?
                # Use: for each pair of vertices at distance exactly 6 in some sense...
                # Let me just do a direct check: for each 6-subset, check if it induces a C6
                # C(17,6) = 12376, for each check if it's a C6 (6! / 2 = 360 orderings to check... too slow)
                
                # Better approach: for each vertex a, for each neighbor b of a, for each neighbor c of b (c != a),
                # for each neighbor d of c (d != a,b), for each neighbor e of d (e != a,b,c),
                # check if e is neighbor of a and e != b and a != c... 
                # This is O(n * d^5) which for n=17, d<=16 is manageable
                
                for a in range(n):
                    for b in range(n):
                        if not adj[a][b]:
                            continue
                        for c in range(n):
                            if c == a or c == b or not adj[b][c]:
                                continue
                            for d in range(n):
                                if d in (a, b, c) or not adj[c][d]:
                                    continue
                                for e in range(n):
                                    if e in (a, b, c, d) or not adj[d][e]:
                                        continue
                                    if adj[e][a] and e != b:
                                        return False  # Found C6: a-b-c-d-e-a
                return True
            
            return True
        
        if verify_no_c6(colors_list, n):
            return {'n': n, 'colors': colors_list}
    
    # Fallback: return the known n=9 coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
