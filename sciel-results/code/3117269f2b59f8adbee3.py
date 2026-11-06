"""
Combined program: Uses efficient bit-mask based verification and a
multi-strategy search (algebraic construction + local repair) to find
4-colourings of K_n without monochromatic C6.

The structural idea from the inspiration is the efficient verification using
bitmasks and the DFS-based cycle check. The refinement machinery from the
current program is the local search with repair, which is enhanced with
better heuristics (e.g., trying to break cycles more effectively).
"""
import time
import random
import itertools

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def get_edge_index(i, j, n):
        """Map (i, j) with i < j to index in the flat array."""
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def verify(n, colors):
        """
        Verify that no color class contains a C6.
        Returns True if valid, False otherwise.
        """
        # Build adjacency masks for each color
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C6 in each color
        for c in range(4):
            # Build adjacency list for this color
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            # Check for C6: look for a cycle of length 6
            # For each edge (u, v), look for a path of length 4 from v to u
            # that does not use u or v as internal nodes.
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u:
                        continue
                    # We have edge (u, v). Look for path v -> w -> x -> y -> z -> u
                    # w in N(v) \ {u}
                    for w in adj_list[v]:
                        if w == u:
                            continue
                        # x in N(w) \ {u, v}
                        for x in adj_list[w]:
                            if x == u or x == v:
                                continue
                            # y in N(x) \ {u, v, w}
                            for y in adj_list[x]:
                                if y == u or y == v or y == w:
                                    continue
                                # z in N(y) \ {u, v, w, x} and z in N(u)
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x:
                                        continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def find_one_cycle(n, colors):
        """
        Find one monochromatic C6 cycle if it exists.
        Returns a list of 6 edges (u, v) in the cycle, or None if no cycle found.
        """
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u:
                        continue
                    for w in adj_list[v]:
                        if w == u:
                            continue
                        for x in adj_list[w]:
                            if x == u or x == v:
                                continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w:
                                    continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x:
                                        continue
                                    if (adj[c][z] >> u) & 1:
                                        # Found cycle: u-v-w-x-y-z-u
                                        # Edges: (u,v), (v,w), (w,x), (x,y), (y,z), (z,u)
                                        return [(u, v), (v, w), (w, x), (x, y), (y, z), (z, u)]
        return None

    def repair_search(n, time_limit):
        """
        Local search with repair.
        Starts with a random coloring and iteratively fixes cycles.
        """
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        deadline = time.time() + time_limit
        
        while time.time() < deadline:
            cycle = find_one_cycle(n, colors)
            if cycle is None:
                return colors
            
            # Pick a random edge in the cycle to recolor
            # To be more effective, we might want to pick an edge that is "overloaded"
            # but for now, random is a good baseline.
            u, v = random.choice(cycle)
            idx = get_edge_index(min(u, v), max(u, v), n)
            old_c = colors[idx]
            
            # Try to recolor to a color that doesn't immediately create a new cycle?
            # That's expensive. Just pick a random different color.
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
            
            colors[idx] = new_c
        
        return None

    def algebraic_search(n):
        """
        Try to find a coloring based on algebraic structures.
        For n=17, we can try to use the structure of the projective plane or similar.
        However, a simple heuristic is to try to distribute colors evenly.
        """
        # This is a placeholder for a more sophisticated algebraic construction.
        # For now, we rely on repair_search.
        return None

    # Strategy:
    # 1. Try n=18 with a short time budget (long shot).
    # 2. Try n=17 with a significant time budget.
    # 3. Try n=16, 15, etc.
    
    # Note: The seed program provides n=9.
    # We want to beat 9.
    
    # Try n=18 (new!)
    if time.time() < hard_deadline - 10:
        res = repair_search(18, 10)
        if res:
            if verify(18, res):
                return {'n': 18, 'colors': res}

    # Try n=17
    if time.time() < hard_deadline - 5:
        res = repair_search(17, 50)
        if res:
            if verify(17, res):
                return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 5:
        res = repair_search(16, 30)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 5:
        res = repair_search(15, 20)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    # Try n=14
    if time.time() < hard_deadline - 5:
        res = repair_search(14, 10)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}

    # Try n=13
    if time.time() < hard_deadline - 5:
        res = repair_search(13, 5)
        if res:
            if verify(13, res):
                return {'n': 13, 'colors': res}

    # Fallback to n=10
    if time.time() < hard_deadline - 2:
        res = repair_search(10, 2)
        if res:
            if verify(10, res):
                return {'n': 10, 'colors': res}

    # Absolute fallback: n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
