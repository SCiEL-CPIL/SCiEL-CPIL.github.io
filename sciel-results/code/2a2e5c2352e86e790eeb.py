import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    best_solution = []
    best_size = 0
    
    def greedy_color(P):
        """
        P is a list of vertices.
        Returns: colors list (same length as P) and max_color
        """
        if not P:
            return [], 0
        colors = [0] * len(P)
        color_map = {}
        for i, v in enumerate(P):
            used = set()
            for u in adj[v]:
                if u in color_map:
                    used.add(color_map[u])
            c = 1
            while c in used:
                c += 1
            colors[i] = c
            color_map[v] = c
        max_c = max(colors) if colors else 0
        return colors, max_c

    def greedy_mis(adj, n):
        """Fallback greedy algorithm"""
        remaining = set(range(1, n + 1))
        result = []
        while remaining:
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            result.append(min_v)
            remaining.discard(min_v)
            remaining -= adj[min_v]
        return result

    def initial_solution():
        """Build a good initial solution: all odds + compatible evens"""
        odds = [i for i in range(1, n + 1) if i % 2 == 1]
        # Check which evens are compatible with ALL odds
        compatible_evens = []
        for e in range(2, n + 1, 2):
            compatible = True
            for o in odds:
                if e in adj[o]:
                    compatible = False
                    break
            if compatible:
                compatible_evens.append(e)
        
        # Now find max independent set among compatible evens
        # (they have no conflicts with odds, but may conflict with each other)
        even_mis = greedy_mis_on_subset(compatible_evens, adj)
        
        return odds + even_mis

    def greedy_mis_on_subset(vertices, adj):
        """Greedy MIS on a subset of vertices"""
        remaining = set(vertices)
        result = []
        while remaining:
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            result.append(min_v)
            remaining.discard(min_v)
            remaining -= adj[min_v]
        return result

    # Get initial solution
    init_sol = initial_solution()
    if len(init_sol) > best_size:
        best_size = len(init_sol)
        best_solution = init_sol[:]
    
    def expand(P, sol):
        nonlocal best_solution, best_size
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol.copy()
            return
            
        # Compute UB using greedy coloring
        colors, max_color = greedy_color(P)
        
        if len(sol) + max_color <= best_size:
            return
            
        # Order P by color (ascending) then by degree (descending)
        # Vertices with lower colors should be considered first
        # This helps find good solutions early
        P_with_colors = list(zip(P, colors))
        P_with_colors.sort(key=lambda x: (x[1], -len(adj[x[0]])))
        P = [v for v, c in P_with_colors]
        colors = [c for v, c in P_with_colors]
        
        for i in range(len(P)):
            v = P[i]
            
            # Include v: remove v and its neighbors from P
            new_P = []
            for j in range(len(P)):
                u = P[j]
                if u != v and u not in adj[v]:
                    new_P.append(u)
            
            # Recurse with v included
            expand(new_P, sol + [v])
            
            # Exclude v: remove v from P
            P.pop(i)
            i -= 1
            
            # Check if remaining P can beat best
            if P:
                _, ub_remaining = greedy_color(P)
                if len(sol) + ub_remaining <= best_size:
                    break

    initial_P = list(range(1, n + 1))
    expand(initial_P, [])
    
    if not best_solution:
        best_solution = greedy_mis(adj, n)

    return list(best_solution)
