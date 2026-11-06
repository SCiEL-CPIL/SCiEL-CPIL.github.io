import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def build_conflict_graph(V):
        n = len(V)
        V_set = set(V)
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(V)}
        
        # Adjacency list
        adj = [set() for _ in range(n)]
        
        # Build edges
        for i in range(n):
            u = V[i]
            for j in range(i + 1, n):
                v = V[j]
                # Check sum condition
                if u + v in V_set:
                    adj[i].add(j)
                    adj[j].add(i)
                # Check double condition (u = 2v or v = 2u)
                # Since u, v distinct, one is not double of other unless 0 involved (excluded)
                if u == 2 * v or v == 2 * u:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def solve_mis(adj, time_limit):
        n = len(adj)
        if n == 0:
            return []
        
        # Initial greedy coloring to get a good lower bound and start
        # We will use a branch and bound with greedy coloring bound
        
        best_size = 0
        best_set = []
        
        # Sort nodes by degree descending for better pruning? 
        # Or just use the order.
        
        # Tomita's algorithm structure
        # We maintain a list of candidates C
        # We color C greedily.
        
        # To speed up, we can use a simple B&B first to get a good lower bound
        # Greedy MIS
        C = list(range(n))
        S = []
        while C:
            # Pick node with min degree in induced subgraph? Or just first
            # For speed, just pick first
            v = C[0]
            S.append(v)
            # Remove v and its neighbors from C
            neighbors = adj[v]
            C = [x for x in C if x != v and x not in neighbors]
        
        best_size = len(S)
        best_set = S[:]
        
        # Branch and Bound
        # Stack: (current_set, candidates)
        # We use recursion or explicit stack
        # Explicit stack to control time
        
        # We need a function to greedily color a set of nodes
        def greedy_color(nodes):
            # Returns list of colors (indices) and color_count
            # nodes is a list of node ids
            # We want to assign colors such that adjacent nodes have different colors
            # This is an upper bound on MIS size
            color_map = {}
            next_color = 0
            # Sort by degree descending in the subgraph? 
            # Simple approach: iterate nodes, assign smallest available color
            # For bound, we just need the number of colors used.
            # Standard greedy coloring for bound:
            # Process nodes in order, assign color 0 if possible, else 1, etc.
            # The max color assigned + 1 is the bound.
            
            # To make the bound tighter, we should process nodes with high degree first?
            # Actually, for MIS bound, we want the coloring to be as "compact" as possible?
            # No, the number of colors is an upper bound on the clique number of the complement?
            # Wait. MIS in G = Clique in G_complement.
            # Greedy coloring of G gives a lower bound on Chromatic Number of G.
            # This is NOT an upper bound on MIS.
            
            # Correction:
            # The standard B&B for MIS uses the bound: |S| + |C|.
            # A tighter bound is obtained by finding the size of the maximum clique in the *complement* of the induced subgraph on C?
            # Or, using the fact that MIS(G) <= Clique(G_comp).
            # Greedy coloring of G gives a lower bound on Chi(G).
            # This doesn't help directly for MIS upper bound.
            
            # However, there is a technique:
            # If we greedily color the candidate set C, and we get k colors,
            # does that bound the MIS?
            # No.
            
            # Let's stick to the simple bound |S| + |C| first, and optimize if needed.
            # For sparse graphs, |S| + |C| is often loose.
            
            # Alternative: Use a heuristic to find a large independent set in C quickly to update best_size.
            
            return 0 # Placeholder

        # Let's implement a simple recursive B&B with time check
        # To avoid recursion depth issues and overhead, we'll use a stack
        # But recursion is easier to write. We'll limit depth or use sys.setrecursionlimit
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Precompute neighbor sets for fast lookup
        # adj is already list of sets
        
        def dfs(S, C):
            nonlocal best_size, best_set, start_time
            
            if time.time() - start_time > time_limit:
                return False # Time out signal
            
            if not C:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                return True
            
            # Bound
            if len(S) + len(C) <= best_size:
                return True
            
            # Heuristic: Try to add a node from C to S
            # Pick a node v from C
            # To improve, pick v with high degree in C?
            # Or just the first one.
            v = C[0]
            
            # Branch 1: Include v
            new_S = S + [v]
            # New C: remove v and its neighbors from C
            # Neighbors of v in C
            C_set = set(C)
            new_C = []
            for u in C:
                if u == v or u in adj[v]:
                    continue
                new_C.append(u)
            
            if dfs(new_S, new_C):
                return False # Time out propagated
            
            # Branch 2: Exclude v
            # New S: S
            # New C: C without v
            new_C2 = C[1:] # Since v is C[0]
            
            if dfs(S, new_C2):
                return False
            
            return True
        
        # Initial call
        # We can try to order nodes to improve pruning
        # Sort C by degree descending?
        # Let's just run it.
        
        # To make it faster, we can use a greedy heuristic to update best_size before B&B
        # Already done above.
        
        # Run B&B
        # We need to be careful with time.
        # The DFS above might be slow if it explores too much.
        # We will add a time check in the main loop of a while-based search if needed, 
        # but recursion with time check at entry is okay for small N.
        
        # One issue: The order of C matters.
        # Let's sort C by degree in the induced subgraph?
        # For now, we use the initial order.
        
        # To improve speed, we can use a "core" node selection:
        # Pick the node with the highest degree in C.
        
        def dfs_optimized(S, C):
            nonlocal best_size, best_set, start_time
            
            if time.time() - start_time > time_limit:
                return False
            
            if not C:
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                return True
            
            if len(S) + len(C) <= best_size:
                return True
            
            # Select pivot node: max degree in C
            # Calculate degrees in C
            C_set = set(C)
            degs = {}
            for u in C:
                degs[u] = len(adj[u] & C_set)
            
            v = max(C, key=lambda x: degs[x])
            
            # Branch 1: Include v
            new_S = S + [v]
            new_C = [u for u in C if u != v and u not in adj[v]]
            
            if dfs_optimized(new_S, new_C):
                return False
            
            # Branch 2: Exclude v
            new_C2 = [u for u in C if u != v]
            
            if dfs_optimized(S, new_C2):
                return False
            
            return True
        
        # Run
        dfs_optimized([], list(range(n)))
        
        return best_set

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback: Greedy
            S_fallback = []
            Sset = set()
            for x in sorted(V, reverse=True):
                valid = True
                for a in S_fallback:
                    if x + a in Sset or x - a in Sset:
                        valid = False
                        break
                if valid:
                    S_fallback.append(x)
                    Sset.add(x)
            results.append(sorted(S_fallback))
            continue
        
        # Allocate time
        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 2
        
        adj = build_conflict_graph(V)
        
        # Solve MIS
        idx_set = solve_mis(adj, time_per_instance)
        
        # Convert indices back to values
        S_final = [V[i] for i in idx_set]
        
        # Safety Check
        Sset = set(S_final)
        valid = True
        S_list = list(S_final)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in Sset:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback
            S_fallback = []
            Sset_f = set()
            for x in sorted(V, reverse=True):
                ok = True
                for a in S_fallback:
                    if x + a in Sset_f or x - a in Sset_f:
                        ok = False
                        break
                if ok:
                    S_fallback.append(x)
                    Sset_f.add(x)
            S_final = S_fallback
            
        results.append(sorted(S_final))
        
    return results
