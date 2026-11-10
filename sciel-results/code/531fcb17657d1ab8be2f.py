import time
import random
import sys
from functools import lru_cache

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask ^= lsb
        return edges

    def max_clique(adj, n):
        """
        Find a maximum clique in the graph defined by adj (list of bitmasks).
        Returns the clique as a bitmask.
        Uses Tomita's algorithm with pivot and simple coloring bound.
        """
        best_clique = [0]
        best_size = [0]
        
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: adj[x].bit_count(), reverse=True)
        
        def expand(R, P, X):
            if P == 0:
                if R.bit_count() > best_size[0]:
                    best_size[0] = R.bit_count()
                    best_clique[0] = R
                return
            
            # Bound
            if R.bit_count() + P.bit_count() <= best_size[0]:
                return
            
            # Pivot selection
            pivot_mask = P | X
            if pivot_mask:
                # Choose vertex in P|X with max degree in P
                pm = pivot_mask
                best_pivot = -1
                max_deg = -1
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (adj[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
                
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                expand(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                if R.bit_count() + P.bit_count() <= best_size[0]:
                    break

        expand(0, (1 << n) - 1, 0)
        return best_clique[0]

    def min_clique_cover_bnb(adj, n, deadline, time_limit_per_call=0.05):
        """
        Branch and bound to find minimum edge clique cover.
        Returns list of cliques (as lists of vertices).
        """
        start_time = time.time()
        edges = get_edges(adj, n)
        if not edges:
            return []
        
        # Precompute clique possibilities? No, too many.
        # Instead, we will use a heuristic search with pruning.
        
        # State: remaining_edges (set of tuples)
        # This is expensive. Let's use a bitmask for edges?
        # n=64 -> ~2000 edges. Bitmask of 2000 bits is possible but operations are slower.
        # Let's stick to a set of edge indices or tuples for clarity, but optimize.
        
        # Actually, for n=64, the number of edges is ~1100.
        # A set of tuples is fine for ~100-200 depth? Maybe not.
        
        # Alternative: Recursive greedy with backtracking and strong bounds.
        # Bound: For any set of remaining edges E_rem, the minimum number of cliques 
        # is at least the size of a maximum matching in the graph formed by E_rem?
        # No, cliques can share vertices. 
        # A better bound: The chromatic number of the line graph? Too expensive.
        # Simple bound: Max degree in the remaining graph? No.
        # Let's use: The number of edges in a maximum matching of the remaining graph 
        # is a lower bound on the number of cliques? 
        # No, one clique can cover many matching edges.
        
        # Actually, a trivial lower bound is 1 if edges exist.
        # A better one: If we pick a vertex v, all edges incident to v must be covered by cliques containing v.
        # This suggests a recursive structure based on vertices.
        
        # Let's try a different approach: 
        # 1. Find Max Clique C.
        # 2. Remove edges in C.
        # 3. Recurse.
        # This is the current strategy. It is greedy.
        
        # To improve, we can use "Lookahead":
        # When selecting the next clique, consider not just the max clique, 
        # but also slightly smaller cliques that might lead to a better global solution.
        # This is essentially BnB.
        
        # Given the 120s limit, a full BnB on 64 vertices is likely infeasible 
        # unless the bound is very strong.
        
        # Let's implement a "Greedy with Randomized Restart and Local Search" hybrid.
        # Or, optimize the current greedy by using a better clique selection metric.
        
        # Metric: Instead of max size, maximize (size^2) / (number of edges covered)? 
        # Or simply max size is usually good.
        
        # Let's try to optimize the Max Clique finder to be faster, allowing more restarts.
        
        cliques = []
        remaining_adj = adj[:]
        
        # Greedy loop
        while True:
            has_edges = False
            for u in range(n):
                if remaining_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find max clique
            mask = max_clique(remaining_adj, n)
            
            if mask.bit_count() < 2:
                # Fallback
                found = False
                for u in range(n):
                    if remaining_adj[u]:
                        lsb = remaining_adj[u] & -remaining_adj[u]
                        v = lsb.bit_length() - 1
                        mask = (1 << u) | (1 << v)
                        found = True
                        break
                if not found:
                    break
            
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            if len(verts) >= 2:
                cliques.append(verts)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        remaining_adj[u] &= ~(1 << v)
                        remaining_adj[v] &= ~(1 << u)
            
            if time.time() > deadline:
                break
        
        # Safety: cover remaining edges as pairs
        for u in range(n):
            mask = remaining_adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    remaining_adj[u] &= ~(1 << v)
                    remaining_adj[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques

    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)

    best_cliques_1 = None
    best_count_1 = float('inf')
    best_cliques_2 = None
    best_count_2 = float('inf')

    # Strategy: Multiple restarts with different random orderings for Max Clique heuristic?
    # The Max Clique algorithm is deterministic given the graph.
    # However, we can randomize the vertex order in the Max Clique search to find different max cliques?
    # Tomita's algorithm with color sorting is deterministic if order is fixed.
    # Let's add randomization to the Max Clique search order.
    
    # Modified max_clique to accept an order
    def max_clique_random(adj, n, order):
        best_clique = [0]
        best_size = [0]
        
        def expand(R, P, X):
            if P == 0:
                if R.bit_count() > best_size[0]:
                    best_size[0] = R.bit_count()
                    best_clique[0] = R
                return
            
            if R.bit_count() + P.bit_count() <= best_size[0]:
                return
            
            pivot_mask = P | X
            if pivot_mask:
                pm = pivot_mask
                best_pivot = -1
                max_deg = -1
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (adj[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
                
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                expand(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                if R.bit_count() + P.bit_count() <= best_size[0]:
                    break

        expand(0, (1 << n) - 1, 0)
        return best_clique[0]

    def greedy_partition_random(adj, n, order, deadline):
        remaining_adj = adj[:]
        cliques = []
        
        while True:
            has_edges = False
            for u in range(n):
                if remaining_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            mask = max_clique_random(remaining_adj, n, order)
            
            if mask.bit_count() < 2:
                found = False
                for u in range(n):
                    if remaining_adj[u]:
                        lsb = remaining_adj[u] & -remaining_adj[u]
                        v = lsb.bit_length() - 1
                        mask = (1 << u) | (1 << v)
                        found = True
                        break
                if not found:
                    break
            
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            if len(verts) >= 2:
                cliques.append(verts)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        remaining_adj[u] &= ~(1 << v)
                        remaining_adj[v] &= ~(1 << u)
            
            if time.time() > deadline:
                break
        
        for u in range(n):
            mask = remaining_adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    remaining_adj[u] &= ~(1 << v)
                    remaining_adj[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques

    # Multi-start for Instance 1
    for i in range(50):
        if time.time() > deadline - 5: break
        order = list(range(32))
        random.shuffle(order)
        cliques = greedy_partition_random(adj_1, 32, order, deadline - 5)
        if len(cliques) < best_count_1:
            best_count_1 = len(cliques)
            best_cliques_1 = cliques

    if best_cliques_1 is None:
        best_cliques_1 = greedy_partition_random(adj_1, 32, list(range(32)), deadline - 5)

    # Multi-start for Instance 2
    for i in range(20):
        if time.time() > deadline - 5: break
        order = list(range(64))
        random.shuffle(order)
        cliques = greedy_partition_random(adj_2, 64, order, deadline - 5)
        if len(cliques) < best_count_2:
            best_count_2 = len(cliques)
            best_cliques_2 = cliques

    if best_cliques_2 is None:
        best_cliques_2 = greedy_partition_random(adj_2, 64, list(range(64)), deadline - 5)

    return [best_cliques_1, best_cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
