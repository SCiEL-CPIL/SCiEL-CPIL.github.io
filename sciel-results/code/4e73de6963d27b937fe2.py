import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    
    # Initialize random graph
    # We need a fast way to represent the graph and update 4-cycle count.
    # adj: n x n boolean matrix
    # common[i][j]: number of common neighbors of i and j
    
    # Random initialization
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    edges = set(all_edges[:target_edges])
    
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Initialize common neighbor counts
    # common[i][j] = sum(adj[i] & adj[j])
    common = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                common[i][j] = np.dot(adj[i], adj[j])
                
    def calc_4_cycles(common):
        # Sum over u < w of binom(common[u][w], 2)
        count = 0
        for u in range(n):
            for w in range(u+1, n):
                c = common[u][w]
                count += c * (c - 1) // 2
        return count // 2 # Divide by 2 because each cycle is counted twice (once for each pair of opposite vertices)
        
    current_4c = calc_4_cycles(common)
    best_4c = current_4c
    best_edges = set(edges)
    
    # Local Search: Simulated Annealing
    # Move: Remove one edge, Add one non-edge.
    # This keeps edge count constant.
    
    import time
    start_time = time.time()
    timeout = 110 # Leave buffer
    
    T = 10.0 # Initial temperature
    cooling = 0.999
    
    # Precompute list of all possible edges for fast access
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Map edge to index for faster lookup? Or just use set.
    # For speed, we can maintain a list of current edges and a set.
    
    # To speed up, we can iterate over a subset of moves.
    # Number of edges = 91. Number of non-edges = 465 - 91 = 374.
    # Pairs of (edge_to_remove, edge_to_add) = 91 * 374 ~ 34k.
    # We can't check all every step. We sample.
    
    while time.time() - start_time < timeout:
        # Pick random edge to remove
        u, v = random.choice(list(edges))
        
        # Pick random non-edge to add
        # We need a non-edge.
        # To avoid bias, we can pick random pair and check if it's in edges.
        # Or maintain a list of non-edges? Updating non-edges list is O(N).
        # Let's just sample.
        found = False
        while not found:
            w, x = random.choice(all_edges_list)
            if (w, x) not in edges:
                found = True
        
        # Calculate delta in 4-cycles
        # Removing (u,v) and adding (w,x)
        
        # Delta from removing (u,v):
        # For every common neighbor pair of u and v... wait.
        # The change in 4-cycles depends on the structure.
        
        # It is easier to just recalculate?
        # No, recalc is O(N^3) ~ 30k ops. 110s / 30k is ~ 3600 steps. Might be too slow for good convergence.
        # Let's try to compute delta incrementally.
        
        # A 4-cycle is affected if it involves u, v, w, or x.
        # This is complex.
        
        # Alternative: Use the formula $\frac{1}{2} \sum_{i<j} \binom{c_{ij}}{2}$.
        # When we remove (u,v), $c_{ij}$ changes for all $i, j$ such that the path $i-u-v-j$ or $i-v-u-j$ existed.
        # Specifically, $c_{ij}$ decreases by 1 if $i$ is neighbor of $u$ and $j$ is neighbor of $v$ (and $i \neq j, i \neq v, j \neq u$) OR $i$ is neighbor of $v$ and $j$ is neighbor of $u$.
        # Actually, $c_{ij}$ is the number of common neighbors.
        # If we remove edge (u,v):
        # For any $k$, if $k$ was a common neighbor of $u$ and some other node... no.
        # $c_{ab}$ changes if $u$ or $v$ is a common neighbor of $a$ and $b$.
        # $u$ is a common neighbor of $a$ and $b$ if $a \sim u$ and $b \sim u$.
        # Removing $(u,v)$ does NOT change the adjacency of $u$ to others, nor $v$ to others.
        # It only changes whether $u$ and $v$ are connected.
        # Wait, $c_{ab}$ counts common neighbors.
        # If we remove edge $(u,v)$, does $c_{ab}$ change?
        # $c_{ab} = \sum_k adj[a][k] adj[b][k]$.
        # The term for $k=u$ is $adj[a][u] adj[b][u]$. This doesn't depend on $adj[u][v]$.
        # The term for $k=v$ is $adj[a][v] adj[b][v]$. This doesn't depend on $adj[u][v]$.
        # So, removing an edge $(u,v)$ does NOT change the number of common neighbors $c_{ab}$ for ANY pair $a,b$?
        # Let's check.
        # $c_{ab}$ is the number of paths of length 2 between $a$ and $b$.
        # Path $a-u-b$. Depends on $a \sim u$ and $u \sim b$.
        # Path $a-v-b$. Depends on $a \sim v$ and $v \sim b$.
        # Removing $u \sim v$ does not break $a \sim u$ or $u \sim b$.
        # So $c_{ab}$ remains unchanged?
        # If $c_{ab}$ remains unchanged, then the number of 4-cycles remains unchanged?
        # That can't be right.
        
        # Let's re-evaluate the 4-cycle count formula.
        # 4-cycle $a-b-c-d-a$.
        # Opposite pairs: $(a,c)$ and $(b,d)$.
        # Common neighbors of $a$ and $c$: $b$ and $d$.
        # So $c_{ac} \ge 2$.
        # If we remove edge $(b,d)$, does $c_{ac}$ change?
        # $b$ is a neighbor of $a$ and $c$. $d$ is a neighbor of $a$ and $c$.
        # $b$ is still a neighbor of $a$ and $c$.
        # $d$ is still a neighbor of $a$ and $c$.
        # So $c_{ac}$ is still 2 (assuming no other common neighbors).
        # So the count of 4-cycles via common neighbors does NOT change when we swap edges?
        
        # Wait. The 4-cycle $a-b-c-d-a$ is a subgraph.
        # If I remove edge $(b,d)$, the cycle is broken.
        # But the formula $\sum \binom{c_{ab}}{2}$ counts the number of *pairs of common neighbors*.
        # If $b$ and $d$ are common neighbors of $a$ and $c$, they form a "diamond" $a-b-c-d-a$? No, a diamond is $a-b-c$ and $a-d-c$. The cycle is $b-c-d-a-b$? No.
        # The cycle is $a-b-c-d-a$.
        # The pairs of common neighbors for $(a,c)$ are $b$ and $d$.
        # The pair $(b,d)$ forms the "other" side of the cycle.
        # The formula counts the existence of the structure $a-b-c$ and $a-d-c$.
        # It does NOT require $b-d$ to be an edge.
        # So, the number of 4-cycles (cycles of length 4) is NOT simply $\frac{1}{2} \sum \binom{c_{ab}}{2}$.
        # That formula counts the number of *closed walks of length 4*? Or *diamonds*?
        # A "diamond" is $K_4$ minus an edge. It contains one 4-cycle? No, $K_4$ has 3 4-cycles. $K_4-e$ has 1 4-cycle?
        # Vertices 1,2,3,4. Edges 1-2, 2-3, 3-4, 4-1, 1-3.
        # Cycle 1-2-3-4-1.
        # Common neighbors of 1,3: 2 and 4. $\binom{2}{2}=1$.
        # Common neighbors of 2,4: 1 and 3. $\binom{2}{2}=1$.
        # Sum = 2. Div by 2 = 1. Correct.
        
        # What if we have edges 1-2, 2-3, 3-4, 4-1, 1-3, 2-4? ($K_4$)
        # Common neighbors of 1,3: 2,4. $\binom{2}{2}=1$.
        # Common neighbors of 2,4: 1,3. $\binom{2}{2}=1$.
        # Sum = 2. Div by 2 = 1.
        # But $K_4$ has 3 4-cycles: 1-2-3-4-1, 1-2-4-3-1, 1-3-2-4-1.
        # So the formula $\frac{1}{2} \sum \binom{c_{ab}}{2}$ counts the number of *induced* 4-cycles? Or just the "diamond" structures?
        # Actually, it counts the number of 4-cycles that are *not* chords?
        # In $K_4$, the cycle 1-2-3-4-1 has chord 1-3 and 2-4.
        # The formula counts 1.
        # So the formula is incorrect for counting ALL 4-cycles in a dense graph.
        
        # Correct formula for number of 4-cycles:
        # $N_{C4} = \frac{1}{8} \left( \text{tr}(A^4) - 2 \sum_i d_i^2 + \dots \right)$?
        # Or simpler:
        # Iterate over all pairs of edges $(u,v)$ and $(x,y)$?
        
        # Let's use a robust but slightly slower method for the local search evaluation.
        # Since N=31 is small, we can recompute the exact number of 4-cycles from scratch for each candidate move.
        # Complexity: $O(N^3)$ or $O(N^2 \cdot \text{deg})$.
        # $31^3 \approx 30,000$ operations.
        # Python is slow. 30,000 ops might take ~0.01s?
        # 110s / 0.01s = 11,000 iterations.
        # This might be enough to find a better solution than 315.
        
        def count_4_cycles_exact(adj):
            # Count 4-cycles
            # For each pair of vertices (i, j), count common neighbors.
            # Let c_ij be common neighbors.
            # The number of 4-cycles containing i and j as opposite vertices is binom(c_ij, 2).
            # Wait, does this count all 4-cycles?
            # In K4, c_13 = 2 (neighbors 2,4). binom(2,2)=1.
            # c_24 = 2 (neighbors 1,3). binom(2,2)=1.
            # Sum = 2. /2 = 1.
            # But K4 has 3 cycles.
            # The missing cycles are those that use the chord?
            # Cycle 1-2-4-3-1. Opposite 1,3. Neighbors 2,4.
            # Cycle 1-3-2-4-1. Opposite 1,3. Neighbors 2,4.
            # All 3 cycles in K4 have opposite pairs (1,3) or (2,4).
            # For (1,3), the common neighbors are 2 and 4.
            # The cycles are:
            # 1-2-3-4-1 (uses edges 1-2, 2-3, 3-4, 4-1)
            # 1-4-3-2-1 (same cycle)
            # 1-2-4-3-1 (uses edges 1-2, 2-4, 4-3, 3-1) -> This is a valid cycle in K4.
            # 1-3-2-4-1 (uses edges 1-3, 3-2, 2-4, 4-1) -> This is a valid cycle in K4.
            
            # So, for a fixed pair of opposite vertices (i,j) with common neighbors {k1, k2, ..., km},
            # any pair of distinct common neighbors (ka, kb) forms a 4-cycle i-ka-j-kb-i.
            # This cycle exists IF edges (i,ka), (ka,j), (j,kb), (kb,i) exist.
            # By definition of common neighbors, (i,ka), (ka,j), (j,kb), (kb,i) ALL exist.
            # So EVERY pair of common neighbors forms a distinct 4-cycle.
            # So why did K4 count only 1?
            # In K4, c_13 = 2. Pairs: {(2,4)}.
            # Cycle: 1-2-3-4-1.
            # Where are the other 2?
            # The other cycles in K4 are:
            # 1-2-4-3-1. Opposite vertices? 1 and 3? No, 1 and 3 are connected.
            # In 1-2-4-3-1, the vertices are 1,2,4,3.
            # Opposite pairs: (1,4) and (2,3).
            # Let's check c_14 in K4. Neighbors of 1: 2,3,4. Neighbors of 4: 1,2,3.
            # Common neighbors of 1 and 4: 2,3.
            # So c_14 = 2. binom(2,2)=1. Cycle: 1-2-4-3-1.
            # Let's check c_23 in K4. Neighbors of 2: 1,3,4. Neighbors of 3: 1,2,4.
            # Common neighbors of 2 and 3: 1,4.
            # So c_23 = 2. binom(2,2)=1. Cycle: 2-1-3-4-2 (which is 1-3-4-2-1).
            
            # So, sum over all pairs (i,j) of binom(c_ij, 2).
            # Pairs: (1,3)->1, (2,4)->1, (1,4)->1, (2,3)->1, (1,2)->? c_12=2 (3,4). 1. (3,4)->? c_34=2 (1,2). 1.
            # Total sum = 6.
            # Each 4-cycle is counted how many times?
            # Cycle 1-2-3-4-1.
            # Opposite pairs: (1,3) and (2,4).
            # Counted in c_13 and c_24.
            # So each cycle is counted exactly 2 times (once for each pair of opposite vertices).
            # So Total Cycles = Sum / 2.
            # For K4: Sum = 6. Cycles = 3. Correct.
            
            # So the formula IS correct: $N_{C4} = \frac{1}{2} \sum_{i<j} \binom{c_{ij}}{2}$.
            
            # Why did I think it was 1 earlier? I only summed (1,3) and (2,4). I missed (1,4), (2,3), (1,2), (3,4).
            # In a simple graph, we sum over ALL pairs i<j.
            
            return_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = np.dot(adj[i], adj[j])
                    return_count += c * (c - 1) // 2
            return return_count // 2

        current_4c = count_4_cycles_exact(adj)
        best_4c = current_4c
        best_edges = set(edges)
        
        # Local Search Loop
        while time.time() - start_time < timeout:
            # Pick random edge to remove
            u, v = random.choice(list(edges))
            
            # Pick random non-edge to add
            # Sample until found
            while True:
                w, x = random.choice(all_edges_list)
                if (w, x) not in edges:
                    break
            
            # Apply move
            edges.remove((u, v))
            edges.add((w, x))
            adj[u, v] = 0
            adj[v, u] = 0
            adj[w, x] = 1
            adj[x, w] = 1
            
            new_4c = count_4_cycles_exact(adj)
            
            if new_4c < best_4c:
                best_4c = new_4c
                best_edges = set(edges)
                
            # Accept or reject
            delta = new_4c - current_4c
            if delta < 0:
                current_4c = new_4c
            else:
                # Reject, revert
                edges.remove((w, x))
                edges.add((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
                adj[w, x] = 0
                adj[x, w] = 0
                # current_4c remains unchanged
                
            T *= cooling
            
    return [list(e) for e in best_edges]
