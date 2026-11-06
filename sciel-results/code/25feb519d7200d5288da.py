import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def check_tournament(n, orientation):
        """
        Verify if the tournament satisfies the condition.
        orientation: list of bits, index k corresponds to pair (i, j) where 
        i = k // (n-1) ... wait, lexicographic order (i,j) with i<j.
        Number of pairs is n*(n-1)/2.
        Index mapping:
        (0,1) -> 0
        (0,2) -> 1
        ...
        (0, n-1) -> n-2
        (1,2) -> n-1
        ...
        General formula for pair (i, j):
        idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        Let's verify:
        i=0, j=1: 0*(n-1) - 0 + 0 = 0. Correct.
        i=0, j=2: 0*(n-1) - 0 + 1 = 1. Correct.
        i=1, j=2: 1*(n-1) - 1*2//2 + 0 = n-1 - 1 = n-2. 
        Wait.
        Pairs starting with 0: (0,1)...(0,n-1) -> n-1 pairs. Indices 0 to n-2.
        Pairs starting with 1: (1,2)...(1,n-1) -> n-2 pairs. Indices n-1 to n-2+n-2 = 2n-4.
        Formula check for i=1, j=2:
        idx = 1*(n-1) - 1*2//2 + (2-1-1) = n-1 - 1 + 0 = n-2.
        But the first pair with i=1 should be at index n-1 (after n-1 pairs of i=0).
        Let's re-derive.
        Number of pairs with first vertex < i is sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2.
        Then add (j - i - 1) for the position within vertex i's pairs.
        So idx = i(n-1) - i(i-1)/2 + (j - i - 1).
        Check i=1, j=2: 1(n-1) - 0 + 0 = n-1. Correct.
        Check i=0, j=1: 0 + 0 = 0. Correct.
        """
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orientation[idx] == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        
        # Build out-masks for faster checking
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        # Check all 4-sets
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                return False
        return True

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Strategy: Greedy Dominator Construction
    # 1. Choose a set of candidate dominators D.
    # 2. For the remaining vertices C, orient edges to maximize the chance that every 4-set is dominated.
    # Actually, a simpler structural approach:
    # Partition vertices into D (dominators) and C (clients).
    # Orient edges within D randomly.
    # Orient edges within C randomly.
    # Orient edges between D and C such that each d in D dominates a specific subset of C.
    # To guarantee every 4-set is dominated, we can try to make D a "dominating set" in a stronger sense.
    # If |D| is large enough, and we orient edges from D to C carefully, we can cover all 4-sets.
    
    # Let's try a specific construction:
    # Let n = k + m.
    # Vertices 0..k-1 are dominators.
    # Vertices k..n-1 are clients.
    # For each dominator d, let S_d be the set of clients it dominates.
    # We want that for any 4-set Q, there exists a dominator d such that Q \ D is subset of S_d (if Q has no dominators) or ...
    # This is complex. Let's stick to the repair method but with a better initialization.
    
    # New Idea: Use the property that if a vertex has out-degree > n/2, it dominates many sets.
    # Let's try a "Blow-up" of a smaller known tournament? No, we don't have one.
    
    # Let's try a different search: Simulated Annealing on the orientation bits directly, 
    # with a cost function that counts the number of violating 4-sets.
    
    def count_violations(n, adj, out_mask):
        count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                count += 1
        return count

    def simulate_annealing(n, initial_adj, max_iters=50000):
        adj = [row[:] for row in initial_adj]
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        current_cost = count_violations(n, adj, out_mask)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        T = 1.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time_left() < 5:
                break
            
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip edge u->v
            old_uv = adj[u][v]
            adj[u][v] = 1 - old_uv
            adj[v][u] = 1 - adj[u][v]
            
            # Update out_masks
            if adj[u][v]:
                out_mask[u] |= (1 << v)
                out_mask[v] &= ~(1 << u)
            else:
                out_mask[u] &= ~(1 << v)
                out_mask[v] |= (1 << u)
                
            new_cost = count_violations(n, adj, out_mask)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
                    if best_cost == 0:
                        break
            else:
                # Revert
                adj[u][v] = old_uv
                adj[v][u] = 1 - old_uv
                if old_uv:
                    out_mask[u] |= (1 << v)
                    out_mask[v] &= ~(1 << u)
                else:
                    out_mask[u] &= ~(1 << v)
                    out_mask[v] |= (1 << u)
                    
            T *= cooling
            
        return best_adj, best_cost

    # Try Simulated Annealing from random starts for smaller n
    for n in range(67, 83):
        if time_left() < 15:
            break
            
        for seed in range(10):
            if time_left() < 10:
                break
                
            random.seed(seed)
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
                        
            best_adj, best_cost = simulate_annealing(n, adj, max_iters=20000)
            
            if best_cost == 0:
                best_n = n
                best_orient = adj_to_orient(n, best_adj)
                break
                
        if best_n < 83:
            break
            
    return {'n': best_n, 'orientation': best_orient}
