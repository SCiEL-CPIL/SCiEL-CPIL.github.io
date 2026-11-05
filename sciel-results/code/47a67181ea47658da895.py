import time
from math import gcd
import random

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Approach: 3-uniform hypergraph maximum independent set via weighted greedy
    (lowest-degree-first) followed by simulated annealing local search.
    """
    DEADLINE = time.time() + 55.0  # safety margin before 60s kill
    
    N = 128
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute for each element x, the set of pairs (u,v) in S such that
    # {x, u, v} forms a bad triple. We'll compute this on the fly using
    # a helper that checks if adding x to a set S creates any bad triple.
    
    def conflicts_with(x, S):
        """Return number of bad triples created by adding x to set S."""
        S_list = list(S)
        count = 0
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = S_list[i], S_list[j]
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    count += 1
        return count
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = S_list[i], S_list[j]
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    return False
        return True
    
    def greedy_build_lowest_degree():
        """
        Greedy build: start empty, repeatedly add the element with the 
        lowest marginal conflict (number of bad triples it would create).
        This is the hypergraph-degree-based approach.
        """
        S = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            if time.time() > DEADLINE:
                break
            # For each candidate, compute marginal conflict
            best_elem = None
            best_conflict = float('inf')
            for x in remaining:
                c = conflicts_with(x, S)
                if c < best_conflict:
                    best_conflict = c
                    best_elem = x
                elif c == best_conflict and best_elem is not None:
                    # Tie-break: prefer lower global degree
                    pass
            
            if best_conflict == 0:
                S.add(best_elem)
                remaining.discard(best_elem)
            else:
                # Even the best candidate has conflicts; still add it if it's the least bad
                # Actually for independent set, we should only add if conflict is 0
                # But that would give us a very small set. Let's add the least-conflicting
                # and accept that we'll fix conflicts later.
                # Better: only add if no conflict, otherwise stop
                break
        
        return S
    
    def greedy_build_with_pruning():
        """
        Greedy build: add elements in order of increasing global conflict degree.
        Only add if it doesn't create a bad triple with existing elements.
        """
        # Compute global degree: for each x, count how many pairs (u,v) 
        # with u,v != x form a bad triple with x
        global_degree = [0] * (N + 1)
        for x in range(1, N + 1):
            for u in range(1, N + 1):
                if u == x:
                    continue
                for v in range(u + 1, N + 1):
                    if v == x:
                        continue
                    if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                        global_degree[x] += 1
        
        # Sort by increasing degree
        candidates = sorted(range(1, N + 1), key=lambda x: global_degree[x])
        
        S = set()
        for x in candidates:
            if time.time() > DEADLINE:
                break
            if can_add(x, S):
                S.add(x)
        
        return S
    
    def simulated_annealing(S, max_iters=None):
        """
        Simulated annealing: try swaps (remove one, add one) and additions.
        """
        if max_iters is None:
            max_iters = 50000
        
        S = set(S)
        all_elems = set(range(1, N + 1))
        not_in_S = all_elems - S
        
        best_S = set(S)
        best_size = len(S)
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            if T < min_T:
                break
            
            # Try a random move
            move_type = random.random()
            
            if move_type < 0.5 and not_in_S:
                # Try addition
                x = random.choice(list(not_in_S))
                if can_add(x, S):
                    S.add(x)
                    not_in_S.discard(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    # Reject (or accept with probability)
                    pass
            elif move_type < 0.8 and S and not_in_S:
                # Try swap: remove a from S, add b to S
                a = random.choice(list(S))
                b = random.choice(list(not_in_S))
                temp_S = S - {a}
                if can_add(b, temp_S):
                    S = temp_S
                    S.add(b)
                    not_in_S.discard(b)
                    not_in_S.add(a)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
            else:
                # Try removal (to escape local optima, though it reduces size)
                if S:
                    a = random.choice(list(S))
                    # Only remove if it helps us add something later
                    # For now, skip pure removals
                    pass
            
            T *= cooling
        
        return best_S
    
    # Phase 1: Greedy build with lowest-degree-first ordering
    best_S = greedy_build_with_pruning()
    best_size = len(best_S)
    
    # Phase 2: Simulated annealing on the greedy result
    S_sa = simulated_annealing(best_S, max_iters=30000)
    if len(S_sa) > best_size:
        best_size = len(S_sa)
        best_S = S_sa
    
    # Phase 3: Multiple randomized greedy starts
    random.seed(123)
    for trial in range(10):
        if time.time() > DEADLINE:
            break
        
        # Randomized ordering with bias toward low-degree
        global_degree = [0] * (N + 1)
        for x in range(1, N + 1):
            for u in range(1, N + 1):
                if u == x:
                    continue
                for v in range(u + 1, N + 1):
                    if v == x:
                        continue
                    if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                        global_degree[x] += 1
        
        # Create randomized ordering: sort by degree + random noise
        candidates = list(range(1, N + 1))
        candidates.sort(key=lambda x: global_degree[x] + random.random() * 5)
        
        S = set()
        for x in candidates:
            if time.time() > DEADLINE:
                break
            if can_add(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # SA on this result
        if time.time() < DEADLINE:
            S_sa = simulated_annealing(S, max_iters=10000)
            if len(S_sa) > best_size:
                best_size = len(S_sa)
                best_S = S_sa
    
    # Phase 4: Final aggressive SA on best
    if time.time() < DEADLINE - 5:
        S_final = simulated_annealing(best_S, max_iters=50000)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    if not best_S:
        best_S = {1}
    
    return sorted(best_S)
