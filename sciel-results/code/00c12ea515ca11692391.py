"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > N:
            return False
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > N:
                return False
    return True

def _divisors_of(n, limit=N):
    res = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            if i <= limit:
                res.add(i)
            j = n // i
            if j != i and j <= limit:
                res.add(j)
    return res

def construct():
    start = time.time()

    # --- Precompute LCM compatibility matrix using bitmasks ---
    # compat_mask[v]: bitmask where bit u is set iff u is in [1,512] and lcm(v,u) <= 512
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))
        compat_mask[v] = mask

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def greedy_augment(seed_set):
        """Start from seed_set, greedily add elements compatible with all current members."""
        S = set(seed_set)
        S_mask = set_to_mask(S)
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if v in S:
                    continue
                if is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    changed = True
        return S

    def greedy_by_degree(seed_set):
        """Greedy: repeatedly add the element (compatible with all in S) that is compatible with the most other candidates."""
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_deg = -1
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & remaining_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    # --- Branch and Bound for Maximum Independent Set ---
    # This can find sets larger than any single divisor set.
    # We use a bitmask representation for the graph.
    # Vertices are 1..512. Bit i represents vertex i+1.
    
    def branch_and_bound(initial_mask, limit_nodes=50000):
        """
        Find a large independent set starting from initial_mask.
        Uses branch and bound on the remaining vertices.
        """
        S_mask = initial_mask
        S_size = S_mask.bit_count()
        
        # Remaining vertices: those compatible with all in S_mask
        rem_mask = 0
        for v in range(1, N + 1):
            if not (S_mask & (1 << (v-1))) and is_compatible(v, S_mask):
                rem_mask |= (1 << (v-1))
        
        best_size = S_size
        best_mask = S_mask
        nodes_visited = 0
        
        # Pre-compute upper bound heuristic: 
        # A simple upper bound for the size of an independent set in a graph G is 
        # the number of vertices. But we can do better.
        # For each vertex v, let d(v) be its degree in the induced subgraph on rem_mask.
        # A clique cover upper bound is complex. 
        # A simple bound: sum of 1/(d(v)+1) is a lower bound on independence number? No.
        # Let's use a greedy coloring bound or just simple counting.
        # Actually, for this specific graph (lcm constraint), the structure is special.
        # Let's stick to a standard B&B with a simple upper bound: |rem_mask|.
        # To make it effective, we need good ordering and pruning.
        
        # Heuristic upper bound: 
        # The maximum independent set in a graph is at most the size of the largest clique in the complement? No.
        # Let's use the fact that if we pick a vertex v, we remove it and its neighbors.
        # Bound: 1 + max_independent_set(rem_mask \ {v} \ N(v))
        # This is recursive.
        
        # Let's use a simpler approach:
        # Order vertices by degree (descending) to prune faster.
        # Upper bound: number of remaining vertices.
        # Pruning: if S_size + popcount(rem_mask) <= best_size, prune.
        
        # To improve the bound, we can use a greedy coloring of the complement graph?
        # Or just a simple greedy independent set on rem_mask to get a lower bound for the subproblem,
        # which helps in branch selection but not directly in upper bound pruning unless we use it cleverly.
        
        # Let's implement a recursive search with memoization? No, state space is too large.
        # Let's use a stack-based DFS.
        
        # State: (current_mask, current_size, rem_mask)
        # We want to maximize current_size + independent_set_size(rem_mask)
        
        # Optimization: If rem_mask is 0, update best.
        # If current_size + popcount(rem_mask) <= best_size, return.
        
        # To make the bound tighter, we can compute a greedy independent set on rem_mask
        # and use its size as a lower bound for the "potential" gain?
        # No, we need an upper bound.
        # A common upper bound for MIS is the size of a minimum vertex cover, which is hard.
        # However, for sparse graphs, simple bounds work.
        
        # Let's try a different bound:
        # For each vertex v in rem_mask, it can be in the independent set or not.
        # If we pick v, we gain 1 and lose N(v) from rem.
        # If we don't pick v, we lose v from rem.
        # This is the standard recursion.
        
        # To speed up, we pre-compute the neighbors mask for each vertex within the full graph.
        # But rem_mask changes.
        
        # Let's use a simple recursive function with a global counter.
        
        def dfs(rem, current_mask, current_size):
            nonlocal best_size, best_mask, nodes_visited
            
            if nodes_visited > limit_nodes:
                return
            
            nodes_visited += 1
            
            if rem == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Upper bound: current_size + popcount(rem)
            if current_size + rem.bit_count() <= best_size:
                return
            
            # Heuristic: pick a vertex to branch on.
            # Choose the vertex with the highest degree in the induced subgraph on rem?
            # Or the one with the lowest degree?
            # Usually, picking a high-degree vertex prunes more because it removes more neighbors.
            
            # Find a vertex in rem.
            # Let's pick the one with the highest degree in the induced subgraph on rem.
            # Degree of v in induced subgraph on rem = popcount(compat_mask[v] & rem)
            
            # To avoid O(N^2) per node, we can just pick the first one or use a heuristic.
            # Let's pick the vertex with the smallest index first for determinism, 
            # but for pruning, high degree is better.
            
            # Let's find the vertex with max degree in rem.
            max_deg = -1
            pivot_v = -1
            r = rem
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & rem).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot_v = v
                r &= r - 1
            
            if pivot_v == -1:
                return
            
            # Branch 1: Include pivot_v
            # New rem: rem \ {pivot_v} \ N(pivot_v)
            # N(pivot_v) in the graph of incompatible pairs?
            # Wait, our graph G has edges between incompatible pairs.
            # We want an Independent Set in G, which is a Clique in the Complement.
            # Or simply: a set of vertices with no edges between them in G.
            # compat_mask[v] gives vertices compatible with v.
            # Incompatible vertices are ~compat_mask[v] (within [1,512]).
            # So if we pick v, we must remove all u such that lcm(v,u) > N.
            # These are u where bit (u-1) is 0 in compat_mask[v].
            
            # Let's define incompat_mask[v] = ~compat_mask[v] & full_mask
            # full_mask has bits 0..511 set.
            
            # Actually, we stored compat_mask.
            # Incompatible with v: full_mask & ~compat_mask[v]
            
            full_mask = (1 << N) - 1
            
            # Branch 1: Pick pivot_v
            new_rem1 = rem & ~(1 << (pivot_v - 1)) & ~compat_mask[pivot_v]
            # Wait, if we pick pivot_v, we cannot pick any u that is INCOMPATIBLE with pivot_v.
            # So we must remove incompatible vertices from rem.
            # Incompatible vertices are those NOT in compat_mask[pivot_v].
            # So new_rem1 = rem & compat_mask[pivot_v] & ~(1 << (pivot_v - 1))
            # Because any u in new_rem1 must be compatible with pivot_v.
            
            new_rem1 = rem & compat_mask[pivot_v] & ~(1 << (pivot_v - 1))
            
            dfs(new_rem1, current_mask | (1 << (pivot_v - 1)), current_size + 1)
            
            if nodes_visited > limit_nodes:
                return
            
            # Branch 2: Exclude pivot_v
            new_rem2 = rem & ~(1 << (pivot_v - 1))
            
            dfs(new_rem2, current_mask, current_size)
        
        dfs(rem_mask, S_mask, S_size)
        
        # Convert best_mask back to set
        best_set = set()
        m = best_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length()
            best_set.add(v)
            m &= m - 1
            
        return best_set

    # --- Find good starting seeds ---
    # Focus on M with many divisors
    candidate_Ms = list(range(1, N + 1))
    
    def div_count(m):
        return len(_divisors_of(m, N))
    
    M_sorted = sorted(candidate_Ms, key=div_count, reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top M values as seeds
    tried_M = set()
    for M in M_sorted:
        if time.time() - start > DEADLINE - 10:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        
        d = _divisors_of(M, N)
        if not d:
            continue
        
        S1 = greedy_augment(d)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
        
        if len(tried_M) >= 20:
            break
    
    # Try from empty set with degree-based greedy
    if time.time() - start < DEADLINE - 15:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3
    
    # Try increasing-order greedy from empty
    if time.time() - start < DEADLINE - 10:
        S = set()
        S_mask = 0
        for v in range(1, N + 1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # --- Branch and Bound Polish ---
    # Apply B&B to the best set found so far to try to extend it further.
    if time.time() - start < DEADLINE - 5:
        # Use a limited node count to ensure we finish in time
        # The search space can be large, so we limit the depth/width.
        # We start from the best_S found by heuristics.
        initial_mask = set_to_mask(best_S)
        
        # Run B&B with a node limit
        # If the graph is dense, B&B is fast. If sparse, it might explode.
        # The lcm graph is somewhat structured.
        # Let's try with a moderate limit.
        polished_S = branch_and_bound(initial_mask, limit_nodes=10000)
        
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_S = polished_S

    # Final polish: try to add one more element to best_S
    if time.time() - start < DEADLINE - 2:
        S_mask = set_to_mask(best_S)
        for v in range(1, N + 1):
            if v not in best_S and is_compatible(v, S_mask):
                best_S.add(v)
                break
    
    # Validate
    if not _validate(best_S):
        # Fallback
        best_S = set(range(1, 21))
    
    return list(best_S)
