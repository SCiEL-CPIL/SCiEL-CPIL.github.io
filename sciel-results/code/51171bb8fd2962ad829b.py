import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    N = 256
    candidates = list(range(2, N + 1))
    M = len(candidates)  # 255 elements

    # Precompute conflict: edge (i,j) if exists z != i,j in [2,256] such that candidates[i] | candidates[j]*z
    # or candidates[j] | candidates[i]*z
    # Actually the constraint is: for distinct a,b,c in S, a does NOT divide b*c.
    # So a conflict pair (a,b) exists if there is some c in [2,256], c != a, c != b, such that a|b*c or b|a*c.
    # Wait, that's not quite right either. Let me re-read.
    # "For every THREE DISTINCT members a,b,c, require that a does not divide b*c"
    # This means: for all ordered triples of distinct elements (a,b,c) from S, a ∤ b*c.
    # By symmetry, this is equivalent to: for all distinct a,b,c in S, a ∤ b*c AND b ∤ a*c AND c ∤ a*b.
    # So a pair (a,b) is "bad" if there exists c in [2,256], c≠a, c≠b, such that a|b*c.
    # If we include both a and b in S, and also include c, then the constraint is violated.
    # But we only care about triples ALL in S. So the constraint is really 3-uniform.
    # However, the strategy says to model as 2-uniform: edge (x,y) if exists ANY z in [2,256] with x|yz.
    # This is a conservative/sufficient condition: if x|yz for some z, then x and y cannot both be in S 
    # along with z. But if z is not in S, it's fine. So this reduction is too conservative.
    # Let me think again...
    # Actually, the pairwise reduction works as follows: if we define conflict(x,y) = "there exists z in [2,256], z≠x,y, such that x|yz",
    # then any independent set in this conflict graph is a valid solution. But it may miss valid solutions
    # where x and y are both included but the conflicting z is not.
    # However, for maximizing, this might still be a good approach since it's much faster.
    # Let me check: is the constraint actually equivalent to pairwise?
    # If a|b*c for distinct a,b,c, then {a,b,c} cannot all be in S.
    # This is a 3-uniform hypergraph constraint. The pairwise reduction creates edges that are sufficient
    # but not necessary conditions for conflict.
    # Let me just implement the direct 3-uniform check with bitsets for efficiency.

    # Direct approach: for each element i, compute a bitmask of elements j such that i|candidates[j]*candidates[k] for some k.
    # Actually, let's precompute for each pair (i,j) whether they conflict with any k.
    # conflict[i][j] = True if exists k != i, k != j such that candidates[i] | candidates[j]*candidates[k]
    # Then a valid set S must not contain any triple (i,j,k) where conflict[i][j] and k is also in S.
    # This is still 3-uniform. Let me just use the direct triple check with efficient bit operations.

    # Precompute: for each i, a bitmask conflict_with[i] where bit j is set if candidates[i] | candidates[j]*candidates[k] for some k.
    # Hmm, this is getting complex. Let me use a different approach.

    # For each element a, precompute which pairs (b,c) with b≠c, b≠a, c≠a have a|b*c.
    # Then when checking if we can add a to set S, we check if there exist b,c in S with b≠c and a|b*c.

    # Let me precompute for each a, a list of "forbidden pairs" - pairs (b,c) such that a|b*c.
    # Then can_add(a, S) checks if S contains any forbidden pair for a.

    # Actually, let's precompute a 2D table: conflict[a_idx][b_idx] = set of c_idx such that a|b*c or b|a*c or c|a*b
    # No wait, the constraint is asymmetric in the sense that for triple (a,b,c), we check all three:
    # a|b*c, b|a*c, c|a*b. Any one being true makes the triple invalid.

    # Let me precompute: for each ordered pair (i,j), i≠j, the set of k (k≠i, k≠j) such that 
    # candidates[i] | candidates[j]*candidates[k]
    # Then a triple (i,j,k) is invalid if i conflicts with (j,k) OR j conflicts with (i,k) OR k conflicts with (i,j).

    # For efficiency with bitsets:
    # For each i, precompute a list of bitmasks: for each j, mask[i][j] = bitmask of k such that candidates[i] | candidates[j]*candidates[k]
    # Then to check if adding i to S is valid: for each pair (j,k) in S, j≠k, check if i conflicts with (j,k) meaning
    # candidates[i] | candidates[j]*candidates[k]. Also need to check j conflicts with (i,k) and k conflicts with (i,j).

    # This is still O(n^2) per addition check. Let me think of a better way.

    # Alternative: precompute for each i, a bitmask bad_pairs[i] where bit (j,k) is set if candidates[i] | candidates[j]*candidates[k].
    # But that's 255*255/2 bits which is ~32KB per i, total ~8MB. That's feasible.
    # Actually, let's index pairs by a single integer: pair_idx = j*256 + k for j<k.
    # Number of pairs: 255*254/2 = 32385. So we need a 32385-bit integer per i. Python big ints handle this.

    # Let me precompute:
    # For each i in range(M):
    #   bad_mask[i] = 0
    #   for each pair (j,k) with j<k, j≠i, k≠i:
    #     if candidates[i] | candidates[j]*candidates[k]:
    #       bad_mask[i] |= (1 << pair_idx(j,k))
    #
    # Then to check if i can be added to S:
    #   Compute the bitmask of all pairs in S: S_pairs_mask
    #   Check if bad_mask[i] & S_pairs_mask == 0
    #   Also need to check the other two directions: j|a*c and k|a*b for pairs (j,k) in S.
    #   This means we also need bad_mask for the other elements.
    #   Actually, when we add i to S, we need to check all triples involving i:
    #   For each pair (j,k) in S: check if i|j*k OR j|i*k OR k|i*j
    #   The first is: bad_mask[i] has pair (j,k) set
    #   The second is: bad_mask[j] has pair (i,k) set  
    #   The third is: bad_mask[k] has pair (i,j) set
    #
    # So we need to check three things. This is still O(|S|) per pair check... no wait,
    # if we precompute bad_mask for all elements, then:
    #   For each j in S: check if bad_mask[j] has pair (i, k) for each k in S, k≠j. That's O(|S|^2).
    #
    # Hmm, this isn't simplifying things much. Let me just use the direct approach with some optimization.

    # Let me try a different strategy: use the pairwise conflict graph as a relaxation.
    # Define conflict(i,j) = True if there exists k in [2,256], k≠i, k≠j, such that 
    # candidates[i] | candidates[j]*candidates[k] OR candidates[j] | candidates[i]*candidates[k]
    # Then any independent set in this graph is valid. Solve max independent set on this graph.
    # This may not be optimal but should give a good lower bound, and we can try to improve.

    # Precompute pairwise conflict
    conflict = [[False] * M for _ in range(M)]
    for i in range(M):
        for j in range(i+1, M):
            ci = candidates[i]
            cj = candidates[j]
            has_conflict = False
            for k in range(M):
                if k == i or k == j:
                    continue
                ck = candidates[k]
                if (cj * ck) % ci == 0 or (ci * ck) % cj == 0:
                    has_conflict = True
                    break
            conflict[i][j] = has_conflict
            conflict[j][i] = has_conflict

    # Build adjacency as bitmasks
    adj = [0] * M
    for i in range(M):
        mask = 0
        for j in range(M):
            if i != j and conflict[i][j]:
                mask |= (1 << j)
        adj[i] = mask

    # Degree of each vertex
    degree = [bin(adj[i]).count('1') for i in range(M)]

    # Greedy: pick low-degree vertices first
    def greedy_select():
        selected = set()
        available = set(range(M))
        while available:
            # Pick vertex with minimum degree among available
            min_deg = float('inf')
            min_v = -1
            for v in available:
                # Degree within available
                deg = 0
                for u in available:
                    if u != v and conflict[v][u]:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            selected.add(min_v)
            available.discard(min_v)
            # Remove all neighbors of min_v from available
            for u in list(available):
                if conflict[min_v][u]:
                    available.discard(u)
        return selected

    # Verify a set is actually valid (full 3-uniform check)
    def is_valid_set(indices):
        vals = [candidates[i] for i in indices]
        n = len(vals)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = vals[i], vals[j], vals[k]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        return False
        return True

    # Branch and bound for maximum independent set
    def max_independent_set():
        best = set()
        best_size = 0
        
        # Order vertices by degree (ascending)
        order = sorted(range(M), key=lambda x: degree[x])
        
        # Use recursive branch and bound
        # State: current set, available set, upper bound
        def bb(current, available, bound):
            nonlocal best, best_size
            if time.time() > deadline:
                return
            if len(current) + len(available) <= best_size:
                return
            if not available:
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                return
            
            # Pick a vertex to branch on
            # Choose vertex with max degree in available for better pruning
            v = max(available, key=lambda x: bin(adj[x] & ((1 << M) - 1)).count('1') if False else 0)
            # Actually let's just pick the first one or use a heuristic
            v = min(available)
            
            # Branch 1: include v
            new_current = current | {v}
            new_available = available - {v}
            # Remove neighbors of v
            for u in list(new_available):
                if conflict[v][u]:
                    new_available.discard(u)
            bb(new_current, new_available, bound)
            
            # Branch 2: exclude v
            bb(current, available - {v}, bound)
        
        bb(set(), set(range(M)), M)
        return best

    # The pairwise conflict graph approach may be too restrictive. Let me try a different approach:
    # Directly search for valid sets using backtracking with pruning.

    # Precompute for each i, which j have i|j*k for some k
    # For efficiency, let's precompute divisors
    # i|j*k means i/gcd(i,j) | k. Let g = gcd(i,j), i' = i/g. Then we need k divisible by i'.
    # So i|j*k iff i/gcd(i,j) divides k.
    
    # Precompute: for each pair (i,j), the value i/gcd(i,j). If this is <= 256 and > 1,
    # then there exists k in [2,256] divisible by i/gcd(i,j), k≠i, k≠j, making (i,j,k) invalid.
    # Actually we need k to be in our candidate set and k≠i, k≠j.
    
    # Let me precompute conflict3[i][j] = True if there exists k≠i,k≠j with i|j*k or j|i*k
    # This is the same as the pairwise conflict I computed above!
    # Because i|j*k for some k means the triple (i,j,k) is invalid, so i and j are "in conflict"
    # in the sense that they can't both be in S with k. But they CAN be in S without k.
    # So the pairwise conflict is a sufficient but not necessary condition.
    
    # Let me just use the direct approach: build valid sets via greedy + local search with exact validity check.

    def can_add_exact(val, current_vals):
        """Check if val can be added to current_vals without violating constraint"""
        for i in range(len(current_vals)):
            b = current_vals[i]
            for j in range(i + 1, len(current_vals)):
                c = current_vals[j]
                if (b * c) % val == 0 or (val * c) % b == 0 or (val * b) % c == 0:
                    return False
        return True

    def greedy_exact(order):
        s = []
        for val in order:
            if can_add_exact(val, s):
                s.append(val)
        return s

    def local_search_exact(s, max_iters=5000):
        s = list(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            for i in range(len(s)):
                removed = s[i]
                new_s = [x for x in s if x != removed]
                for a in candidates:
                    if a in new_s:
                        continue
                    if a == removed:
                        continue
                    if can_add_exact(a, new_s):
                        new_s.append(a)
                        if len(new_s) > len(s):
                            s = new_s
                            improved = True
                            break
                if improved:
                    break
        return s

    # Strategy: multiple greedy orders + local search
    best_set = []

    # Order 1: descending
    s1 = greedy_exact(sorted(candidates, reverse=True))
    if len(s1) > len(best_set):
        best_set = s1

    # Order 2: ascending
    s2 = greedy_exact(sorted(candidates))
    if len(s2) > len(best_set):
        best_set = s2

    # Order 3: primes first
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    s3 = greedy_exact(sorted(primes, reverse=True) + sorted(composites, reverse=True))
    if len(s3) > len(best_set):
        best_set = s3

    # Local search on best
    if time.time() < deadline:
        improved = local_search_exact(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # Randomized restarts
    for _ in range(300):
        if time.time() > deadline:
            break
        order = candidates[:]
        random.shuffle(order)
        s = greedy_exact(order)
        if len(s) > len(best_set):
            best_set = s
        if time.time() < deadline:
            improved = local_search_exact(best_set, max_iters=2000)
            if len(improved) > len(best_set):
                best_set = improved

    # Final local search with more iterations
    if time.time() < deadline:
        improved = local_search_exact(best_set, max_iters=10000)
        if len(improved) > len(best_set):
            best_set = improved

    return list(best_set)
