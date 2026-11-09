"""Branch-and-bound maximum independent set in 3-uniform hypergraph."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)
    # Index mapping: candidate value -> index 0..254
    val_to_idx = {v: i for i, v in enumerate(all_candidates)}
    idx_to_val = all_candidates  # idx_to_val[i] = i+2

    # Build conflict structure:
    # For each pair (i,j) with i<j, determine which k (k!=i,j) conflict with {i,j}
    # Conflict: for distinct a,b,c in S, a ∤ b*c
    # So triple {i,j,k} is invalid if any of:
    #   idx_to_val[i] | (idx_to_val[j] * idx_to_val[k])
    #   idx_to_val[j] | (idx_to_val[i] * idx_to_val[k])
    #   idx_to_val[k] | (idx_to_val[i] * idx_to_val[j])
    
    # conflict_mask[i][j] = bitmask of k values (k != i, j) such that {i,j,k} is invalid
    # We'll use Python integers as bitmasks (256 bits fits easily)
    
    # Precompute: for each pair (i,j), the set of k that conflict
    # conflict_with_pair[i][j] = set of k such that {i,j,k} violates the condition
    conflict_with_pair = [[None] * n_total for _ in range(n_total)]
    
    for i in range(n_total):
        vi = idx_to_val[i]
        for j in range(i + 1, n_total):
            vj = idx_to_val[j]
            prod_ij = vi * vj
            bad_ks = set()
            for k in range(n_total):
                if k == i or k == j:
                    continue
                vk = idx_to_val[k]
                # Check if triple {i,j,k} is invalid
                # i divides j*k?
                if (vj * vk) % vi == 0:
                    bad_ks.add(k)
                    continue
                # j divides i*k?
                if (vi * vk) % vj == 0:
                    bad_ks.add(k)
                    continue
                # k divides i*j?
                if prod_ij % vk == 0:
                    bad_ks.add(k)
                    continue
            conflict_with_pair[i][j] = bad_ks
            conflict_with_pair[j][i] = bad_ks
    
    # For efficient branching, precompute for each vertex i:
    # degree[i] = number of vertices j such that there exists k with {i,j,k} invalid
    # Actually, we need: given a partial set S, which vertices can still be added?
    
    # Key insight: A vertex v can be added to current set S iff:
    # For all pairs (a,b) in S, v is NOT in conflict_with_pair[a][b]
    # AND for all a in S, there's no b in S\{a} such that v conflicts with {a,b}
    # Wait, let me re-think.
    
    # The condition is: for every THREE DISTINCT members a,b,c in S, a ∤ b*c
    # So S is valid iff no triple {a,b,c} ⊆ S is "bad" (where bad means some element divides product of other two)
    
    # A vertex v can be added to S iff for every pair {a,b} ⊆ S, the triple {v,a,b} is NOT bad.
    # i.e., v ∉ conflict_with_pair[a][b] for all pairs {a,b} ⊆ S.
    
    # So we can maintain: for current set S, the set of "available" vertices = all vertices v such that
    # for all pairs {a,b} ⊆ S, v ∉ conflict_with_pair[a][b].
    
    # Let's precompute a faster representation.
    # For each vertex v, define: forbidden_pairs(v) = set of pairs {a,b} such that v ∈ conflict_with_pair[a][b]
    # Then v can be added to S iff no pair in S is in forbidden_pairs(v).
    
    # Alternative approach: maintain a "compatible" bitmask.
    # compat_mask[v] = bitmask of vertices w such that for any pair containing both v and w... 
    # Hmm, this is getting complex. Let me use a different approach.
    
    # Simpler: for branch and bound, at each node with current set S (as a bitmask),
    # the candidates that can be added are those v not in S such that
    # for all pairs {a,b} in S, v is not in conflict_with_pair[a][b].
    
    # To speed this up, precompute for each vertex v:
    # v_conflicts_with = for each pair {a,b}, whether v is in conflict_with_pair[a][b]
    # This is equivalent to: v is incompatible with S iff there exist a,b in S such that v ∈ conflict_with_pair[a][b]
    
    # Let's precompute: for each v, a list of pairs (a,b) such that v ∈ conflict_with_pair[a][b]
    # Then v is compatible with S iff none of these pairs is a subset of S.
    
    # For efficiency, let's precompute for each v:
    # incompatible_pairs_mask[v] = a set of (a,b) pairs
    # And maintain S as a set, checking if any incompatible pair is contained in S.
    
    # Even better: for each v, compute a bitmask of "witness pairs"
    # But checking if a pair is in S requires O(|S|) per pair...
    
    # Let me use a different strategy: 
    # Precompute for each vertex v, the set of vertices u such that there exists w where {v,u,w} is bad.
    # i.e., v is "2-conflict" with u if ∃w: v ∈ conflict_with_pair[u][w] or u ∈ conflict_with_pair[v][w]
    # But this isn't quite right either.
    
    # Let me just use the direct approach with bitmasks for the current set.
    
    # Represent S as a bitmask (Python int).
    # Precompute for each vertex v:
    # v_forbidden_mask[v] = bitmask of vertices u such that for some w, {v,u,w} is bad
    # Wait, that's not sufficient. We need: v can join S iff for all pairs {a,b} ⊆ S, {v,a,b} is good.
    
    # Let me precompute differently. For each vertex v, define:
    # v_bad_pairs[v] = set of (a,b) with a < b, a != v, b != v, such that v ∈ conflict_with_pair[a][b]
    # Then v is compatible with S iff no (a,b) in v_bad_pairs[v] has both a and b in S.
    
    v_bad_pairs = [set() for _ in range(n_total)]
    for i in range(n_total):
        for j in range(i + 1, n_total):
            for k in conflict_with_pair[i][j]:
                v_bad_pairs[k].add((i, j))
    
    # Now, for a given S (as a set or bitmask), v is compatible iff
    # for all (a,b) in v_bad_pairs[v], NOT (a in S AND b in S)
    
    # For branch and bound, we need to:
    # 1. Find candidates compatible with current S
    # 2. Compute upper bound on how many more we can add
    # 3. Branch on adding or not adding a candidate
    
    # Upper bound: the number of candidates compatible with S.
    # Better upper bound: clique cover or packing argument.
    # For now, let's use: upper_bound = len(compatible_candidates)
    # We can improve: if two candidates are mutually incompatible (i.e., they conflict with some pair in S together), 
    # we can't add both. But this is complex.
    
    # Let's start with a simple upper bound and see if it's fast enough.
    
    def is_compatible(v, S_mask):
        """Check if vertex v can be added to set represented by S_mask."""
        for (a, b) in v_bad_pairs[v]:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1:
                return False
        return True
    
    # Greedy initial solution
    def greedy_solution(order):
        """Build a solution greedily following the given order."""
        S_mask = 0
        S_set = set()
        for v in order:
            if is_compatible(v, S_mask):
                S_mask |= (1 << v)
                S_set.add(v)
        return S_mask, S_set
    
    # Try multiple greedy orderings
    best_mask = 0
    best_size = 0
    best_set = set()
    
    orders = [
        list(range(n_total)),  # ascending
        list(range(n_total - 1, -1, -1)),  # descending
    ]
    random.seed(42)
    for _ in range(30):
        shuffled = list(range(n_total))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        mask, s = greedy_solution(order)
        if len(s) > best_size:
            best_mask = mask
            best_size = len(s)
            best_set = s
    
    # Branch and bound
    def branch_and_bound(S_mask, candidates_mask, current_size):
        """
        S_mask: bitmask of vertices in current solution
        candidates_mask: bitmask of vertices that can potentially be added (compatible with S)
        current_size: number of vertices in current solution
        Returns: (best_mask, best_size) found in this subtree
        """
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 5:
            return best_mask, best_size
        
        # Update upper bound
        upper_bound = current_size + bin(candidates_mask).count('1')
        if upper_bound <= best_size:
            return best_mask, best_size
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_mask = S_mask
                best_size = current_size
            return best_mask, best_size
        
        # Select a vertex to branch on
        # Heuristic: pick vertex with fewest compatible candidates among remaining
        # Or just pick the first one for simplicity
        # Let's pick the vertex that appears in the most bad pairs among candidates (most constrained)
        
        # For efficiency, just pick the lowest-index candidate
        # Find lowest set bit
        v = (candidates_mask & (-candidates_mask)).bit_length() - 1
        
        # Branch 1: include v
        new_S_mask = S_mask | (1 << v)
        # Compute new candidates: those in candidates_mask that are compatible with new_S_mask
        new_candidates = 0
        temp_mask = candidates_mask
        while temp_mask:
            u = (temp_mask & (-temp_mask)).bit_length() - 1
            temp_mask &= temp_mask - 1
            if u != v and is_compatible(u, new_S_mask):
                new_candidates |= (1 << u)
        
        branch_and_bound(new_S_mask, new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        new_candidates2 = candidates_mask & ~(1 << v)
        branch_and_bound(S_mask, new_candidates2, current_size)
        
        return best_mask, best_size
    
    # Run branch and bound from the greedy solution
    # First, compute candidates compatible with best_mask
    init_candidates = 0
    for v in range(n_total):
        if not (best_mask >> v) & 1 and is_compatible(v, best_mask):
            init_candidates |= (1 << v)
    
    branch_and_bound(best_mask, init_candidates, best_size)
    
    # Final verification
    result = [idx_to_val[i] for i in range(n_total) if (best_mask >> i) & 1]
    
    # Verify validity
    def is_valid_set(s_list):
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not is_valid_set(result):
        # Fallback
        current = set()
        current_mask = 0
        for c in all_candidates:
            idx = val_to_idx[c]
            if is_compatible(idx, current_mask):
                current.add(c)
                current_mask |= (1 << idx)
        result = sorted(current)
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
