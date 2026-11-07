"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)  # 255 elements
    
    # Prime factorization for each candidate
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    factorizations = [factorize(x) for x in candidates]
    
    # Check if a divides b*c using prime factor exponents
    # a | b*c iff for all primes p: exp_p(a) <= exp_p(b) + exp_p(c)
    def divides_product(a_idx, b_idx, c_idx):
        fa = factorizations[a_idx]
        fb = factorizations[b_idx]
        fc = factorizations[c_idx]
        for p, ea in fa.items():
            eb = fb.get(p, 0)
            ec = fc.get(p, 0)
            if ea > eb + ec:
                return False
        return True
    
    # Build conflict: for each pair (b_idx, c_idx) with b_idx != c_idx,
    # find all a_idx != b_idx, a_idx != c_idx such that a | b*c
    # conflict_mask[a_idx] = set of (b_idx, c_idx) pairs that conflict with a_idx
    # But that's too large. Instead, for each a, store which pairs (b,c) conflict.
    
    # Better approach: for each a, compute the set of indices that are "dangerous"
    # Actually, let's think differently.
    # For a fixed a, the constraint is violated if there exist distinct b,c in S\{a} with a|b*c.
    # So a is "safe" in S iff no pair (b,c) from S\{a} has a|b*c.
    
    # Precompute: for each pair (i,j), i<j, which a's are divisible by candidates[i]*candidates[j]?
    # That's expensive. Instead, for each a, precompute the set of (i,j) pairs where a | candidates[i]*candidates[j]
    
    # Let's use a different approach: greedy with local search
    # Start with all candidates, then remove elements that cause conflicts
    # Then try to add back removed elements
    
    # First, let's identify which elements are "most constrained"
    # An element a is constrained if many pairs (b,c) satisfy a|b*c
    
    # Precompute conflict pairs for each element
    # conflict_pairs[a_idx] = set of frozenset({b_idx, c_idx}) where a|b*c and b!=c, b!=a, c!=a
    # This is O(n^3) which is 255^3 ~ 16M, might be tight but doable
    
    # Actually, let's be smarter. For each a, we only need to know if ANY pair in the current set conflicts.
    # So let's use a greedy approach with efficient checking.
    
    # Strategy: 
    # 1. Start with all elements
    # 2. Repeatedly find an element with the most conflicts and remove it
    # 3. Then try to add back removed elements (local search)
    
    # Precompute: for each pair (i, j), i < j, compute the product's prime factorization
    # and which a's divide that product
    
    # Actually, let's just do a direct approach with bitmasks for efficiency
    # n = 255, so we can use Python integers as bitmasks
    
    # For each a_idx, compute conflict_mask[a_idx] as a bitmask of size n
    # where bit j is set if... hmm, the constraint involves pairs, not single elements.
    
    # Let me think again. The constraint is: for distinct a,b,c in S, a ∤ b*c.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Greedy approach:
    # 1. Start with S = all candidates
    # 2. While there's a violation, remove the element involved in the most violations
    # 3. Local search: try swapping one in, one out
    
    # To check violations efficiently, for each a in S, check if any pair (b,c) in S\{a} has a|b*c
    
    # Let's precompute for each a, the list of pairs (b,c) that conflict with a
    # conflict_pairs[a] = list of (b, c) where b < c, b != a, c != a, and a | candidates[b]*candidates[c]
    
    print(f"Precomputing conflict pairs...", flush=True)
    
    conflict_pairs = [[] for _ in range(n)]
    
    for a in range(n):
        fa = factorizations[a]
        for b in range(n):
            if b == a:
                continue
            fb = factorizations[b]
            for c in range(b + 1, n):
                if c == a:
                    continue
                fc = factorizations[c]
                # Check if a | candidates[b] * candidates[c]
                ok = True
                for p, ea in fa.items():
                    eb = fb.get(p, 0)
                    ec = fc.get(p, 0)
                    if ea > eb + ec:
                        ok = False
                        break
                if ok:
                    conflict_pairs[a].append((b, c))
    
    print(f"Done. Max conflicts: {max(len(cp) for cp in conflict_pairs)}", flush=True)
    
    # Greedy removal
    S = set(range(n))
    
    def has_violation(a, S_set):
        """Check if a has any conflict pair within S_set (excluding a itself)"""
        for (b, c) in conflict_pairs[a]:
            if b in S_set and c in S_set:
                return True
        return False
    
    # Iteratively remove the most conflicted element
    changed = True
    while changed:
        changed = False
        worst = -1
        worst_count = 0
        for a in S:
            count = 0
            for (b, c) in conflict_pairs[a]:
                if b in S and c in S:
                    count += 1
                    if count >= worst_count:
                        break
            if count > worst_count:
                worst_count = count
                worst = a
        if worst_count > 0:
            S.discard(worst)
            changed = True
    
    print(f"After greedy: {len(S)}", flush=True)
    
    # Local search: try to add back removed elements
    removed = list(set(range(n)) - S)
    improved = True
    while improved:
        improved = False
        for r in removed:
            # Try adding r
            new_S = S | {r}
            # Check if adding r creates any violation
            valid = True
            for a in new_S:
                if a == r:
                    continue
                for (b, c) in conflict_pairs[a]:
                    if b in new_S and c in new_S:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S = new_S
                removed.remove(r)
                improved = True
                print(f"Added back {candidates[r]}, size now {len(S)}", flush=True)
    
    # Also try swaps: remove one from S, add one from removed
    for r in list(removed):
        for s in list(S):
            # Check if removing s and adding r is valid
            new_S = (S - {s}) | {r}
            valid = True
            for a in new_S:
                for (b, c) in conflict_pairs[a]:
                    if b in new_S and c in new_S:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(new_S) > len(S):
                S = new_S
                removed.remove(r)
                improved = True
                break
        if improved:
            break
    
    print(f"Final size: {len(S)}", flush=True)
    
    # Convert back to actual values
    result = [candidates[i] for i in sorted(S)]
    
    # Verify
    def verify(result_list):
        for i in range(len(result_list)):
            for j in range(len(result_list)):
                if i == j:
                    continue
                for k in range(len(result_list)):
                    if i == k or j == k:
                        continue
                    a, b, c = result_list[i], result_list[j], result_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not verify(result):
        # Fallback to primes
        result = [a for a in range(2, N+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    if time.time() - start_time > DEADLINE:
        # Shouldn't happen, but safety
        pass
    
    return result
# EVOLVE-BLOCK-END
