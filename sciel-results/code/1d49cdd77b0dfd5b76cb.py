"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

    # Fallback: Paley tournament on 67 vertices
    def get_paley_67():
        p = 67
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    best_result = get_paley_67()

    # Fast validity check and repair logic using bitmasks
    def build_masks(n, orientation):
        out_mask = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    out_mask[i] |= (1 << j)
                else:
                    # j -> i
                    out_mask[j] |= (1 << i)
        return out_mask

    def count_violations(n, out_mask):
        """
        Counts the number of 4-sets that do NOT have a common dominator.
        A 4-set {a,b,c,d} is valid if there exists v in V such that v->a, v->b, v->c, v->d.
        This is equivalent to: (In(a) & In(b) & In(c) & In(d)) != 0.
        Where In(x) is the set of vertices pointing to x.
        Note: v can be any vertex, including a,b,c,d themselves.
        If v=a, then a->b, a->c, a->d.
        If v is outside, v->a, v->b, v->c, v->d.
        
        Using out_mask:
        out_mask[v] has bit x set if v->x.
        So v is a dominator of {a,b,c,d} if (out_mask[v] & (1<<a) & (1<<b) & (1<<c) & (1<<d)) != 0.
        
        We can precompute for each 4-set if it's valid.
        To count violations efficiently:
        For each 4-set {a,b,c,d}, check if union of In(a), In(b), In(c), In(d) covers V? No.
        We need intersection of In(a), In(b), In(c), In(d) to be non-empty.
        
        In(x) = {v | v->x} = {v | out_mask[v] has bit x set}.
        
        Let's just iterate 4-sets. For n=60, C(60,4) ~ 487k.
        For each, we need to check if intersection of 4 in-neighborhoods is non-empty.
        In-neighborhood of x can be computed on the fly or precomputed.
        Precomputing In(x) as bitmask:
        in_mask[x] = 0
        for v in range(n):
            if (out_mask[v] >> x) & 1:
                in_mask[x] |= (1 << v)
        
        Then for 4-set {a,b,c,d}:
        if (in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d]) == 0: violation.
        
        This is O(n^4 * 4) bit operations.
        """
        in_mask = [0] * n
        for v in range(n):
            m = out_mask[v]
            # For each u that v points to, add v to in_mask[u]
            # This loop is O(n^2) total if done carefully, but simple loop is O(n^2)
            # Actually, iterating bits of m is faster if sparse, but here dense.
            # Simple way:
            for u in range(n):
                if (m >> u) & 1:
                    in_mask[u] |= (1 << v)
        
        violations = 0
        # Iterate 4-sets
        for a in range(n - 3):
            ma = in_mask[a]
            for b in range(a + 1, n - 2):
                mb = in_mask[b]
                inter_ab = ma & mb
                if inter_ab == 0: 
                    # If intersection of first two is empty, all 4-sets with these two are violations
                    # Number of pairs (c,d) with c>b, d>c is (n-b-1)*(n-b-2)/2
                    # Wait, c > b, d > c.
                    # c ranges from b+1 to n-2. d ranges from c+1 to n-1.
                    # Sum_{c=b+1}^{n-2} (n-1-c)
                    # Let k = n-1-c. As c goes b+1..n-2, k goes n-1-(b+1)..1 = n-b-2..1
                    # Sum = (n-b-2)(n-b-1)/2
                    violations += (n - b - 2) * (n - b - 1) // 2
                    continue
                for c in range(b + 1, n - 1):
                    mc = in_mask[c]
                    inter_abc = inter_ab & mc
                    if inter_abc == 0:
                        # d ranges from c+1 to n-1. Count is n-1-c
                        violations += (n - 1 - c)
                        continue
                    for d in range(c + 1, n):
                        md = in_mask[d]
                        if (inter_abc & md) == 0:
                            violations += 1
        return violations

    def flip_edge(n, orientation, i, j, out_mask, bit_val):
        """
        Flip edge between i and j.
        If bit was 0 (i->j), becomes 1 (j->i).
        If bit was 1 (j->i), becomes 0 (i->j).
        Updates orientation and out_mask.
        Returns new out_mask (modified in place for speed, but we need to be careful with local search state).
        Actually, for local search, we want to try a flip, check delta, and revert if bad.
        So we should return the changes or use a copy.
        Given n is small, copying out_mask (list of ints) is fast.
        """
        idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        # Formula for index of pair (i,j) in lexicographic order:
        # Number of pairs (k, l) with k < i is i*(i-1)/2.
        # Number of pairs (i, l) with l < j is j - i - 1.
        # Total index = i*(i-1)/2 + (j - i - 1)
        # Let's verify.
        # n=3. Pairs: (0,1), (0,2), (1,2).
        # i=0, j=1: 0*... + 0 = 0. Correct.
        # i=0, j=2: 0*... + 1 = 1. Correct.
        # i=1, j=2: 1*0/2 + 0 = 0? No.
        # Pairs for i=1: (1,2).
        # Pairs before i=1: (0,1), (0,2). Count = 2.
        # Formula: i*(i-1)/2 + (j-i-1)
        # i=1: 1*0/2 + (2-1-1) = 0. But index should be 2.
        # Correct formula:
        # Pairs (k, l) with k < i: sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2.
        # Pairs (i, l) with l < j: j - i - 1.
        # Index = i(n-1) - i(i-1)/2 + j - i - 1
        
        idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        current_bit = orientation[idx]
        
        if current_bit == 0:
            # i -> j becomes j -> i
            out_mask[i] &= ~(1 << j)
            out_mask[j] |= (1 << i)
            orientation[idx] = 1
        else:
            # j -> i becomes i -> j
            out_mask[j] &= ~(1 << i)
            out_mask[i] |= (1 << j)
            orientation[idx] = 0

    def try_local_search(n, max_iters=2000, restarts=5):
        """
        Random restart local search.
        """
        for r in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize random tournament
            num_edges = n * (n - 1) // 2
            orientation = [random.randint(0, 1) for _ in range(num_edges)]
            out_mask = build_masks(n, orientation)
            
            current_violations = count_violations(n, out_mask)
            
            if current_violations == 0:
                return {'n': n, 'orientation': orientation}
            
            best_violations = current_violations
            best_orientation = orientation[:]
            best_out_mask = out_mask[:]
            
            no_improvement_count = 0
            max_no_improvement = 50
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Pick a random edge to flip
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                
                # Save state
                old_out_mask = out_mask[:]
                old_orientation_bit = orientation[i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)]
                
                # Flip
                flip_edge(n, orientation, i, j, out_mask, old_orientation_bit)
                
                # Check violations
                new_violations = count_violations(n, out_mask)
                
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_orientation = orientation[:]
                    best_out_mask = out_mask[:]
                    no_improvement_count = 0
                    if new_violations == 0:
                        return {'n': n, 'orientation': orientation}
                else:
                    # Revert
                    out_mask[:] = old_out_mask
                    orientation[i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)] = old_orientation_bit
                    no_improvement_count += 1
                
                if no_improvement_count > max_no_improvement:
                    break
            
            # If we found a better solution in this restart, use it as start for next?
            # Or just restart from scratch? Restart from scratch is simpler.
            # But if we found a significantly better one, maybe continue from there?
            # Let's stick to full restarts for simplicity and robustness.
            
        return None

    # Search for n < 67
    # We try a few values. The probability of random tournament working is low.
    # Local search might help.
    # We prioritize smaller n.
    
    # Try n=66 first? Or start from 31?
    # Starting from 31 is hard.
    # Let's try a range.
    
    # Given time constraints, we can only afford to try a few n values with limited iterations.
    # Let's try n=66, 65, 64.
    
    # Actually, let's try to be smarter.
    # If we find a valid tournament for n, we return it.
    
    # We will allocate time budget.
    # 10s for n=66, 10s for n=65, etc.
    
    for n in [66, 65, 64, 63]:
        if time.time() > deadline - 5.0:
            break
        
        # Adjust max_iters based on time remaining?
        # Just run try_local_search with limited restarts.
        
        # For n=66, C(66,4) is ~ 736k. Counting violations takes ~ 1-2s?
        # Let's assume it's fast enough for a few iterations.
        
        res = try_local_search(n, max_iters=100, restarts=2)
        if res:
            return res
            
    # Fallback
    return best_result

# EVOLVE-BLOCK-END
