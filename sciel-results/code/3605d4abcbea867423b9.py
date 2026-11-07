import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for the 60s limit (or 90s if that's the hard limit, but 55s is safe)
    
    # 1. Precomputation: Sieve for smallest prime factors (SPF) and Square-Free Kernels (Masks)
    max_val = 10210
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute masks for all numbers up to max_val
    # mask[x] represents the parity of exponents in the prime factorization of x
    # We use the prime value itself as the bit position. Since max prime < 10210, 
    # the integer size is manageable (approx 1.3KB).
    masks = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        n = x
        m = 0
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                m |= (1 << p)
        masks[x] = m
        
    # Target mask for 210
    # 210 = 2 * 3 * 5 * 7
    target_mask = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    best_val = float('inf')
    best_list = None
    
    # Helper to update best if valid
    def try_update(lst):
        nonlocal best_val, best_list
        if not lst:
            return
        # Check constraints
        if len(set(lst)) != len(lst):
            return
        if any(n <= 210 or n > 10210 for n in lst):
            return
        
        # Verify perfect square condition exactly
        prod = 1
        for n in lst:
            prod *= n
        val_check = 210 * prod
        s = math.isqrt(val_check)
        if s * s != val_check:
            return
            
        v = max(lst) - 210
        if v < best_val:
            best_val = v
            best_list = [int(n) for n in lst]

    # 2. Check k=1
    # x = 210 * k^2
    # 210 < x <= 10210
    # k=1 -> 210 (invalid)
    # k=2 -> 840
    x1 = 210 * 4
    if 210 < x1 <= 10210:
        try_update([x1])

    # 3. Check k=2
    # x1 * x2 = 210 * m^2
    # We want min(max(x1, x2))
    # x1, x2 > 210, distinct
    # Iterate m upwards. Smaller m -> smaller product -> smaller max.
    m_max = int(math.sqrt(10210**2 / 210))
    
    # Optimization: We can stop searching for k=2 if the theoretical minimum max for current m
    # is already >= best_val.
    # min max for product P is approx sqrt(P).
    # If math.isqrt(210*m*m) - 210 >= best_val, then no pair with this m can improve best_val.
    
    for m in range(15, m_max + 1):
        if time.time() > deadline:
            break
        
        P = 210 * m * m
        # Theoretical lower bound for max(x1, x2) is sqrt(P)
        # If even the best case (x1=x2=sqrt(P)) doesn't beat best_val, skip.
        # Note: x1 != x2, so max > sqrt(P).
        # If isqrt(P) - 210 >= best_val, then max >= isqrt(P) > 210 + best_val - 1, so max-210 >= best_val.
        # Actually, if isqrt(P) - 210 >= best_val, then max >= isqrt(P) (since integers) -> max-210 >= best_val.
        # So we can't improve.
        if math.isqrt(P) - 210 >= best_val:
            continue
            
        limit = int(math.isqrt(P))
        if limit < 211:
            continue
            
        # Find divisor d of P such that 210 < d <= limit and P/d > 210 and d != P/d
        # We want to minimize max(x1, x2) = P/d.
        # This is minimized when d is as large as possible (close to sqrt(P)).
        # So iterate d downwards from limit to 211.
        # First valid d found gives the smallest max for this m.
        
        found = False
        for d in range(limit, 210, -1):
            if P % d == 0:
                x2 = P // d
                if x2 > 210 and x2 != d:
                    try_update([d, x2])
                    found = True
                    break # Largest d gives smallest max, so no need to check smaller d for this m

    # 4. Check k=3 using Gaussian Elimination on Masks
    # We want a subset S of {211, ..., M} such that XOR(mask[s] for s in S) == target_mask.
    # We want to minimize max(S).
    # We can iterate M from 211 upwards and maintain a linear basis of masks for {211, ..., M}.
    # Once target_mask is in the span, we have a solution.
    # The solution found will have max element <= M.
    # Since we iterate M upwards, the first time we find a solution, M is minimized?
    # Not necessarily. The basis might use elements < M, but the "max" of the specific combination
    # might be less than M. However, if we just want *a* solution with max <= M, and we check increasing M,
    # the first M where a solution exists gives the minimal possible max?
    # Yes, because if a solution exists with max <= M, it exists for all M' >= M.
    # The minimal M such that a solution exists is the minimal max value.
    # Wait, the solution must use distinct integers from {211..M}.
    # The Gaussian elimination finds *a* linear combination.
    # Does it guarantee the max element in the combination is exactly M?
    # No. But if a solution exists in {211..M}, it might use only elements from {211..M-1}.
    # If it used only {211..M-1}, then the solution would have been found at M-1.
    # So if we check M=211, 212, ..., and return the first M where target is in span,
    # the combination found might have max < M.
    # But that's fine! We just need *a* valid set.
    # However, we want to minimize max(selected).
    # If we find a solution at M, its max is <= M.
    # If we found a solution at M-1, we would have returned earlier.
    # So the first M where target is in span gives a solution with max <= M.
    # Is it possible that the solution found at M has max < M, but a better solution (smaller max)
    # exists at M' < M that we missed?
    # No, because we check all M' < M. If a solution with max <= M' existed, target would be in span of {211..M'}.
    # So the first M where target is in span corresponds to the minimal possible max value?
    # Actually, the max value of the specific combination found might be less than M, but it's <= M.
    # And no combination with max < M exists (otherwise we would have found it at M-1 or earlier).
    # So max(combination) is exactly M? Or could it be < M?
    # If max(combination) < M, then the combination is in {211..M-1}, so target would be in span of {211..M-1},
    # and we would have found it at M-1.
    # Therefore, if we return at M, the max element of the combination MUST be M.
    # (Unless the combination is empty? No, target != 0).
    # So iterating M upwards and returning the first valid combination gives the minimal max.
    
    # We only need to do this if we haven't found a solution with max < 211 (impossible) 
    # or if we want to beat the current best_val.
    # Current best_val is likely 30 (from k=2, x=240? No, 210*15^2 = 47250. sqrt(47250) ~ 217. 
    # Let's check k=2 result from current program.
    # m=15, P=47250. sqrt(P) ~ 217.
    # Divisors of 47250 near 217.
    # 47250 / 225 = 210. (210 not > 210)
    # 47250 / 210 = 225. (210 not > 210)
    # 47250 / 250 = 189.
    # 47250 / 189 = 250.
    # Pair (189, 250)? 189 < 210. Invalid.
    # Let's check m=16. P = 210 * 256 = 53760. sqrt ~ 231.
    # 53760 / 240 = 224. Pair (224, 240). Both > 210. Distinct.
    # max = 240. val = 30.
    # So best_val is likely 30.
    
    # We only need to search k=3 if we can find max < 240 (val < 30).
    # So we only need to check M up to 239.
    
    if best_val > 29: # Only if we can beat 30
        # We need to check M from 211 to min(239, max_val)
        # But wait, the combination must have size >= 3?
        # The problem says "1..100 distinct integers".
        # Gaussian elimination finds *a* combination. It might have size 1, 2, etc.
        # If it finds a combination of size 1: x1. Then 210*x1 = square.
        # This is the k=1 case. We already checked k=1.
        # If it finds a combination of size 2: x1, x2.
        # This is the k=2 case. We already checked k=2.
        # So if we find a solution at M < 240 via Gaussian Elimination,
        # and it has size 1 or 2, we would have found it in previous steps.
        # So any new solution found here must have size >= 3.
        # This is consistent with "k=3" in my previous thought process, but really it's "any k".
        
        basis = []
        combos = []
        pivot_map = {}
        
        # We iterate M from 211 to min(239, max_val)
        # But we also need to consider that the combination might use numbers > 239?
        # No, we are iterating M, so all numbers in the basis are <= M.
        # So the combination will only use numbers <= M.
        
        for M in range(211, min(240, max_val) + 1):
            if time.time() > deadline:
                break
                
            i = M - 211
            m = masks[M]
            c = {i}
            
            # Reduce m using existing basis
            while m != 0:
                hb = m.bit_length() - 1
                if hb in pivot_map:
                    idx = pivot_map[hb]
                    m ^= basis[idx]
                    c ^= combos[idx]
                else:
                    break
            
            if m != 0:
                hb = m.bit_length() - 1
                new_idx = len(basis)
                basis.append(m)
                combos.append(c)
                pivot_map[hb] = new_idx
                
            # Check if target is in span
            t = target_mask
            tc = set()
            while t != 0:
                hb = t.bit_length() - 1
                if hb in pivot_map:
                    idx = pivot_map[hb]
                    t ^= basis[idx]
                    tc ^= combos[idx]
                else:
                    break
            
            if t == 0:
                # Solution found
                # tc contains indices relative to 211
                # Convert to numbers
                selected_nums = [idx + 211 for idx in tc]
                
                # Verify
                # Check distinct and range
                if len(set(selected_nums)) == len(selected_nums):
                    if all(210 < n <= 10210 for n in selected_nums):
                        # We know max <= M <= 239
                        # So val <= 29
                        # This will beat best_val (which is > 29)
                        try_update(selected_nums)
                        # Since we iterate M upwards, this is the minimal max.
                        # We can break?
                        # Yes, because any further M will have max >= current M.
                        # And we want to minimize max.
                        # So we found the best possible max for k>=3 (within this range).
                        # And since we already checked k=1,2, this is the global best.
                        break

    if best_list is None:
        return [840]
        
    return best_list
