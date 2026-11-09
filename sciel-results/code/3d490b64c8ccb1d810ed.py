import random

def construct():
    n = 512
    max_size = 48
    
    def is_valid_set(selected, candidate):
        """
        Check if adding candidate to selected keeps the set valid.
        1. Fast check for 3-term arithmetic progressions (subset size 2).
        2. Full bitset check for larger subset sizes.
        """
        if not selected:
            return True
            
        selected_set = set(selected)
        m = len(selected)
        
        # 1. Fast check: 3-term APs
        # Case A: candidate is the mean of two elements in selected
        # 2 * candidate = u + v  => v = 2 * candidate - u
        for u in selected:
            v = 2 * candidate - u
            if v in selected_set:
                # Ensure u != v and u != candidate and v != candidate
                # Since elements are distinct in selected, u != v.
                # If u == candidate, v == candidate, but candidate not in selected yet.
                # If v == candidate, u == candidate.
                # So just check if v is in selected.
                # We must ensure the subset has at least 2 DISTINCT OTHER elements.
                # The subset is {u, v}.
                # If u == v, it's not a subset of distinct elements? 
                # "subset of at least two distinct OTHER elements".
                # So u and v must be distinct.
                if u != v:
                    return False
        
        # Case B: an element a in selected is the mean of candidate and another element b in selected
        # 2 * a = candidate + b => b = 2 * a - candidate
        for a in selected:
            b = 2 * a - candidate
            if b in selected_set:
                if b != candidate: # b is in selected, candidate is not, so b != candidate
                    # Subset is {candidate, b}. Distinct? Yes.
                    # Are they OTHER elements relative to a? Yes.
                    return False
        
        # 2. Full check for subset sizes >= 3
        # We need to check if candidate is mean of subset T of selected (|T|>=2)
        # AND if any a in selected is mean of subset T' of (selected \ {a}) U {candidate}
        
        # Check 1: candidate is mean of T subset of selected
        # sum(T) = |T| * candidate
        # Use bitset DP
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in selected:
            for j in range(m - 1, -1, -1):
                if dp[j]:
                    dp[j + 1] |= (dp[j] << x)
        
        for k in range(2, m + 1):
            target = k * candidate
            if (dp[k] >> target) & 1:
                return False
                
        # Check 2: a in selected is mean of T' subset of (selected \ {a}) U {candidate}
        # T' must include candidate? 
        # If T' does not include candidate, it's already checked in previous validity of 'selected'.
        # So we only need to check subsets T' that INCLUDE candidate.
        # Let T' = {candidate} U T'' where T'' subset of selected \ {a}.
        # |T'| = 1 + |T''|.
        # sum(T') = candidate + sum(T'') = (1 + |T''|) * a.
        # sum(T'') = (1 + k) * a - candidate, where k = |T''|.
        # k can be 0? 
        # If k=0, T'' is empty. T' = {candidate}. Size 1.
        # Problem says "subset of at least two OTHER elements".
        # So T' must have size >= 2.
        # Since candidate is in T', and T' is a subset of OTHERS relative to a,
        # T' can be {candidate, b} (size 2, k=1) or larger.
        # So k >= 1.
        
        for a in selected:
            others = [x for x in selected if x != a]
            m_o = len(others)
            if m_o < 1:
                continue
                
            dp_o = [0] * (m_o + 1)
            dp_o[0] = 1
            for x in others:
                for j in range(m_o - 1, -1, -1):
                    if dp_o[j]:
                        dp_o[j + 1] |= (dp_o[j] << x)
                        
            for k in range(1, m_o + 1):
                target_sum = (k + 1) * a - candidate
                if target_sum < 0:
                    continue
                if (dp_o[k] >> target_sum) & 1:
                    return False
                    
        return True

    best_set = []
    candidates = list(range(1, n + 1))
    
    # Deterministic orders
    orders = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
        list(range(n, 0, -2)) + list(range(n - 1, 0, -2)),
        list(range(n // 2 + 1, n + 1)) + list(range(1, n // 2 + 1))
    ]
    
    # Random shuffles
    for _ in range(20):
        random.shuffle(candidates)
        orders.append(candidates[:])
        
    for order in orders:
        selected = []
        for x in order:
            if len(selected) >= max_size:
                break
            if is_valid_set(selected, x):
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= max_size:
                break
                
    return best_set[:max_size]
