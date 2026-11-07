# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _squarefree_signature(n):
    """Return frozenset of primes with odd exponent in n."""
    factors = _factorize(n)
    sig = frozenset(p for p, e in factors.items() if e % 2 == 1)
    return sig

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    target_n = P['n']  # 210
    target_sig = _squarefree_signature(target_n)  # frozenset({2,3,5,7})
    
    min_val = target_n + 1  # 211
    max_val = target_n + P['maximum_window']  # 10210
    
    # Precompute signatures for all integers in range
    sigs = {}
    for i in range(min_val, max_val + 1):
        sigs[i] = _squarefree_signature(i)
    
    # Build mapping from signature to list of integers that have it
    # We'll grow this incrementally as M increases
    # For each M, we need to check if target_sig can be formed by XOR of 
    # distinct elements from [min_val, M]
    
    # Strategy: iterate M from min_val upward. For each M, maintain 
    # a dict: sig -> list of ints in [min_val, M] with that signature.
    # Then check if target_sig can be expressed as XOR of 1-5 distinct sigs.
    
    sig_to_nums = defaultdict(list)
    
    for M in range(min_val, max_val + 1):
        if time.time() > deadline:
            break
        
        # Add M to the pool
        sig_M = sigs[M]
        sig_to_nums[sig_M].append(M)
        
        # Now check if we can form target_sig using distinct integers from [min_val, M]
        # Try sizes 1, 2, 3, 4, 5
        
        found = None
        
        # Size 1: need target_sig in sig_to_nums
        if target_sig in sig_to_nums:
            found = [target_sig]  # just the signature, we'll pick the actual number later
        
        # Size 2: need sig_a ^ sig_b = target_sig
        if found is None:
            # Iterate over unique signatures present
            present_sigs = list(sig_to_nums.keys())
            for sa in present_sigs:
                sb = target_sig ^ sa
                if sb in sig_to_nums:
                    # Need distinct numbers
                    # If sa == sb, need at least 2 numbers with that sig
                    if sa == sb:
                        if len(sig_to_nums[sa]) >= 2:
                            found = [sa, sb]
                    else:
                        found = [sa, sb]
                    break
        
        # Size 3: need sig_a ^ sig_b ^ sig_c = target_sig
        if found is None:
            present_sigs = list(sig_to_nums.keys())
            n_sigs = len(present_sigs)
            for i in range(n_sigs):
                sa = present_sigs[i]
                for j in range(i, n_sigs):  # allow same sig if enough numbers
                    sb = present_sigs[j]
                    sc = target_sig ^ sa ^ sb
                    if sc in sig_to_nums:
                        # Check we can pick distinct numbers
                        # Count how many of sa, sb, sc are the same
                        from collections import Counter
                        sig_counts = Counter([sa, sb, sc])
                        ok = True
                        for s, cnt in sig_counts.items():
                            if len(sig_to_nums[s]) < cnt:
                                ok = False
                                break
                        if ok:
                            found = [sa, sb, sc]
                            break
                if found:
                    break
        
        # Size 4: need sig_a ^ sig_b ^ sig_c ^ sig_d = target_sig
        if found is None:
            present_sigs = list(sig_to_nums.keys())
            n_sigs = len(present_sigs)
            # Only do this if n_sigs is small enough
            if n_sigs <= 200:
                for i in range(n_sigs):
                    sa = present_sigs[i]
                    for j in range(i, n_sigs):
                        sb = present_sigs[j]
                        for k in range(j, n_sigs):
                            sc = present_sigs[k]
                            sd = target_sig ^ sa ^ sb ^ sc
                            if sd in sig_to_nums:
                                from collections import Counter
                                sig_counts = Counter([sa, sb, sc, sd])
                                ok = True
                                for s, cnt in sig_counts.items():
                                    if len(sig_to_nums[s]) < cnt:
                                        ok = False
                                        break
                                if ok:
                                    found = [sa, sb, sc, sd]
                                    break
                            if time.time() > deadline:
                                break
                        if found or time.time() > deadline:
                            break
                    if found or time.time() > deadline:
                        break
        
        if found is not None:
            # Now pick actual distinct integers from the signatures
            # We need to select one number per signature, ensuring all are distinct
            # Since signatures are frozensets, multiple numbers can share a signature
            # We need to pick distinct numbers
            
            # Build a list of (signature, [available numbers])
            # Greedily pick: for each signature in 'found', pick an unused number
            used = set()
            selected = []
            # Sort by number of available choices (ascending) to handle tight cases first
            sig_list = list(found)
            # Sort by length of available numbers
            sig_list.sort(key=lambda s: len([x for x in sig_to_nums[s] if x not in used]))
            
            success = True
            for s in sig_list:
                picked = None
                for num in sig_to_nums[s]:
                    if num not in used:
                        picked = num
                        break
                if picked is None:
                    success = False
                    break
                used.add(picked)
                selected.append(picked)
            
            if success:
                # Verify
                prod = target_n
                for v in selected:
                    prod *= v
                if _is_perfect_square(prod):
                    return [int(x) for x in selected]
                else:
                    # Should not happen, but try next M
                    continue
            else:
                # Couldn't pick distinct numbers, try next M
                continue
    
    # Fallback
    return [224, 240]
# EVOLVE-BLOCK-END
