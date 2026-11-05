"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _paley_tournament(p):
    """
    Returns orientation list for Paley tournament on p vertices.
    p must be a prime congruent to 3 mod 4.
    Orientation bit for pair (i, j) with i < j:
    0 means i -> j
    1 means j -> i
    """
    # Precompute squares modulo p
    squares = set((x * x) % p for x in range(1, p))
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            # In Paley tournament, i -> j if (j-i) is a quadratic residue mod p
            # Here 0 means i -> j, 1 means j -> i
            if d in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

def _fast_check_property(n, orientation):
    """
    Check if every 4-set has a common dominator using bitmasks.
    Returns True if valid.
    """
    import numpy as np
    
    out_mask = np.zeros(n, dtype=np.uint64)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
            else:
                out_mask[j] |= (1 << i)
    
    # Iterate over 4-sets
    # For n=67, this is ~750k iterations.
    # NumPy doesn't help much with the loop structure unless we vectorize the check,
    # which is hard for combinations.
    # We stick to Python loop but ensure it's tight.
    
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    mask_abcd = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    
                    # Check if any of a,b,c,d dominates the others
                    # a dominates b,c,d?
                    if not ((out_mask[a] & (mask_abcd ^ (1 << a))) == (mask_abcd ^ (1 << a)) or
                            (out_mask[b] & (mask_abcd ^ (1 << b))) == (mask_abcd ^ (1 << b)) or
                            (out_mask[c] & (mask_abcd ^ (1 << c))) == (mask_abcd ^ (1 << c)) or
                            (out_mask[d] & (mask_abcd ^ (1 << d))) == (mask_abcd ^ (1 << d))):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Candidate primes p ≡ 3 mod 4, sorted by n
    # We want the smallest valid n.
    # Lower bound is 31.
    # Known smallest is 67.
    # Let's try primes in range [31, 90] that are ≡ 3 mod 4.
    # Primes: 31, 43, 47, 59, 67, 71, 79, 83.
    # Note: 31 % 4 = 3, 43 % 4 = 3, 47 % 4 = 3, 59 % 4 = 3, 67 % 4 = 3, 71 % 4 = 3, 79 % 4 = 3, 83 % 4 = 3.
    
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    last_result = None
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        res = _paley_tournament(p)
        last_result = res
        
        # Verify with fast checker
        # For n=31, C(31,4) = 31465 checks. Very fast.
        # For n=67, C(67,4) = 751,450 checks. Still fast with bitmasks (< 1s usually).
        if _fast_check_property(p, res['orientation']):
            return res
            
    # If loop finished or timed out, return the last one generated.
    # If we timed out before checking a valid one, we might have an invalid one in last_result.
    # However, the loop only returns if valid.
    # If we exit the loop due to deadline, we return last_result which might be invalid.
    # To be safe, if we haven't returned a valid result, we should return the known valid 67.
    # But wait, if 67 was in the list and we didn't reach it, last_result is smaller and invalid.
    # We should prioritize returning a KNOWN valid result if the search fails to find a smaller one in time.
    
    # The known valid result is Paley(67).
    # If we haven't returned yet, it means either:
    # 1. All candidates up to deadline were invalid.
    # 2. We timed out.
    
    # Let's ensure we return a valid tournament.
    # If last_result is None, generate 67.
    # If last_result is not None but we didn't verify it (because we broke out of loop? No, we verify inside loop).
    # Actually, if we break out of the loop, last_result is the one we just generated but maybe didn't finish verifying?
    # No, _fast_check_property is synchronous. If it returns False, we continue. If True, we return.
    # So if we exit the loop, last_result is the last one we generated and verified as INVALID (or we didn't get to verify it if we broke *before* generating? No, we check time *before* generating).
    
    # Wait, the check `if time.time() > deadline` is at the start of the loop.
    # So if we are at p=67 and time is close to deadline, we generate and check.
    # If check takes too long, we might hit the kill signal.
    # But _fast_check_property for 67 is fast.
    
    # The main risk is if we break out of the loop, last_result is the last *attempted* one.
    # If we didn't return it, it was invalid.
    # So returning last_result would be returning an invalid tournament.
    
    # Strategy:
    # 1. Try candidates in order.
    # 2. If any is valid, return it.
    # 3. If loop finishes without finding a valid one (or times out), return the known valid Paley(67).
    
    # However, if we timed out *during* the check of 67, we might not have a valid result stored.
    # But 67 is known to be valid.
    
    # Let's refine the fallback.
    # If we didn't return inside the loop, we don't have a verified valid result from the loop.
    # We should return the known valid Paley(67).
    
    # But what if 67 was already checked and found invalid (impossible, but theoretically)?
    # Then we'd return 67 again.
    
    # What if a smaller one was valid? We would have returned it.
    
    # So the fallback is simply: return Paley(67).
    
    # One edge case: What if 67 is not in the candidate list? It is.
    # What if we didn't reach 67 due to timeout? Then we return 67 anyway.
    
    # So:
    return _paley_tournament(67)
# EVOLVE-BLOCK-END
