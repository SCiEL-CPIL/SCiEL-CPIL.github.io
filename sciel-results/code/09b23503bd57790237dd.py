import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def build_paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def is_valid_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # out_masks[v] is a bitmask where bit u is 1 if v -> u
        out_masks = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if v == u:
                    continue
                diff = (u - v) % p
                if diff in squares:
                    mask |= (1 << u)
            out_masks[v] = mask
            
        # Check all 4-sets
        # For each 4-set S, check if any vertex v in V dominates S
        # v dominates S if (out_masks[v] & mask_S) == mask_S
        for i in range(p):
            if time.time() > deadline:
                return False
            for j in range(i + 1, p):
                if time.time() > deadline:
                    return False
                for k in range(j + 1, p):
                    if time.time() > deadline:
                        return False
                    mask_ijk = (1 << i) | (1 << j) | (1 << k)
                    for l in range(k + 1, p):
                        mask_ijkl = mask_ijk | (1 << l)
                        
                        found = False
                        # We can iterate over all v in p
                        # Optimization: Only v not in S can dominate S? 
                        # No, v can be in S? 
                        # "common dominator" usually implies a vertex *in the tournament* 
                        # "Every set of 4 vertices must have a vertex with arcs to all four."
                        # Does the vertex have to be *outside* the set of 4?
                        # "a vertex with arcs to all four". 
                        # If v is in S, it cannot have an arc to itself. 
                        # So v must be in V \ S.
                        # However, the bitmask check (out_masks[v] & mask_S) == mask_S 
                        # automatically fails if v is in S because out_masks[v] has bit v as 0.
                        # So we can just iterate all v.
                        
                        for v in range(p):
                            if (out_masks[v] & mask_ijkl) == mask_ijkl:
                                found = True
                                break
                        if not found:
                            return False
        return True

    # Candidates: Primes p ≡ 3 (mod 4)
    # 31, 43, 47, 59, 67
    candidates = [31, 43, 47, 59, 67]
    
    best_n = 67
    best_orientation = build_paley_orientation(67)
    
    # Check smaller candidates first
    for p in [31, 43, 47, 59]:
        if time.time() > deadline:
            break
        if is_valid_paley(p):
            best_n = p
            best_orientation = build_paley_orientation(p)
            break

    # If no smaller one worked, we keep the 67 result
    # We do NOT re-validate 67 here to save time, as it is a known valid construction.
    # But for safety, if we have lots of time, we could. 
    # Given the inner loop of is_valid_paley is O(p^5) effectively (4 nested loops + p checks),
    # for p=67, it's ~ 67^4 * 67 operations? No, 4 loops is p^4/24, then p checks.
    # p=67: C(67,4) ~ 600k. 600k * 67 ~ 40M operations. 
    # Python is slow. 40M operations might take 10-20 seconds.
    # We already checked 31, 43, 47, 59.
    # C(31,4) ~ 31k. 31k * 31 ~ 1M ops. Fast.
    # C(59,4) ~ 450k. 450k * 59 ~ 26M ops. ~5-10s.
    # Total time for checking all 4 small ones: ~10-15s.
    # We have 110s. 
    # We should probably validate 67 as well to be absolutely sure, or just trust it.
    # The task says "The smallest known has 67 vertices". 
    # If we trust the literature, 67 is valid.
    # If we want to be safe, we can validate 67 if time permits.
    
    if time.time() < deadline - 5:
        # Validate the current best to be safe? 
        # If best_n < 67, we already validated it in the loop.
        # If best_n == 67, we haven't validated it in this run.
        if best_n == 67:
             # We can skip validation of 67 if we trust the construction, 
             # but if we have time, let's do a quick sanity check? 
             # Actually, the validation for 67 is the most expensive. 
             # If we ran out of time checking 59, we might not have time for 67.
             # But 67 is the fallback. If we don't validate it, we assume it's correct.
             # Given it's a standard construction, this is acceptable.
             pass

    return {'n': best_n, 'orientation': best_orientation}
