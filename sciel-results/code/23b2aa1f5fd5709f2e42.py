import time
from math import gcd

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums."""
    if not S:
        return True
    n = len(S)
    sums = [(0, 1)]
    for a in S:
        new_sums = []
        for (num, den) in sums:
            new_num = num * a + den
            new_den = den * a
            g = gcd(new_num, new_den)
            new_sums.append((new_num // g, new_den // g))
        sums.extend(new_sums)
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    deadline = time.time() + 80
    
    # Compute LCM of 1..22
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    
    # Integer coefficients: c[i] = lcm_val // (i+1) for element i+1
    # So 1/(i+1) = c[i] / lcm_val
    c = [lcm_val // (i+1) for i in range(22)]
    
    # Try MILP approach
    best = []
    
    # First do a quick greedy to get a starting point
    def greedy_add(order):
        current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            if _is_dissociated(test):
                current = test
        return current
    
    import random
    random.seed(42)
    
    # Quick greedy to get initial best
    for order in [list(range(1,23)), list(range(22,0,-1))]:
        result = greedy_add(order)
        if len(result) > len(best):
            best = result
    
    # Now try MILP to find larger sets
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # We'll try increasing cardinality k from len(best)+1 up to 22
        # For each k, check if there exists a dissociated subset of size k
        
        for k in range(len(best)+1, 23):
            if time.time() > deadline - 5:
                break
            
            # Variables: x[0..21] binary (include element i+1 or not)
            # Plus auxiliary variables y[0..2^k - 2] for subset sum ordering constraints
            # 
            # Actually, the constraint is: for all distinct subsets A, B of the chosen elements,
            # sum_{i in A} c[i] != sum_{i in B} c[i]
            #
            # This is hard to formulate directly. Let me think differently.
            #
            # Alternative: enumerate all 2^k subsets, and for each pair of distinct subsets,
            # add a constraint that their sums differ. But we don't know which k elements are chosen.
            #
            # Better approach: For a FIXED set S of size k, the constraint is that all 2^k subset sums are distinct.
            # The subset sums are: sum_{i in T} c[i] for T subset of S.
            # Two subsets T1, T2 have equal sums iff sum_{i in T1\T2} c[i] = sum_{i in T2\T1} c[i].
            # So we need: for all non-empty disjoint pairs (A, B) of subsets of S, sum_{i in A} c[i] != sum_{i in B} c[i].
            #
            # This is equivalent to: the set {c[i] : i in S} is dissociated, i.e., all subset sums are distinct.
            #
            # For MILP, we can use the following approach:
            # Introduce binary variables x_i for each element.
            # The constraint that all subset sums are distinct can be enforced by:
            # For every pair of disjoint non-empty subsets (A, B) of [1..22],
            # if both A and B are subsets of the chosen set, then sum_{i in A} c[i] != sum_{i in B} c[i].
            #
            # This is still complex. Let me try a different formulation.
            #
            # Actually, let's just use the fact that for small k, we can enumerate.
            # For k up to ~12, 2^k = 4096 subset sums, and we need to check all pairs.
            #
            # Let me try a simpler approach: use the existing _is_dissociated check with a smarter search.
            
            # Let's do a branch and bound with the LCM-based integer check
            def _is_dissociated_int(S):
                """Check using integer arithmetic with precomputed coefficients."""
                if not S:
                    return True
                n = len(S)
                if n > 20:
                    return False
                # Compute all subset sums as integers
                sums = [0]
                for idx in S:
                    new_sums = [s + c[idx] for s in sums]
                    sums.extend(new_sums)
                # Check for duplicates
                if len(set(sums)) != len(sums):
                    return False
                return True
            
            # Branch and bound
            def bb_search(available, current):
                nonlocal best
                if time.time() > deadline - 2:
                    return
                if len(current) + len(available) < k:
                    return
                if len(current) == k:
                    if _is_dissociated_int(current):
                        best = current[:]
                        return True
                    return False
                if not available:
                    return False
                
                # Try including available[0]
                a = available[0]
                rest = available[1:]
                
                # Check if current + [a] can still lead to a valid solution
                if _is_dissociated_int(current + [a]):
                    if bb_search(rest, current + [a]):
                        return True
                
                # Try excluding available[0]
                if len(current) + len(rest) >= k:
                    if bb_search(rest, current):
                        return True
                
                return False
            
            # Order elements: larger numbers first (fewer conflicts typically)
            bb_order = list(range(22, -1, -1))
            if bb_search(bb_order, []):
                if len(best) >= k:
                    # Found a valid set of size k
                    pass
            else:
                # No valid set of size k found
                break
                
    except ImportError:
        pass
    except Exception:
        pass
    
    # Final verification
    if best and not _is_dissociated(best):
        # Fallback: use greedy result
        best = greedy_add(list(range(1, 23)))
    
    return best
