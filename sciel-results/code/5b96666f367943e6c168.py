import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 192
    
    # Initial set: All numbers > N/2
    # These are "safer" because their multiples are sparser in the sum range.
    initial_S = set(range(N // 2 + 1, N + 1))
    
    # Function to find all violations in a set S
    def find_violations(S):
        violations = []
        S_list = list(S)
        n = len(S_list)
        for idx_a, a in enumerate(S_list):
            # T is S without a
            T = [x for x in S_list if x != a]
            if len(T) < 3:
                continue
            
            # Map residues to elements in T
            # We need to check if any 3 distinct elements in T sum to 0 mod a
            # Pair sums approach:
            # For each pair (u, v) in T, check if there is w in T \ {u, v} 
            # such that (u+v+w) % a == 0
            # w % a == -(u+v) % a
            
            # Precompute residues of T
            # Since a <= 192, we can use a list of lists for residues
            res_map = {}
            for val in T:
                r = val % a
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(val)
            
            # Iterate pairs in T
            # Optimization: We can just check existence.
            # To avoid O(N^3), we can iterate pairs and check map.
            # O(N^2) per a.
            for i in range(len(T)):
                u = T[i]
                for j in range(i+1, len(T)):
                    v = T[j]
                    s2 = u + v
                    target_r = (-s2) % a
                    if target_r in res_map:
                        candidates = res_map[target_r]
                        # Check if there is a w in candidates distinct from u and v
                        for w in candidates:
                            if w != u and w != v:
                                violations.append((a, u, v, w))
                                break # One violation per pair is enough to mark a as problematic
                        # We can break early if we just want to know if valid, 
                        # but we want to collect stats for removal heuristic
        return violations

    def is_valid(S):
        return len(find_violations(S)) == 0

    S = set(initial_S)
    removed = set()
    
    # Repair Phase
    # We iteratively remove elements causing violations
    # To make it deterministic and efficient, we pick the element involved in the most violations
    # If tie, pick the smallest (or largest? Smallest numbers divide more sums, so they are more dangerous).
    
    max_repair_steps = 100
    for step in range(max_repair_steps):
        if time.time() > deadline:
            break
            
        violations = find_violations(S)
        if not violations:
            break
            
        # Count frequency of each element in violations
        freq = {}
        for v in violations:
            for elem in v:
                freq[elem] = freq.get(elem, 0) + 1
        
        # Pick element with max frequency
        # If tie, pick smallest value (more likely to cause future issues)
        worst_elem = min(freq.keys(), key=lambda x: (-freq[x], x))
        
        S.remove(worst_elem)
        removed.add(worst_elem)
        
    # Backfill Phase
    # Try to add back removed elements
    # Sort removed elements. Maybe try larger ones first? Or random?
    # Larger numbers are safer.
    removed_list = sorted(list(removed), reverse=True)
    
    # We can do multiple passes of backfilling
    backfill_passes = 5
    for p in range(backfill_passes):
        if time.time() > deadline:
            break
        for x in removed_list:
            if x in S:
                continue
            # Check if S U {x} is valid
            # Fast check:
            # 1. x does not divide sum of any 3 in S
            # 2. No a in S divides sum of any 3 in S U {x} involving x
            
            valid = True
            
            # Check 1: x divides sum of 3 in S?
            if len(S) >= 3:
                # Use similar pair-sum logic for x
                T = list(S)
                res_map = {}
                for val in T:
                    r = val % x
                    if r not in res_map:
                        res_map[r] = []
                    res_map[r].append(val)
                
                for i in range(len(T)):
                    u = T[i]
                    for j in range(i+1, len(T)):
                        v = T[j]
                        s2 = u + v
                        target_r = (-s2) % x
                        if target_r in res_map:
                            for w in res_map[target_r]:
                                if w != u and w != v:
                                    valid = False
                                    break
                    if not valid:
                        break
            
            if valid:
                # Check 2: No a in S divides sum of 3 involving x
                # Sums are x + u + v for u, v in S distinct.
                # We need (x + u + v) % a != 0 for all a in S, u, v in S distinct.
                if len(S) >= 2:
                    for a in S:
                        # Check if exists u, v in S distinct such that (x + u + v) % a == 0
                        # (u + v) % a == (-x) % a
                        target_r = (-x) % a
                        T = list(S)
                        # We can use a set of pair sums mod a?
                        # Or just iterate pairs. O(N^2).
                        # Since we do this for each a, total O(N^3).
                        # N is ~50. 125k ops. OK.
                        
                        # Optimization: Precompute pair sums mod a?
                        # Let's just iterate.
                        found = False
                        for i in range(len(T)):
                            u = T[i]
                            for j in range(i+1, len(T)):
                                v = T[j]
                                if (x + u + v) % a == 0:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            valid = False
                            break
                            
            if valid:
                S.add(x)
                removed.remove(x)
                # Restart backfill loop? Or just continue?
                # Continuing is faster, but might miss opportunities.
                # Let's break and restart to ensure we try in sorted order again?
                # No, just continue. The set S changes, so previous "invalid" might become valid?
                # Adding an element can only create MORE violations, not remove them.
                # So if x was invalid before, it will remain invalid if we add MORE elements to S.
                # Wait, we are ADDING to S.
                # If S U {x} was invalid, adding more to S will keep it invalid.
                # So we don't need to re-check elements that failed, UNLESS we removed something else?
                # We are only adding. So if x failed, it will fail again later.
                # So we can skip it for the rest of this pass.
                # But if we do multiple passes, and we don't remove anything, it's the same.
                # However, if we do a repair phase again, we might remove elements, making room for x.
                
    # Final check and return
    # Ensure no violations
    while find_violations(S) and time.time() < deadline:
        violations = find_violations(S)
        freq = {}
        for v in violations:
            for elem in v:
                freq[elem] = freq.get(elem, 0) + 1
        worst_elem = min(freq.keys(), key=lambda x: (-freq[x], x))
        S.remove(worst_elem)
        removed.add(worst_elem)
        
    return list(S)
