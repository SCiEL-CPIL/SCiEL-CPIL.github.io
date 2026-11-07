import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Candidates 1..256
    # We sort descending. Large numbers are less likely to divide each other.
    # Also, avoiding 1 is crucial as 1 divides all sums.
    candidates = list(range(256, 0, -1))
    
    best_set = []
    best_size = 0

    # To speed up, we can pre-filter candidates.
    # 1 is very bad. 2 is bad if we have any even sum.
    # Let's just run the search.

    def is_valid_extension(current_sums, x):
        """
        Check if adding x to a set with sums 'current_sums' is valid.
        Returns the new set of sums if valid, else None.
        """
        # Generate new sums
        # new_sums = {x} U {s + x for s in current_sums}
        
        # Optimization: If x is in current_sums, duplicate sum x. 
        # But x is a new element, so x is not in current_sums unless x=0 (not allowed) or we picked x before (we don't).
        # However, it is possible that s1 + x == s2 for some s1, s2 in current_sums?
        # No, current_sums are sums of a subset. If we add x, the new sums are sums of subsets containing x.
        # The old sums are sums of subsets NOT containing x.
        # So we just need to check if any new sum equals any old sum.
        
        new_sums = []
        # Add x itself
        # Check if x is in current_sums
        if x in current_sums:
            return None
        
        # Check divisibility of x with current_sums
        for s in current_sums:
            if s % x == 0 or x % s == 0:
                return None
        new_sums.append(x)
        
        # Generate s + x for all s in current_sums
        for s in current_sums:
            ns = s + x
            # Check if ns is in current_sums (duplicate sum)
            if ns in current_sums:
                return None
            # Check if ns is in new_sums (duplicate among new sums)
            # Since s are distinct and x is constant, s+x are distinct.
            # So no duplicates within new_sums from this loop.
            
            # Check divisibility with current_sums
            # ns must not divide any s, and no s must divide ns.
            # Since ns > s (x > 0), s cannot divide ns unless ns % s == 0.
            # ns cannot divide s (ns > s).
            for cs in current_sums:
                if ns % cs == 0:
                    return None
                # Also check if cs divides ns? Yes, that's ns % cs == 0.
                # What if cs is a multiple of ns? Impossible since ns > s >= cs? No, cs can be larger than s?
                # current_sums contains all subset sums. 
                # If s is a small sum, ns = s+x is larger.
                # A larger existing sum cs could divide ns? No, cs <= max(current_sums).
                # Wait, ns could be smaller than some cs?
                # Example: current_sums = {5, 10}. x = 2.
                # s=5 -> ns=7. 7 not in {5,10}. 7%5!=0, 7%10!=0.
                # s=10 -> ns=12. 12 not in {5,10}. 12%5!=0, 12%10!=0.
                # But what if current_sums has 6? 
                # If current_sums = {3, 6}. x=3.
                # s=3 -> ns=6. 6 is in current_sums. Duplicate.
                
            new_sums.append(ns)
            
        # Now check divisibility among new_sums themselves
        # new_sums are distinct.
        # We need to ensure no pair (a, b) in new_sums has a|b or b|a.
        n = len(new_sums)
        for i in range(n):
            ai = new_sums[i]
            for j in range(i + 1, n):
                aj = new_sums[j]
                if ai > aj:
                    ai, aj = aj, ai
                if aj % ai == 0:
                    return None

        return set(new_sums)

    def dfs(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 2.0:
            return

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # If we've considered all candidates, return
        if idx >= len(candidates):
            return

        # Option 1: Skip candidates[idx]
        dfs(idx + 1, current_set, current_sums)
        
        # Option 2: Include candidates[idx]
        x = candidates[idx]
        
        # Pruning: If x is too large or creates conflicts, skip
        # We can check validity before recursing
        new_sums = is_valid_extension(current_sums, x)
        if new_sums is not None:
            current_set.append(x)
            dfs(idx + 1, current_set, new_sums)
            current_set.pop()

    # Start DFS
    # To improve efficiency, we can limit the search depth or use heuristics.
    # With 256 candidates, full DFS is too slow.
    # We need a better heuristic or a limited search.
    
    # Let's try a greedy approach first to get a good initial bound, then local search?
    # Or use a limited depth DFS?
    # Max size is likely small (around 8-12).
    # Branching factor is 2. Depth 256 is impossible.
    
    # We must prune aggressively.
    # Observation: If we pick small numbers, we create many small sums, which are prone to divisibility.
    # If we pick large numbers, sums are large and sparse.
    # Let's try searching with a limit on the number of candidates considered, or use a heuristic ordering.
    
    # Better strategy: 
    # 1. Find a good set using randomized greedy with restarts (fast).
    # 2. Use local search to improve it.
    
    # Let's stick to the Branch and Bound but with a strict time limit and a smarter candidate ordering.
    # Actually, the `is_valid_extension` is O(N^2) where N is number of sums.
    # For depth 10, N ~ 1000. 1000^2 = 1e6. Too slow for many nodes.
    
    # Let's optimize `is_valid_extension` using a boolean array for sums up to a max value.
    # Max sum for 10 elements from 1..256 is roughly 10*256 = 2560.
    # We can use a list of bools.
    
    def is_valid_extension_fast(current_sums_set, current_max_sum, x):
        # current_sums_set is a set for fast lookup
        # We need to check new sums against existing and each other.
        
        # 1. Check x against existing
        if x in current_sums_set:
            return None, current_max_sum
        for s in current_sums_set:
            if s % x == 0 or x % s == 0:
                return None, current_max_sum
        
        new_sums = [x]
        max_new_sum = x
        for s in current_sums_set:
            ns = s + x
            if ns in current_sums_set:
                return None, current_max_sum
            # Check divisibility with existing
            # ns > s. So only check if ns % existing == 0.
            # We don't need to check if existing % ns == 0 because existing < ns (since x>0).
            # Wait, is existing always < ns? Yes, ns = s + x > s.
            # But there might be OTHER existing sums larger than s?
            # Yes. But if existing > ns, then existing cannot be a multiple of ns?
            # If existing = k * ns, then existing >= 2*ns > ns.
            # So we must check if any existing sum is a multiple of ns.
            # This is harder.
            
            # Let's just check all pairs for safety, but optimize the loop.
            for cs in current_sums_set:
                if cs == ns: continue # already checked duplicate
                if cs > ns:
                    if cs % ns == 0:
                        return None, current_max_sum
                else:
                    if ns % cs == 0:
                        return None, current_max_sum
            new_sums.append(ns)
            if ns > max_new_sum:
                max_new_sum = ns
                
        # Check new sums among themselves
        n = len(new_sums)
        for i in range(n):
            ai = new_sums[i]
            for j in range(i+1, n):
                aj = new_sums[j]
                if ai > aj:
                    ai, aj = aj, ai
                if aj % ai == 0:
                    return None, current_max_sum
                    
        return set(new_sums), max_new_sum

    # The DFS with 256 candidates is still too large.
    # We need to reduce the candidate pool.
    # It is known that for this problem, the optimal sets tend to use numbers in a specific range or with specific properties.
    # Let's try a randomized local search instead, which is more robust for time-limited evaluation.
    
    def greedy_randomize():
        current_set = []
        current_sums = set()
        candidates = list(range(1, 257))
        import random
        random.shuffle(candidates)
        
        for x in candidates:
            # Use fast check
            # Re-implementing check inline for speed
            new_sums = [x]
            valid = True
            if x in current_sums:
                valid = False
            else:
                for s in current_sums:
                    if s % x == 0 or x % s == 0:
                        valid = False
                        break
                    ns = s + x
                    if ns in current_sums:
                        valid = False
                        break
                    for cs in current_sums:
                        if cs == ns: continue
                        if cs > ns:
                            if cs % ns == 0:
                                valid = False
                                break
                        else:
                            if ns % cs == 0:
                                valid = False
                                break
                    if not valid: break
                    new_sums.append(ns)
                    
            if valid:
                # Check new sums among themselves
                for i in range(len(new_sums)):
                    ai = new_sums[i]
                    for j in range(i+1, len(new_sums)):
                        aj = new_sums[j]
                        if ai > aj: ai, aj = aj, ai
                        if aj % ai == 0:
                            valid = False
                            break
                    if not valid: break
                
            if valid:
                current_set.append(x)
                for ns in new_sums:
                    current_sums.add(ns)
                
        return current_set, current_sums

    best_set, _ = greedy_randomize()
    best_size = len(best_set)
    
    # Local Search: Try to swap elements
    def local_search(initial_set):
        current_set = list(initial_set)
        current_sums = set()
        for x in current_set:
            new_sums = [x]
            for s in current_sums:
                new_sums.append(s + x)
            for ns in new_sums:
                current_sums.add(ns)
                
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add any unused number
            used = set(current_set)
            unused = [x for x in range(1, 257) if x not in used]
            import random
            random.shuffle(unused)
            
            for x in unused:
                # Check if x can be added
                new_sums = [x]
                valid = True
                if x in current_sums:
                    valid = False
                else:
                    for s in current_sums:
                        if s % x == 0 or x % s == 0:
                            valid = False
                            break
                        ns = s + x
                        if ns in current_sums:
                            valid = False
                            break
                        for cs in current_sums:
                            if cs == ns: continue
                            if cs > ns:
                                if cs % ns == 0:
                                    valid = False
                                    break
                            else:
                                if ns % cs == 0:
                                    valid = False
                                    break
                        if not valid: break
                        new_sums.append(ns)
                        
                if valid:
                    for i in range(len(new_sums)):
                        ai = new_sums[i]
                        for j in range(i+1, len(new_sums)):
                            aj = new_sums[j]
                            if ai > aj: ai, aj = aj, ai
                            if aj % ai == 0:
                                valid = False
                                break
                        if not valid: break
                    
                if valid:
                    current_set.append(x)
                    for ns in new_sums:
                        current_sums.add(ns)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set.copy()
                    break # Restart loop with new set
                    
            if not improved:
                # Try swaps? Too complex. Just break.
                pass
                
        return current_set

    # Run local search on the greedy result
    result = local_search(best_set)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Run a few more random restarts if time permits
    import random
    restarts = 0
    while time.time() < deadline - 10.0:
        s, _ = greedy_randomize()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # Improve it
            result = local_search(best_set)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
        restarts += 1
        if restarts > 50: break

    # Final verification
    if best_size >= 2:
        final_sums = set()
        for x in best_set:
            new_sums = [x]
            for s in final_sums:
                new_sums.append(s + x)
            for ns in new_sums:
                final_sums.add(ns)
        
        valid = True
        # Check distinctness (handled by set, but we need to ensure no duplicates were formed during construction? 
        # The set construction ensures distinctness of the final sums. 
        # But the constraint is "no two distinct resulting values may divide one another".
        # If sums are distinct, we just check divisibility.)
        
        sums_list = list(final_sums)
        for i in range(len(sums_list)):
            si = sums_list[i]
            for j in range(i+1, len(sums_list)):
                sj = sums_list[j]
                if si > sj: si, sj = sj, si
                if sj % si == 0:
                    valid = False
                    break
            if not valid: break
        
        if not valid:
            # Fallback to a small known valid set
            best_set = [11, 13, 17]
            best_size = 3

    return [int(x) for x in best_set]

if __name__ == "__main__":
    construct()
