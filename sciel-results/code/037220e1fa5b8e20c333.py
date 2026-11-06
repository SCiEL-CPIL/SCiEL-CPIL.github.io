import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # 1. Base set: [91, 180]
    S_large = set(range(91, 181))
    
    # 2. Optimize subset of [1, 45]
    # These elements do not form triples with S_large
    # because 1/x > 1/b + 1/c for x<=45 and b,c>=91 is impossible?
    # Wait, 1/45 = 0.0222. 1/91+1/92 = 0.0218.
    # So 1/x CAN be > 1/b+1/c.
    # But we need 1/x = 1/b+1/c.
    # Max sum is 0.0218. Min 1/x for x<=45 is 1/45=0.0222.
    # So 1/x is ALWAYS > 1/b+1/c for x<=45, b,c>=91.
    # Thus, NO triple of type (x, b, c) with x in [1,45], b,c in [91,180] exists.
    # Also, 1/b = 1/x + 1/c implies 1/c = 1/b - 1/x.
    # Since b>=91, x<=45, 1/b < 1/x, so 1/c < 0. Impossible.
    # So [1,45] is completely independent from [91,180] regarding triples.
    
    candidates_small = list(range(1, 46))
    S_small = set(candidates_small)
    
    # Helper to check if a set is valid (no triples)
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        # Precompute reciprocals? Or just check pairs.
        # 1/a = 1/b + 1/c
        # a = bc / (b+c)
        # For each pair b,c, check if a is in S.
        # O(N^2). N=45 is small.
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                denom = b + c
                numer = b * c
                if numer % denom == 0:
                    a = numer // denom
                    if a in S and a != b and a != c:
                        return False
        return True
    
    # Greedy initialization: remove elements involved in triples
    # Start with all, remove one from each triple
    # Better: Start empty, add if valid? No, we want max size.
    # Start with all, iteratively remove "most problematic" element.
    
    current_small = set(candidates_small)
    
    # Count conflicts for each element
    def get_conflicts(S):
        conflicts = {x: 0 for x in S}
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                denom = b + c
                numer = b * c
                if numer % denom == 0:
                    a = numer // denom
                    if a in S and a != b and a != c:
                        conflicts[a] += 1
                        conflicts[b] += 1
                        conflicts[c] += 1
        return conflicts
    
    # Local Search on S_small
    best_small = set(current_small)
    best_size = len(best_small)
    
    # Initial cleanup
    while not is_valid_set(best_small):
        confs = get_conflicts(best_small)
        # Remove element with max conflicts
        max_conf_elem = max(confs, key=lambda x: confs[x])
        best_small.remove(max_conf_elem)
        best_size = len(best_small)
        
    # Iterative improvement
    improvement = True
    while improvement and time.time() < deadline - 5:
        improvement = False
        
        # Try to add missing elements
        missing = [x for x in candidates_small if x not in best_small]
        random.shuffle(missing)
        for x in missing:
            new_set = best_small | {x}
            if is_valid_set(new_set):
                best_small = new_set
                best_size += 1
                improvement = True
                break # Restart loop? Or continue? Continue to add more if possible.
        
        # If no single add worked, try swaps
        if not improvement:
            # Try removing one and adding one
            removed = list(best_small)
            random.shuffle(removed)
            for r in removed:
                temp_set = best_small - {r}
                for x in missing: # missing is outdated, recalculate?
                    # Just check all candidates
                    pass
                # Better: just try all x in candidates_small
                for x in candidates_small:
                    if x in best_small:
                        continue
                    new_set = temp_set | {x}
                    if is_valid_set(new_set):
                        best_small = new_set
                        improvement = True
                        break
                if improvement:
                    break
            if improvement:
                continue
            
            # Try removing one and adding two
            # This is O(N^3) roughly, might be slow.
            # For N=45, 45^3 = 91125. is_valid is O(45^2) = 2025.
            # 91125 * 2025 is too slow.
            # Skip this for now or limit it.
            
    # 3. Try to add elements from [46, 90]
    # These might conflict with S_large.
    S_current = S_large | best_small
    best_total_size = len(S_current)
    
    # Check if we can add any x in [46, 90]
    # For each x, find conflicts in S_large.
    # If conflicts are empty, add x (check conflicts with best_small).
    # If conflicts exist, check if we can remove them.
    
    candidates_mid = list(range(46, 91))
    random.shuffle(candidates_mid)
    
    for x in candidates_mid:
        if time.time() > deadline - 2:
            break
        
        # Check conflicts with S_large
        # 1/x = 1/b + 1/c => (b-x)(c-x) = x^2
        conflicts_large = []
        b_range = range(91, 181)
        for b in b_range:
            if b == x: continue
            # 1/c = 1/x - 1/b = (b-x)/(bx)
            # c = bx / (b-x)
            if b > x:
                denom = b - x
                numer = b * x
                if numer % denom == 0:
                    c = numer // denom
                    if 91 <= c <= 180 and c != b and c != x:
                        conflicts_large.append((b, c))
            # Note: if b < x, then 1/b > 1/x, so 1/c < 0, impossible.
            # So only b > x matters.
            
        if not conflicts_large:
            # Check conflicts with best_small
            # 1/x = 1/s1 + 1/s2 with s1,s2 in best_small
            # 1/s1 = 1/x + 1/s2
            # Since x >= 46 and s <= 45, 1/x < 1/45 <= 1/s.
            # 1/s1 = 1/x + 1/s2 > 1/s2 => s1 < s2.
            # Also 1/s1 > 1/x => s1 < x.
            # So s1, s2 are small.
            valid_small = True
            s_list = list(best_small)
            for i in range(len(s_list)):
                s1 = s_list[i]
                for j in range(i+1, len(s_list)):
                    s2 = s_list[j]
                    # Check 1/x = 1/s1 + 1/s2
                    denom = s1 + s2
                    numer = s1 * s2
                    if numer % denom == 0:
                        a = numer // denom
                        if a == x:
                            valid_small = False
                            break
                    # Check 1/s1 = 1/x + 1/s2
                    # 1/s2 = 1/s1 - 1/x = (x-s1)/(s1*x)
                    # s2 = s1*x / (x-s1)
                    if x > s1:
                        denom2 = x - s1
                        numer2 = s1 * x
                        if numer2 % denom2 == 0:
                            s2_calc = numer2 // denom2
                            if s2_calc in best_small and s2_calc != s1 and s2_calc != x:
                                valid_small = False
                                break
                    if not valid_small:
                        break
                    # Check 1/s2 = 1/x + 1/s1 (symmetric)
                    if x > s2:
                        denom3 = x - s2
                        numer3 = s2 * x
                        if numer3 % denom3 == 0:
                            s1_calc = numer3 // denom3
                            if s1_calc in best_small and s1_calc != s2 and s1_calc != x:
                                valid_small = False
                                break
            if valid_small:
                best_small.add(x)
                best_total_size += 1
                # Note: x is now in best_small. This might affect other mid candidates?
                # We are iterating, so yes.
                
    # Final check and return
    final_set = list(S_large | best_small)
    return final_set

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
