import time

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill
    n = 5040
    target = 4979

    # Generate divisors
    divs = []
    for d in range(1, int(n**0.5)+1):
        if n % d == 0:
            divs.append(d)
            if d != n//d:
                divs.append(n//d)
    divs.sort()
    # Exclude divisors > target (can't be in a sum equal to target with positive others)
    divs = [d for d in divs if d <= target]
    m = len(divs)

    def check_time():
        return time.time() > deadline

    best = None

    # k=1
    if target in divs:
        return [target]

    # k=2
    sset = set(divs)
    for i in range(m):
        a = divs[i]
        b = target - a
        if b > a and b in sset:
            return [a, b]

    # k=3
    for i in range(m):
        a = divs[i]
        for j in range(i+1, m):
            b = divs[j]
            c = target - a - b
            if c > b and c in sset:
                return [a, b, c]
        if check_time():
            break

    # k=4: use pair sums
    # Build dict of pair sum -> list of (i,j) for i<j
    pair_sum = {}
    for i in range(m):
        a = divs[i]
        for j in range(i+1, m):
            b = divs[j]
            s = a + b
            if s > target:
                break  # divs sorted, further j larger
            if s not in pair_sum:
                pair_sum[s] = []
            pair_sum[s].append((i, j))
        if check_time():
            break

    # Now find two disjoint pairs (i,j) and (k,l) with sum to target
    for i in range(m):
        a = divs[i]
        for j in range(i+1, m):
            b = divs[j]
            s1 = a + b
            need = target - s1
            if need in pair_sum:
                for k, l in pair_sum[need]:
                    # Ensure disjoint
                    if i != k and i != l and j != k and j != l:
                        return [divs[i], divs[j], divs[k], divs[l]]
        if check_time():
            break

    # k=5: find a divisor and a k=4 solution for target-d
    # Build k=4 solutions for various targets on the fly or use a helper
    # To keep it simple and fast, we can try to find a 4-sum for target-d for each d
    # But building pair_sum for each d is expensive. 
    # Instead, we can reuse the pair_sum logic.
    # Let's try a few largest divisors first as they are more likely to leave a residual that is a sum of 4 smaller ones?
    # Actually, any d works.
    
    # Let's try to find a 5-term sum by iterating d and checking if target-d can be formed by 4 distinct divisors (excluding d).
    # This is O(m * cost_k4).
    # We can optimize by only checking if target-d is achievable.
    
    # Let's try a specific strategy: 
    # For each d in divs:
    #   rem = target - d
    #   If rem can be sum of 4 distinct divisors from divs (excluding d), return [d] + those 4.
    
    # To check "rem can be sum of 4 distinct divisors", we can use a similar pair-sum approach but filtered.
    # This might be too slow if done naively for all d.
    
    # Let's try just a few d's first, or a smarter approach.
    # Actually, since 5040 has only 60 divisors, even O(m^4) is ~12 million, which is doable in Python if optimized? No, 60^4 is 12,960,000. 
    # Inner loop operations are simple. 10M ops in Python is ~1s. 
    # So we can do a brute force k=5 search if needed, but we need to be careful with time.
    
    # Let's do a smarter k=5:
    # Iterate d1.
    # For remaining, find 4 distinct divisors summing to target-d1.
    # We can use the pair_sum approach for the 4-sum part, but we need to exclude d1.
    
    # Let's try to find a 5-sum by checking if there exists d such that target-d is a 4-sum of divisors not including d.
    
    # We can pre-compute all 4-sums? No, too many.
    
    # Let's try iterating d and doing a quick check.
    # Since m is small (60), we can try to find a 4-sum for target-d using the pair_sum dictionary built for the full set, 
    # but we must ensure the 4 divisors don't include d.
    
    # Let's build pair_sum again (or reuse) and for each d, check if there are two disjoint pairs summing to target-d, 
    # none of which contain d.
    
    # This is O(m * (number of pairs for target-d)).
    
    # Let's implement this.
    
    # Reuse pair_sum from k=4 phase? Yes, it's already built.
    # But pair_sum contains pairs that might include d.
    
    for d_idx in range(m):
        d = divs[d_idx]
        rem = target - d
        if rem <= 0:
            continue
        if rem in pair_sum:
            found = False
            for k, l in pair_sum[rem]:
                # Check if k or l is d_idx
                if k == d_idx or l == d_idx:
                    continue
                # Need another pair summing to rem? No, pair_sum[rem] gives pairs summing to rem.
                # We need TWO pairs summing to rem? No, we need 4 numbers summing to rem.
                # So we need two pairs (k,l) and (p,q) such that (k,l) sums to s1 and (p,q) sums to rem-s1?
                # No, the pair_sum dictionary maps sum -> pairs.
                # So for 4-sum, we need s1 + s2 = rem.
                # We iterate s1, find pair1 summing to s1, pair2 summing to rem-s1.
                
                # This logic is getting complicated to reuse the single target pair_sum.
                # Let's just do a dedicated k=5 search using nested loops for the last 2, and pair-sum for the first 3?
                # Or just brute force 5 loops? 60^5 is 777M, too slow.
                
                # Let's stick to the pair-sum method for 4-sum part.
                # For a fixed d, we want to find 4 distinct indices i,j,k,l != d_idx such that divs[i]+divs[j]+divs[k]+divs[l] = rem.
                # We can iterate i,j (i<j, i,j != d_idx), s1 = divs[i]+divs[j], need = rem - s1.
                # Check if need can be formed by 2 distinct divs k,l != d_idx, i, j.
                
                # This is O(m^2) per d, so O(m^3) total. 60^3 = 216,000. Very fast.
                
                for i in range(m):
                    if i == d_idx: continue
                    for j in range(i+1, m):
                        if j == d_idx: continue
                        s1 = divs[i] + divs[j]
                        if s1 >= rem: break # divs sorted
                        need = rem - s1
                        # Find k,l such that divs[k]+divs[l]=need, k,l distinct, k,l not in {d_idx, i, j}
                        # We can use a set for remaining divs? Or just loop k,l.
                        # Loop k,l is O(m^2). Total O(m^4) = 12M. Might be tight but okay.
                        # Let's optimize by using a set of available divs for the last pair.
                        # But we need distinct indices.
                        
                        # Let's just do a simple loop for k,l for now, as m is small.
                        for k in range(m):
                            if k in (d_idx, i, j): continue
                            for l in range(k+1, m):
                                if l in (d_idx, i, j): continue
                                if divs[k] + divs[l] == need:
                                    return [divs[d_idx], divs[i], divs[j], divs[k], divs[l]]
        if check_time():
            break

    # Fallback: greedy (seed-like) to guarantee validity
    remainder = target
    chosen = []
    for d in range(n, 0, -1):
        if n % d == 0 and d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    if remainder == 0:
        return chosen
    # Should not reach here
    return chosen
