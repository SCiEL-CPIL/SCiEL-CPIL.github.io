"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 256] such that for every a < b <= c in the set,
    a does not divide b+c. The goal is to maximize the cardinality of the set.
    
    The strategy is to use a greedy approach starting from the largest numbers down to 1.
    Larger numbers are 'safer' because the condition a | (b+c) is harder to satisfy when a is large
    relative to the possible sums b+c. Specifically, if 3a > 2*max(S), then no multiple of a
    can be formed by b+c for b,c in S with b,c > a.
    
    We start with all numbers 256 down to 1, and greedily remove numbers that cause conflicts.
    However, a more effective heuristic is to start with the known safe block {171..256} and then
    try to add smaller numbers if they don't conflict with the existing set.
    
    Actually, the most robust approach for this specific constraint is to iterate from n down to 1,
    maintaining a set S. For each candidate x, check if adding x violates the condition with any
    existing pairs (b, c) in S. If it doesn't, add x.
    
    Let's refine the greedy strategy:
    Start with S = empty.
    Iterate x from 256 down to 1:
        Check if x can be added to S.
        For x to be added, for all b, c in S such that x < b <= c, x must NOT divide b+c.
        Also, for all b, c in S such that b < x <= c, x must NOT divide b+c.
        Also, for all b, c in S such that b <= c < x, x must NOT divide b+c (but here x is larger, 
        so the condition is about smaller a dividing larger sums. If x is added as a new 'a', we check 
        x | (b+c) for b,c in S with x < b <= c. Since we are iterating down, S contains numbers > x.
        So we only need to check if x divides b+c for any b,c in S with b > x and c >= b).
        
        Wait, the condition is: for every a < b <= c in the set, a NOT divide b+c.
        If we add x to S, x can play the role of 'a'. The other elements in S are all > x (since we go down).
        So we need to check: for all b, c in S such that x < b <= c, does x divide b+c?
        If yes, x cannot be added.
        If no, x can be added.
        
        Note: x cannot play the role of 'b' or 'c' in a violation where 'a' is smaller, because 
        the smaller 'a's are not in S yet (we are going down). When we later add a smaller a, 
        we will check if a divides b+c where b,c include x. So the greedy check only needs to 
        verify the condition where the new element is the divisor 'a'.
    """
    n = 256
    S = []  # List of elements in the set, maintained in descending order for easy access? 
            # Actually, just a list is fine. We need to check pairs.
    
    # To speed up checks, we can store S as a set for membership, but we need to iterate pairs.
    # Since n=256, the max size is 256. Pairs are at most 256^2/2 ~ 32k. 
    # For each candidate x (256 candidates), we check up to 32k pairs.
    # Total operations ~ 256 * 32k = 8M. This is well within 60s in Python.
    
    # Let's use a list for S.
    # We iterate x from n down to 1.
    for x in range(n, 0, -1):
        can_add = True
        # Check all pairs (b, c) in S such that x < b <= c
        # Since S contains numbers > x, all elements in S are > x.
        # So we just need to check all pairs (b, c) in S with b <= c.
        # Wait, the condition is a < b <= c. Here a=x.
        # So we need b > x and c >= b.
        # Since all elements in S are > x (because we've only added larger numbers),
        # any pair (b, c) from S with b <= c satisfies b > x and c >= b.
        
        # Optimization: If len(S) is small, it's fast. As S grows, it gets slower.
        # But S will likely stay around 80-100 elements.
        
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                # We need b <= c. Since we iterate j from i, S[j] >= S[i] is not guaranteed 
                # if S is not sorted. Let's keep S sorted or just ensure b<=c in check.
                # Actually, my loop `for j in range(i, len(S))` assumes S is sorted ascending?
                # No, I'm adding x which is smaller than existing elements.
                # So S is in descending order of addition, meaning S[0] is largest.
                # Let's just check all pairs (u, v) from S with u <= v.
                
                # To be safe and simple:
                # Get all elements in S.
                # For each pair (b, c) in S such that b <= c:
                #   if (b + c) % x == 0:
                #       can_add = False
                #       break
                # break outer loop if not can_add.
                
                # Let's restructure for clarity and correctness.
                pass
        
        # Let's implement the check properly.
        # S is a list of integers.
        # We need to check if there exist b, c in S such that x < b <= c and (b+c) % x == 0.
        # Since all elements in S are > x, the condition x < b is always true for any b in S.
        # So we just need to check if there exist b, c in S with b <= c such that (b+c) % x == 0.
        
        # Optimization: 
        # If we find one such pair, we skip x.
        
        # Let's store S as a set for fast lookup? No, we need pairs.
        # List iteration is fine.
        
        # Re-do the loop logic correctly.
        valid = True
        # To avoid O(N^2) with high constant, we can precompute sums? 
        # No, just nested loops.
        for i in range(len(S)):
            for j in range(i, len(S)):
                b = S[i]
                c = S[j]
                if b == c:
                    s = b + c
                else:
                    s = b + c
                # We need b <= c.
                # If S is not sorted, we should ensure b <= c.
                # Let's keep S sorted in ascending order to make b <= c easy?
                # Or just check if min(b,c) <= max(b,c) which is always true, 
                # but the condition is a < b <= c.
                # So we need to assign the smaller of the two to b and larger to c.
                # If S[i] > S[j], then b=S[j], c=S[i].
                # If S[i] <= S[j], then b=S[i], c=S[j].
                
                # Actually, it's easier to just iterate all unique pairs {u,v} from S.
                # For each pair, let b = min(u,v), c = max(u,v).
                # Check if (b+c) % x == 0.
                
                # Since we are iterating i, j with i <= j, and S is a list,
                # S[i] and S[j] are just elements.
                # Let's just compute the sum and check divisibility.
                # The condition b <= c is satisfied by definition of the pair in the problem statement.
                # The problem says "for every a < b <= c". It implies b and c are specific values from the set.
                # So if the set contains {10, 20}, the pairs (b,c) with b<=c are (10,10), (10,20), (20,20).
                # So we need to check all combinations with replacement where b <= c.
                
                # My loop `for i in range(len(S)): for j in range(i, len(S))` covers each unordered pair {u,v} once.
                # If u != v, it covers (u,v) and (v,u) as the same pair? No, it covers the set {u,v} once.
                # But we need to check if EITHER ordering (which is just b=min, c=max) causes a violation.
                # Since divisibility of sum is symmetric, (u+v) % x == 0 is the same for (u,v) and (v,u).
                # And b <= c is just a canonical ordering.
                # So checking (S[i] + S[j]) % x == 0 is sufficient.
                
                if (S[i] + S[j]) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    # S contains the selected numbers.
    # The problem asks for a nonempty subset.
    # We should return it as a list.
    # Order doesn't matter for the set, but let's return sorted ascending for cleanliness.
    return sorted(S)

# EVOLVE-BLOCK-END
