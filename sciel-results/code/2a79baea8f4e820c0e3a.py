import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Greedy construction from largest to smallest
    S = set()
    
    def is_valid_with_new(x):
        # Check if S U {x} is valid
        # x is the new element
        # We need to check all triples (a, b, c) in S U {x}
        # Since S is already valid, we only need to check triples involving x
        # Cases:
        # 1. x is the smallest: a=x, b,c in S, x < b <= c
        #    Check x | (b+c)
        # 2. x is in the middle: a in S, b=x, c in S, a < x <= c
        #    Check a | (x+c)
        # 3. x is the largest: a,b in S, c=x, a < b <= x
        #    Check a | (b+x)
        
        # Optimization: 
        # If x is large, cases 2 and 3 involve a,b < x.
        # If x is small, case 1 involves b,c > x.
        
        # Iterate through pairs in S
        # This is O(|S|^2) per insertion.
        # Total O(N^3) worst case. 256^3 is ~16 million. 
        # Python might be slow for 16M ops in 85s? 
        # 16M simple ops is usually < 1s in C, maybe 5-10s in Python.
        # We can optimize with early termination.
        
        S_list = list(S)
        
        # Case 1: x is smallest divisor
        # a = x
        # b, c in S such that x < b <= c
        # x | (b+c)
        # Since x is added in decreasing order, S contains numbers > x.
        # So all b, c in S satisfy b > x and c > x.
        # We just need to check if any pair sums to a multiple of x.
        # Sum range: [2*min(S), 2*max(S)]
        # Multiples of x in that range.
        
        # We can just iterate pairs.
        # To speed up, we can use a set of sums? No, need to check divisibility.
        
        # Let's just iterate.
        # Since S is a tail, it's dense.
        
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            # b > x
            # c >= b
            # Check x | (b+c)
            # c can be any element in S >= b
            # We can precompute sums?
            # Or just loop c.
            for j in range(i, len(S_list)):
                c = S_list[j]
                if c < b:
                    continue
                # b <= c
                # Check x | (b+c)
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is middle summand
        # a in S, b=x, c in S
        # a < x <= c
        # a | (x+c)
        # a must be < x. Since we add in decreasing order, S only has elements > x.
        # So there are NO elements a in S with a < x.
        # Thus Case 2 is empty.
        
        # Case 3: x is largest summand
        # a, b in S, c=x
        # a < b <= x
        # a | (b+x)
        # Similarly, a, b must be < x.
        # S only has elements > x.
        # So there are NO elements a, b in S with a, b < x.
        # Thus Case 3 is empty.
        
        # Wait, this logic holds ONLY if we add in strictly decreasing order.
        # If we add 256, then 255, S={256, 255}.
        # Add 254. S={256, 255, 254}.
        # When adding x, S contains only elements > x.
        # So a, b, c in S are all > x.
        # Triples involving x:
        # 1. x is smallest: a=x, b,c in S. (Valid check needed)
        # 2. x is middle: a in S, b=x, c in S. Requires a < x. But a in S implies a > x. Contradiction.
        # 3. x is largest: a,b in S, c=x. Requires b <= x. But b in S implies b > x. Contradiction.
        
        # So we ONLY need to check Case 1!
        # This simplifies the check to:
        # Does there exist b, c in S such that x < b <= c and x | (b+c)?
        
        return True

    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_with_new(x):
            S.add(x)
            
    return list(S)
