import time

def construct():
    n = 256
    S = set()
    
    # Greedy addition from largest to smallest
    # This ensures that when we add x, all elements in S are > x.
    # Thus, x can only act as 'a' in the condition a < b <= c.
    # We just need to check if x divides any sum of two elements in S.
    
    for x in range(n, 0, -1):
        valid = True
        # Check if there exist b, c in S such that x < b <= c and x | (b+c)
        # Since all elements in S are > x, the condition x < b is always true for b in S.
        # We just need to check if any pair sum is divisible by x.
        # Optimization: iterate through S
        items = list(S)
        for i in range(len(items)):
            b = items[i]
            # b is always > x
            for j in range(i, len(items)):
                c = items[j]
                # c >= b > x
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    return list(S)
