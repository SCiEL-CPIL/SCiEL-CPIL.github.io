import time

def construct():
    deadline = time.time() + 50  # Safety margin
    
    def is_valid_addition(S_set, x):
        """
        Check if adding x to S_set keeps the set valid.
        We only need to check triples that include x, as triples not including x 
        were already valid in S_set.
        """
        # Triples involving x must be of the form {a, b, x} with a, b in S_set, a != b
        # Sums: a+b, a+x, b+x
        # All three sums must be in S_set U {x}
        
        for a in S_set:
            for b in S_set:
                if a == b:
                    continue
                # Check if {a, b, x} forms a forbidden triple
                s1 = a + b
                s2 = a + x
                s3 = b + x
                
                # The sums must be in the new set S_set U {x}
                # Note: s1, s2, s3 must be distinct? No, the problem says "six displayed values need not be mutually distinct".
                # But a,b,x are distinct.
                # The condition is: a+b in S', a+c in S', b+c in S'.
                
                in_s1 = (s1 in S_set) or (s1 == x)
                in_s2 = (s2 in S_set) or (s2 == x)
                in_s3 = (s3 in S_set) or (s3 == x)
                
                if in_s1 and in_s2 and in_s3:
                    return False
        return True

    # Greedy construction: Add largest elements first
    S = set()
    for x in range(128, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(S, x):
            S.add(x)
    
    # Local Search: Try to improve the set size
    # We can try to swap an element in S with an element not in S if it allows adding more elements?
    # Or simply try to add any excluded element that might have been blocked by a specific element in S.
    
    # A more effective local search: 
    # 1. Identify candidates not in S.
    # 2. For each candidate x, find which elements in S block it.
    # 3. If removing one blocking element y allows adding x AND we can add another element z, we gain size.
    
    # Simpler local search: Randomized restarts of greedy with different orders?
    # Or just try to add elements in ascending order to the existing set?
    # If we have S_greedy, can we add small elements that were skipped?
    # They were skipped because they conflicted with larger elements.
    # If we remove a large element, maybe we can add two small ones?
    
    # Let's try a limited local search:
    # Try to remove one element y from S and add two elements x1, x2 not in S.
    # This is expensive.
    
    # Alternative: Try to add any element from 1..128 not in S, checking validity.
    # If we can add one, we do it. Repeat until no more can be added.
    # This handles cases where the greedy order missed a combination.
    
    improved = True
    while improved:
        improved = False
        for x in range(1, 129):
            if x in S:
                continue
            if is_valid_addition(S, x):
                S.add(x)
                improved = True
                break # Restart loop to check other elements with new S
    
    # Ensure we return a list of Python ints
    return list(S)

# Test run
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    S = set(res)
    valid = True
    triples = 0
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if a+b in S and a+c in S and b+c in S:
                    valid = False
                    triples += 1
    print(f"Valid: {valid}, Forbidden Triples: {triples}")
