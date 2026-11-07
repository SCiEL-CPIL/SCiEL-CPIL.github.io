import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # S[k] is a set of integers representing achievable sums with exactly k elements
    # Initially, empty set has sum 0 with cardinality 0
    S = [set()]
    S[0].add(0)
    
    result = []
    max_val = 256
    
    # Try adding numbers from largest to smallest
    # This tends to keep sum ranges for different cardinalities separated
    for v in range(max_val, 0, -1):
        if time.time() > deadline:
            break
            
        n = len(S) - 1
        valid = True
        
        # Check if adding v causes any collision between different cardinalities
        # New sums of cardinality k will be S[k] U {s+v for s in S[k-1]}
        # We need to ensure:
        # 1. {s+v for s in S[k-1]} does not intersect S[m] for any m != k
        # Note: S[k] and S[m] are already disjoint for k != m.
        # Also {s+v for s in S[k-1]} and {t+v for s in S[m-1]} are disjoint if S[k-1] and S[m-1] are.
        
        for k in range(1, n + 1):
            # Generate shifted sums for cardinality k
            # We can do this lazily or as a set
            # Since max sum is ~15000, set creation is cheap
            shifted = {s + v for s in S[k-1]}
            
            # Check against all existing S[m] where m != k
            for m in range(0, n + 1):
                if m == k:
                    continue
                # If there is any common element, it's a collision
                if shifted & S[m]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Update S
            new_S = [set(s) for s in S]
            # Add new sums
            for k in range(1, n + 2):
                if k <= n:
                    # S[k] gets new elements from S[k-1] + v
                    new_S[k].update(s + v for s in S[k-1])
                else:
                    # k = n+1, only from S[n] + v
                    new_S[k] = {s + v for s in S[n]}
            S = new_S
            result.append(v)
            
    return result
