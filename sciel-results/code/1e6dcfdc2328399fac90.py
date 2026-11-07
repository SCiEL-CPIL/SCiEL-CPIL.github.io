import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for overhead
    
    N = 10000000
    
    # Compute sigma(n) for n in [1, N]
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        # Fallback if sigma computation takes too long
        # Return a known valid pair if possible, or just a default
        # 11 and 6: sigma(11)=12, sigma(6)=12. Ratio 11/6 = 1.833. 3/2 = 1.5.
        # Error = |1.833 - 1.5| = 0.333
        return [11, 6]
        
    best_a = 11
    best_b = 6
    best_error = float('inf')
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let 2a - 3b = d. We want small |d| and large b.
    # d=1 => 2a = 3b+1 => b even, b=2m, a=3m+1.
    # d=-1 => 2a = 3b-1 => b even, b=2m, a=3m-1.
    # d=2 => 2a = 3b+2 => b even, b=2m, a=3m+1. Wait.
    # If d=2, 2a = 3(2m)+2 = 6m+2 => a = 3m+1.
    # So for b=2m:
    # d=1 => a=3m+1 (requires 3b+1 even => b even)
    # d=-1 => a=3m-1 (requires 3b-1 even => b even)
    # d=2 => a=3m+1 (requires 3b+2 even => b even)
    # d=-2 => a=3m-1 (requires 3b-2 even => b even)
    
    # Basically, for a given b=2m, the closest integers to 1.5b are 3m and 3m+1 (if 1.5b is half-integer) or 3m-1, 3m (if integer).
    # 1.5 * (2m) = 3m. This is an integer.
    # So a=3m gives error 0. (Invalid)
    # Next closest are a=3m+1 and a=3m-1.
    # Error for a=3m+1: |(3m+1)/(2m) - 3/2| = |3/2 + 1/(2m) - 3/2| = 1/(2m).
    # Error for a=3m-1: |(3m-1)/(2m) - 3/2| = |3/2 - 1/(2m) - 3/2| = 1/(2m).
    # So we just need to check if sigma(3m+1) == sigma(2m) or sigma(3m-1) == sigma(2m).
    
    max_m = N // 3  # Since a = 3m +/- 1 <= N, 3m+1 <= N => m <= (N-1)/3
    # N=10^7, max_m = 3333333
    
    # Search from largest m down to 1
    # We check d=1 and d=-1 cases first as they yield the minimal error for a given m.
    # If we find a match for any m, the error is 1/(2m).
    # Since m is decreasing, the first match gives the global minimum error among d=1, -1 cases.
    # We should also consider if a larger m with d=2 is better than a smaller m with d=1?
    # Error d=2 is 2/(2m) = 1/m. Error d=1 is 1/(2m).
    # 1/m < 1/(2m) is false. 1/m > 1/(2m).
    # So d=1 is always better than d=2 for the same m.
    # But could a smaller m with d=1 be worse than a larger m with d=2?
    # Error(m, d=1) = 1/(2m).
    # Error(m', d=2) = 2/(2m') = 1/m'.
    # We want to minimize error.
    # If we find a solution with d=1 at m1, error is 1/(2m1).
    # Is there a solution with d=2 at m2 > m1 such that 1/m2 < 1/(2m1)?
    # 1/m2 < 1/(2m1) => 2m1 < m2.
    # So if there is a d=2 solution with m2 > 2*m1, it would be better.
    # However, we are searching for the *best* possible.
    # The absolute best error is determined by the maximum possible b.
    # The smallest possible non-zero error is 1/(2 * max_b).
    # If we find a d=1 solution at max_m, that's the best possible.
    # If not, we go down.
    
    # Strategy:
    # Iterate m from max_m down to 1.
    # Check if sigma(3m+1) == sigma(2m). If yes, return [3m+1, 2m]. Error 1/(2m).
    # Check if sigma(3m-1) == sigma(2m). If yes, return [3m-1, 2m]. Error 1/(2m).
    # If we don't find d=1, do we need to check d=2?
    # If we find d=1 at m, error is 1/(2m).
    # Any d=2 solution at m' < m has error 1/m' > 1/m > 1/(2m). So worse.
    # Any d=2 solution at m' > m has error 1/m'.
    # If 1/m' < 1/(2m), i.e., m' > 2m, it's better.
    # But we are scanning down from max_m.
    # If we didn't find d=1 for m in [max_m, m_current+1], we haven't checked d=2 for them either.
    # However, d=1 is "closer" to 3/2.
    # It is highly probable that a d=1 solution exists near the top.
    # If we find a d=1 solution at m*, the error is 1/(2m*).
    # To beat this, we need a solution with error < 1/(2m*).
    # This requires |2a-3b|/2b < 1/(2m*) => |2a-3b|/b < 1/m*.
    # If |2a-3b|=1, then 1/b < 1/m* => b > m*.
    # Since b=2m, this means 2m > m* => m > m*/2.
    # We already checked all m >= m* for d=1.
    # Did we check d=1 for m in (m*/2, m*)? Yes, we scanned down from max_m.
    # So if we found the first d=1 solution at m*, there are no d=1 solutions with m > m*.
    # Are there d=1 solutions with m in (m*/2, m*)? No, we would have found them first.
    # So the best d=1 solution is at m*.
    # Could a d=2 solution be better?
    # Error = 1/m'. We need 1/m' < 1/(2m*) => m' > 2m*.
    # We checked m' in [max_m, m*] for d=1.
    # We did NOT check m' in [max_m, m*] for d=2.
    # If there is a d=2 solution at m' where m* < m' <= max_m, its error is 1/m'.
    # Is 1/m' < 1/(2m*)?
    # This requires m' > 2m*.
    # So if there is a d=2 solution at m' > 2m*, it is better than the d=1 solution at m*.
    # This means we cannot just stop at the first d=1 solution.
    
    # Revised Strategy:
    # We want to minimize E = |2a-3b| / (2b).
    # Let b=2m. E = |2a-6m| / (4m) = 2|a-3m| / (4m) = |a-3m| / (2m).
    # Let k = |a-3m|. E = k / (2m).
    # We want to minimize k/m.
    # For each m, we can find the minimum k such that sigma(3m+k) == sigma(2m) or sigma(3m-k) == sigma(2m).
    # Note a must be positive and <= N.
    # k=0 is invalid (error 0).
    # We check k=1, 2, ...
    # We iterate m from max_m down to 1.
    # For each m, we find min k.
    # We keep track of best_error = min(best_error, k/(2m)).
    # We can prune: if 1/(2m) >= best_error, then even k=1 for this m and all smaller m will have error >= 1/(2m) >= best_error.
    # So we can break.
    
    sigma_list = sigma.tolist() # Faster access? Maybe. Or use numpy array directly.
    # Numpy array access is fast enough.
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We need to check k=1, 2, ... up to some limit?
    # If k is large, error is large.
    # We break if 1/(2m) >= best_error.
    
    for m in range(max_m, 0, -1):
        if 1.0 / (2.0 * m) >= best_error:
            break
            
        b = 2 * m
        sb = sigma[b]
        
        # Check k=1
        # a = 3m + 1 or 3m - 1
        found = False
        # Check 3m+1
        a1 = 3 * m + 1
        if a1 <= N and sigma[a1] == sb:
            # Error is 1/(2m)
            err = 1.0 / (2.0 * m)
            if err < best_error:
                best_error = err
                best_a = a1
                best_b = b
            found = True
            # Since this is the min possible error for this m, and we are going down m,
            # we can't find a better error for this m.
            # But we continue to next m?
            # The pruning condition 1/(2m) >= best_error will handle stopping.
            # Since we found a solution with error 1/(2m), best_error <= 1/(2m).
            # For next m' = m-1, min error is 1/(2(m-1)) > 1/(2m) >= best_error.
            # So the loop will break immediately after this m if we found a k=1 solution.
            # Wait. 1/(2(m-1)) > 1/(2m).
            # If best_error == 1/(2m), then 1/(2(m-1)) > best_error.
            # So the condition 1/(2m') >= best_error will be true for m' = m-1.
            # So the loop terminates.
            # This means the first m with a k=1 solution gives the global optimum.
            break
        
        # Check 3m-1
        a2 = 3 * m - 1
        if a2 >= 1 and a2 <= N and sigma[a2] == sb:
            err = 1.0 / (2.0 * m)
            if err < best_error:
                best_error = err
                best_a = a2
                best_b = b
            found = True
            break
            
        # If no k=1 found, check k=2?
        # If we don't find k=1, we might find k=2.
        # Error would be 2/(2m) = 1/m.
        # This is larger than 1/(2m).
        # But if no k=1 exists for any m >= current, we need k=2.
        # However, the break condition is based on min possible error 1/(2m).
        # If we don't find k=1, we should check k=2, 3...
        # But if we find a k=1 solution at some m, we break.
        # If we don't find k=1 for large m, we go down.
        # Eventually we might find a k=1 solution.
        # If we never find k=1, we need to handle k>1.
        
        # Let's check k=2 just in case, but it's unlikely to be the global min if k=1 exists anywhere.
        # Actually, if k=1 exists at m1, and k=2 exists at m2 > m1,
        # Error(m1, 1) = 1/(2m1). Error(m2, 2) = 1/m2.
        # If m2 > 2m1, then 1/m2 < 1/(2m1). So k=2 at larger m is better.
        # So we CANNOT just look for k=1.
        
        # We must check all k such that k/(2m) < best_error.
        # i.e., k < 2m * best_error.
        
        max_k = int(2 * m * best_error)
        if max_k < 1:
             # No k can improve best_error for this m
             # But we still need to check if we can update best_error?
             # No, if min k=1 gives error >= best_error, no k will.
             continue
             
        # Check k from 1 to max_k
        for k in range(1, max_k + 1):
            a_plus = 3 * m + k
            a_minus = 3 * m - k
            
            if a_plus <= N and sigma[a_plus] == sb:
                err = k / (2.0 * m)
                if err < best_error:
                    best_error = err
                    best_a = a_plus
                    best_b = b
                break # Found best k for this m
            
            if a_minus >= 1 and a_minus <= N and sigma[a_minus] == sb:
                err = k / (2.0 * m)
                if err < best_error:
                    best_error = err
                    best_a = a_minus
                    best_b = b
                break # Found best k for this m
                
        # Time check
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    res = construct()
    print(res)
