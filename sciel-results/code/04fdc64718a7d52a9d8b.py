import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N
    # Using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Optimization: Split loop into two parts?
    # For small i, vectorized is fast.
    # For large i, loop overhead dominates.
    # But 10^7 iterations is manageable.
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 10:
        # Fallback
        return [11, 6]

    # 2. Search for best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    max_b = int(N * 2 / 3)
    
    # Iterate b from max_b down to 1
    # This allows early termination if we find a k=1 solution
    for b in range(max_b, 0, -1):
        sb = sigma[b]
        
        # Check a values
        # 2a = 3b + 1  => a = (3b+1)/2
        # 2a = 3b - 1  => a = (3b-1)/2
        # 2a = 3b + 3  => a = (3b+3)/2
        # 2a = 3b - 3  => a = (3b-3)/2
        
        # We check these 4 candidates
        # Note: 3b+1 and 3b-1 are even if b is odd.
        # 3b+3 and 3b-3 are even if b is odd?
        # If b is odd, 3b is odd.
        # 3b+1 is even.
        # 3b-1 is even.
        # 3b+3 is even.
        # 3b-3 is even.
        # If b is even, 3b is even.
        # 3b+1 is odd. (a not integer)
        # So only odd b can have k=1 or k=3 solutions?
        # Wait. 2a is always even.
        # 3b + k must be even.
        # If b is even, 3b is even, so k must be even.
        # If b is odd, 3b is odd, so k must be odd.
        
        # So:
        # If b is odd: check k = 1, -1, 3, -3
        # If b is even: check k = 2, -2, 4, -4 (next best after 1,3)
        # Actually, min |2a-3b| for even b is 2.
        # Error = 2/(2b) = 1/b.
        # For odd b, min error is 1/(2b).
        # So odd b are always better for same magnitude of k?
        # Yes.
        
        if b % 2 == 1:
            # Check k = 1, -1
            a1 = (3 * b + 1) // 2
            if a1 <= N and a1 >= 1:
                if sigma[a1] == sb:
                    err = 1.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a1
                        best_b = b
                        # Since we iterate downwards, this is the best possible for k=1
                        # Any smaller b will have larger error for k=1.
                        # Any k=3 solution with b' < b will have error 3/(2b') > 3/(2b) > 1/(2b).
                        # So we can return immediately?
                        # Wait, what if there was a k=3 solution with b' > b that was better?
                        # We would have found it earlier and set best_error.
                        # If err < best_error, it means it's better than all previous.
                        # So it is the global best.
                        return [int(best_a), int(best_b)]
                        
            a2 = (3 * b - 1) // 2
            if a2 <= N and a2 >= 1:
                if sigma[a2] == sb:
                    err = 1.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a2
                        best_b = b
                        return [int(best_a), int(best_b)]
                        
            # Check k = 3, -3
            a3 = (3 * b + 3) // 2
            if a3 <= N and a3 >= 1:
                if sigma[a3] == sb:
                    err = 3.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a3
                        best_b = b
                        # Cannot return here, because a k=1 solution with smaller b 
                        # might be better? No, k=1 with smaller b is worse.
                        # But a k=1 solution with LARGER b was already checked.
                        # So if this k=3 is the best so far, we keep it.
                        # But we cannot stop, because a k=1 solution at b-2 might be better?
                        # No, 1/(2(b-2)) > 1/(2b) > 3/(2b) is false?
                        # 1/(2(b-2)) vs 3/(2b).
                        # b > 3(b-2) => b > 3b - 6 => 6 > 2b => b < 3.
                        # So for b >= 3, k=1 is always better than k=3 for same b.
                        # But we are looking for global min.
                        # If we found k=3 at b, we found k=1 at b' > b?
                        # If we found k=1 at b' > b, we would have returned.
                        # So no k=1 solution exists for b' > b.
                        # So we must continue to check for k=1 at smaller b?
                        # No, k=1 at smaller b is worse than k=1 at b.
                        # But we didn't find k=1 at b.
                        # So we must check k=1 at b-2, b-4...
                        # Their error is 1/(2b') < 1/(2b) < 3/(2b) ?
                        # 1/(2b') < 3/(2b) => b < 3b'.
                        # Since b' < b, this is always true.
                        # So a k=1 solution at b' < b is BETTER than a k=3 solution at b.
                        # So we CANNOT stop when we find k=3.
                        # We can only stop when we find k=1.
                        pass
            else:
                a3 = None # Just for logic clarity, not used

            a4 = (3 * b - 3) // 2
            if a4 <= N and a4 >= 1:
                if sigma[a4] == sb:
                    err = 3.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a4
                        best_b = b
        else:
            # b is even
            # k must be even. Min |2a-3b| = 2.
            # Check k = 2, -2
            a1 = (3 * b + 2) // 2
            if a1 <= N and a1 >= 1:
                if sigma[a1] == sb:
                    err = 2.0 / (2 * b) # 1/b
                    if err < best_error:
                        best_error = err
                        best_a = a1
                        best_b = b
                        # Can we stop?
                        # If we find k=2 at b, can a k=2 at b' < b be better?
                        # 1/b' > 1/b. No.
                        # Can a k=4 at b'' > b be better?
                        # 4/(2b'') = 2/b''.
                        # 2/b'' < 1/b => b < b''/2.
                        # We already checked b'' > b.
                        # If we found a better one, we would have updated best_error.
                        # If this is the best, can we stop?
                        # No, because a k=2 solution at b-2 is worse.
                        # But wait, we are iterating downwards.
                        # If we find k=2 at b, it is the best k=2 solution for b' <= b.
                        # But we might have missed a k=4 solution at b'' > b that is better?
                        # We checked it. If it was better, best_error would be lower.
                        # If this k=2 is better than best_error, it is the global best?
                        # No. A k=2 solution at b-2 is worse.
                        # But a k=1 solution at b_odd < b?
                        # Error 1/(2b_odd).
                        # 1/(2b_odd) < 1/b => b < 2b_odd.
                        # Since b_odd < b, this is possible.
                        # Example: b=100, err=0.01.
                        # b_odd=99, err=1/198=0.005.
                        # So we CANNOT stop on k=2.
                        pass

            a2 = (3 * b - 2) // 2
            if a2 <= N and a2 >= 1:
                if sigma[a2] == sb:
                    err = 1.0 / b
                    if err < best_error:
                        best_error = err
                        best_a = a2
                        best_b = b

            # Check k = 4, -4
            a3 = (3 * b + 4) // 2
            if a3 <= N and a3 >= 1:
                if sigma[a3] == sb:
                    err = 4.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a3
                        best_b = b

            a4 = (3 * b - 4) // 2
            if a4 <= N and a4 >= 1:
                if sigma[a4] == sb:
                    err = 4.0 / (2 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a4
                        best_b = b

        # Time check
        if b % 1000000 == 0 and time.time() > deadline - 5:
            break

    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    res = construct()
    print(res)
