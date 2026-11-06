import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    K = len(primes)
    
    if K < 2:
        return primes

    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index i' such that the subsequence ending with primes[i'], primes[i] was extended to primes[j]
    
    # We only need i < j.
    # Initialize dp[i][j] = 2 for all i < j.
    # We can use a list of lists.
    dp = [[0] * K for _ in range(K)]
    prev = [[-1] * K for _ in range(K)]
    
    for j in range(K):
        for i in range(j):
            dp[i][j] = 2
            
    max_len = 2
    best_end_j = 0
    best_end_k = 1

    # j is the second-to-last prime index
    # k is the last prime index
    # i is the third-to-last prime index
    
    for j in range(1, K):
        if time.time() > deadline:
            break
            
        # Build suffix max and argmax for dp[0..j-1][j]
        # M[i] = max(dp[t][j] for t in [i, j-1])
        # ArgM[i] = t that achieves M[i]
        
        # We only need M and ArgM for i in 0..j-1
        # We can store them in temporary lists
        M = [0] * j
        ArgM = [0] * j
        
        M[j-1] = dp[j-1][j]
        ArgM[j-1] = j-1
        
        for i in range(j-2, -1, -1):
            if dp[i][j] > M[i+1]:
                M[i] = dp[i][j]
                ArgM[i] = i
            else:
                M[i] = M[i+1]
                ArgM[i] = ArgM[i+1]
        
        # i_ptr is the smallest index i such that primes[i] >= 2*primes[j] - primes[k]
        # As k increases, L decreases, so i_ptr can only decrease or stay same.
        # We start i_ptr at j-1.
        i_ptr = j - 1
        
        for k in range(j+1, K):
            L = 2 * primes[j] - primes[k]
            
            # Adjust i_ptr
            # We want the smallest i such that primes[i] >= L.
            # Currently i_ptr is a candidate.
            # If primes[i_ptr] < L, then no i <= i_ptr works?
            # No, primes is increasing. If primes[i_ptr] < L, then primes[i] < L for all i <= i_ptr.
            # But we need primes[i] >= L.
            # So if primes[i_ptr] < L, we might need a LARGER i?
            # Wait.
            # L decreases as k increases.
            # Initially k=j+1, L is max.
            # We need primes[i] >= L_max.
            # If primes[j-1] < L_max, then no i works.
            # If primes[j-1] >= L_max, then i_ptr = j-1 is a valid start.
            # As k increases, L decreases.
            # The condition primes[i] >= L becomes easier.
            # So the set of valid i expands to the LEFT (smaller indices).
            # So i_min decreases.
            
            # So we start with i_ptr = j-1.
            # For the first k (j+1), we check if primes[j-1] >= L.
            # If yes, i_min could be smaller.
            # We want the SMALLEST i.
            # So we should decrement i_ptr while primes[i_ptr - 1] >= L.
            
            # But what if primes[j-1] < L?
            # Then no i works.
            # In this case, we skip updating dp[j][k] (it stays 2).
            # And for subsequent k, L is smaller, so primes[j-1] might become >= L.
            # So we need to handle the case where i_ptr is "invalid" initially.
            
            # Let's re-evaluate.
            # i_ptr maintains the smallest index found SO FAR that satisfies the condition for the CURRENT k?
            # No, i_ptr is monotonic.
            # Let's just use a while loop to find the correct i_ptr for each k?
            # No, that's O(K) per k, total O(K^3).
            
            # Correct logic:
            # i_ptr starts at j-1.
            # For each k:
            #   L = 2*P[j] - P[k]
            #   If primes[i_ptr] < L:
            #       Then no i <= i_ptr works.
            #       Since i_ptr is the max possible index (j-1), if P[j-1] < L, no i works.
            #       So new_len = 2.
            #       # Note: For next k, L is smaller, so P[j-1] might be >= L_next.
            #       # So we don't change i_ptr?
            #       # If P[j-1] < L, then for next k, if P[j-1] >= L_next,
            #       # then i_ptr should be j-1 (or smaller).
            #       # So we keep i_ptr at j-1 and re-check?
            #       # But if P[j-1] >= L_next, we then check if P[j-2] >= L_next, etc.
            #   Else (primes[i_ptr] >= L):
            #       # i_ptr is a valid index.
            #       # Can we go smaller?
            #       while i_ptr > 0 and primes[i_ptr - 1] >= L:
            #           i_ptr -= 1
            #       # Now i_ptr is the smallest index such that P[i_ptr] >= L.
            #       new_len = M[i_ptr] + 1
            
            # This logic works because:
            # 1. If P[i_ptr] < L, we do nothing. i_ptr stays.
            # 2. Next k, L' < L.
            # 3. If P[i_ptr] >= L', we enter the else branch and potentially decrease i_ptr.
            # 4. If P[i_ptr] < L', we do nothing.
            
            # Is it possible that P[i_ptr] < L, but a LARGER index works?
            # No, i_ptr starts at j-1 (max index).
            # If P[j-1] < L, then all P[i] for i<j are < L.
            # So no valid i.
            
            if primes[i_ptr] >= L:
                while i_ptr > 0 and primes[i_ptr - 1] >= L:
                    i_ptr -= 1
                new_len = M[i_ptr] + 1
                if new_len > dp[j][k]:
                    dp[j][k] = new_len
                    prev[j][k] = ArgM[i_ptr]
                    if new_len > max_len:
                        max_len = new_len
                        best_end_j = j
                        best_end_k = k
            # else: new_len remains 2, which is already initialized.
            # No update needed.

    # Reconstruct sequence
    if max_len == 2:
        return primes[:2]
        
    seq = [primes[best_end_k], primes[best_end_j]]
    cur_j = best_end_j
    cur_k = best_end_k
    
    while prev[cur_j][cur_k] != -1:
        prev_i = prev[cur_j][cur_k]
        seq.append(primes[prev_i])
        cur_k = cur_j
        cur_j = prev_i
        
    seq.reverse()
    return seq
