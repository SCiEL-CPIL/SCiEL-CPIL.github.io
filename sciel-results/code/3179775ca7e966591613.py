import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    N = 1000001  # we need n and n+1, so up to 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    def powerful_part(n):
        if n <= 1:
            return 1
        pp = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count >= 2:
                pp *= p ** count
        return pp
    
    # Precompute powerful parts for all n in [1, N]
    pp_arr = [0] * (N + 1)
    pp_arr[0] = 0
    pp_arr[1] = 1
    for i in range(2, N + 1):
        pp_arr[i] = powerful_part(i)
    
    # Find n in [10000, 1000000] that maximizes pp_arr[n] * pp_arr[n+1]
    best_n = 10000
    best_val = pp_arr[10000] * pp_arr[10001]
    
    for n in range(10000, 1000001):
        val = pp_arr[n] * pp_arr[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    # Phase 2: CRT-based search for large powerful parts
    # Look for n = A*s, n+1 = B*t where A, B have large powerful parts
    # and gcd(A,B) = 1 (since gcd(n, n+1) = 1)
    
    # Collect numbers with large powerful parts
    large_pp = {}  # pp_value -> list of numbers with that powerful part
    for i in range(10000, 1000001):
        pp = pp_arr[i]
        if pp >= 100:  # threshold for "large"
            if pp not in large_pp:
                large_pp[pp] = []
            large_pp[pp].append(i)
    
    # Sort by powerful part descending
    sorted_pps = sorted(large_pp.keys(), reverse=True)
    
    # For top powerful parts, try CRT combinations
    for A_pp in sorted_pps[:50]:  # top 50 powerful part values
        if time.time() > deadline:
            break
        A_nums = large_pp[A_pp]
        
        for B_pp in sorted_pps[:50]:
            if time.time() > deadline:
                break
            
            # Check if A_pp and B_pp are coprime (necessary for CRT)
            if gcd(A_pp, B_pp) != 1:
                continue
            
            B_nums = large_pp[B_pp]
            
            # For each pair, try to find n = A*s, n+1 = B*t
            # This means n ≡ 0 (mod A) and n ≡ -1 (mod B)
            # CRT: n ≡ x (mod lcm(A,B)) where x satisfies the system
            # Since gcd(A,B)=1, solution exists and is unique mod A*B
            
            for a in A_nums[:10]:  # limit to first 10 to save time
                if time.time() > deadline:
                    break
                # a should be divisible by A_pp? No, a's powerful part IS A_pp
                # We need n such that pp(n) >= A_pp and pp(n+1) >= B_pp
                # Actually, let's think differently:
                # If n has powerful part A_pp, then A_pp | n
                # If n+1 has powerful part B_pp, then B_pp | (n+1)
                # So n ≡ 0 (mod A_pp) and n ≡ -1 (mod B_pp)
                
                # Find all n in range where n ≡ 0 (mod A_pp) and n ≡ -1 (mod B_pp)
                # Using CRT: n = A_pp * k, and A_pp * k ≡ -1 (mod B_pp)
                # k ≡ -A_pp^{-1} (mod B_pp)
                
                try:
                    # Find inverse of A_pp mod B_pp
                    inv = pow(A_pp, -1, B_pp)
                    k0 = (-inv) % B_pp  # k ≡ k0 (mod B_pp)
                    
                    # n = A_pp * (k0 + m * B_pp) for m = 0, 1, 2, ...
                    # n in [10000, 1000000]
                    # A_pp * k0 + A_pp * m * B_pp >= 10000
                    # A_pp * k0 + A_pp * m * B_pp <= 1000000
                    
                    base = A_pp * k0
                    step = A_pp * B_pp
                    
                    if step == 0:
                        continue
                    
                    m_min = max(0, (10000 - base + step - 1) // step)
                    m_max = min((1000000 - base) // step, 1000000 // step)
                    
                    for m in range(m_min, m_max + 1):
                        n = base + m * step
                        if 10000 <= n <= 1000000:
                            val = pp_arr[n] * pp_arr[n + 1]
                            if val > best_val:
                                best_val = val
                                best_n = n
                except ValueError:
                    # Inverse doesn't exist (shouldn't happen since gcd=1)
                    continue
    
    return best_n
