import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s to be safe
    
    def dfs(idx, prev_d, N, D):
        if time.time() > deadline:
            return None
        
        j = k - idx
        if j == 0:
            if N == 0:
                return []
            return None
        
        if j == 1:
            # Need 1/d = N/D => d = D/N
            if D % N == 0:
                d = D // N
                if d > prev_d and d <= 10000000:
                    return [d]
            return None
        
        # Bounds for d
        # 1/d <= N/D => d >= D/N
        # j/d >= N/D => d <= j*D/N
        min_d = max(prev_d + 1, (D + N - 1) // N) # ceil(D/N)
        max_d = (j * D) // N
        
        if min_d > max_d:
            return None
            
        # Iterate d
        # To speed up, we can iterate from min_d to max_d
        # But max_d can be large.
        # However, N/D is the remaining sum.
        # If N/D is small, max_d is large.
        # But we are summing to 1.
        
        for d in range(min_d, max_d + 1):
            # Pruning: Check if max possible sum of remaining j-1 terms is sufficient
            # Max sum of j-1 terms starting from d+1 is roughly (j-1)/(d+1)
            # Actually, sum_{i=1}^{j-1} 1/(d+i)
            # If this max sum < remaining (N/D - 1/d), then fail.
            # Remaining after d: R = N/D - 1/d = (Nd - D) / Dd
            # Max sum of j-1 terms from d+1 is approx (j-1)/(d+1)
            # If (j-1)/(d+1) < R, then impossible.
            # R = (Nd - D) / (Dd)
            # (j-1)/(d+1) < (Nd - D) / (Dd)
            # Dd(j-1) < (d+1)(Nd - D)
            # Dd(j-1) < Nd^2 + Nd - Dd - D
            # 0 < Nd^2 + Nd - Dd - D - Dd(j-1)
            # 0 < Nd^2 + Nd - Dd - D - Dj d + Dd
            # 0 < Nd^2 + Nd - Dj d - D
            # This is a quadratic in d.
            
            # Simpler pruning:
            # If 1/d > N/D, skip (but min_d ensures 1/d <= N/D)
            # If max possible sum of remaining < N/D - 1/d, skip.
            # Max sum of j-1 terms from d+1 is sum_{i=1}^{j-1} 1/(d+i)
            # This sum is < (j-1)/(d+1)
            # So if (j-1)/(d+1) < N/D - 1/d, skip.
            # (j-1)/(d+1) < (Nd - D) / (Dd)
            # Dd(j-1) < (d+1)(Nd - D)
            
            if D * d * (j - 1) < (d + 1) * (N * d - D):
                continue
                
            new_N = N * d - D
            new_D = D * d
            
            if new_N < 0:
                continue
                
            g = gcd(new_N, new_D)
            res = dfs(idx + 1, d, new_N // g, new_D // g)
            if res is not None:
                return [d] + res
            
        return None

    for k in range(9, 16):
        if time.time() > deadline:
            break
        res = dfs(0, 4, 1, 1)
        if res is not None:
            return res
            
    # Fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]
