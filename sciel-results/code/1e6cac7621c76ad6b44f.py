import math

def construct():
    # We established k=5,6,7 are impossible for min denom 5.
    # Try k=8, 9, ...
    
    def find_solution(k):
        # N/D is the fraction to represent
        # We want sum_{i=1..k} 1/d_i = N/D
        # d_i distinct, >= 5, strictly increasing
        path = []
        
        def dfs(idx, N, D, last_denom):
            if idx == k:
                return N == 0
            
            remaining_terms = k - idx
            
            # Lower bound for d:
            # 1/d <= N/D => d >= D/N
            # Also d > last_denom
            if N == 0:
                return False
            
            d_min = max(last_denom + 1, math.ceil(D / N))
            
            # Upper bound for d:
            # We need N/D - 1/d <= sum_{j=1}^{remaining_terms-1} 1/(d+j)
            # Approximate sum_{j=1}^{remaining_terms-1} 1/(d+j) <= (remaining_terms-1)/(d+1)
            # N/D - 1/d <= (remaining_terms-1)/(d+1)
            # N/D <= 1/d + (remaining_terms-1)/(d+1)
            # N/D <= (d+1 + d(remaining_terms-1)) / (d(d+1))
            # N/D <= (d * remaining_terms + 1) / (d(d+1))
            # N * d * (d+1) <= D * (d * remaining_terms + 1)
            # N * d^2 + N * d <= D * remaining_terms * d + D
            # N * d^2 + (N - D * remaining_terms) * d - D <= 0
            
            # Solve N * d^2 + (N - D * remaining_terms) * d - D = 0 for d
            # a = N, b = N - D * remaining_terms, c = -D
            a = N
            b = N - D * remaining_terms
            c = -D
            
            # Discriminant
            disc = b * b - 4 * a * c
            if disc < 0:
                return False
            
            sqrt_disc = math.isqrt(disc)
            # d <= (-b + sqrt_disc) / (2 * a)
            # Since d must be integer, d_max = floor((-b + sqrt_disc) / (2 * a))
            # Note: if sqrt_disc^2 != disc, we should be careful, but isqrt floors.
            # The root is exactly (-b + sqrt(disc)) / 2a.
            # We want the largest integer d satisfying the quadratic inequality.
            # Since the parabola opens upward (a>0), the inequality holds between the roots.
            # The lower root is negative (since c<0, product of roots c/a < 0), so we only care about the positive root.
            # d_max = floor((-b + sqrt_disc) / (2 * a))
            
            numerator = -b + sqrt_disc
            denominator = 2 * a
            d_max = numerator // denominator
            
            if d_max < d_min:
                return False
            
            for d in range(d_min, d_max + 1):
                # Check if 1/d is a valid step
                # New fraction: N/D - 1/d = (N*d - D) / (D*d)
                new_N = N * d - D
                new_D = D * d
                
                if new_N < 0:
                    # This shouldn't happen if d >= D/N, but due to integer division 
                    # in d_min calculation, D/N might be slightly less than d_min if N/D is not integer.
                    # Actually d_min = ceil(D/N). So d >= D/N => N*d >= D => new_N >= 0.
                    continue
                
                if new_N == 0:
                    if remaining_terms == 1:
                        path.append(d)
                        return True
                    else:
                        # We have 0 remaining sum but need more terms? 
                        # This means we summed to 1 too early. 
                        # But we are looking for exactly k terms. 
                        # If new_N == 0, we can't add more positive unit fractions.
                        # So this branch is a dead end unless we are at the last term.
                        continue
                
                # Pruning: 
                # The next denominators must be > d.
                # The minimum possible sum for the remaining (remaining_terms - 1) terms
                # is not easily bounded from below to prune "too large" new_N?
                # Actually, we can check if the maximum possible sum of the remaining terms
                # is less than new_N / new_D.
                # Max sum of (remaining_terms - 1) distinct unit fractions with denom > d
                # is sum_{i=1}^{remaining_terms-1} 1/(d+i).
                # If new_N/new_D > sum_{i=1}^{remaining_terms-1} 1/(d+i), then it's impossible.
                
                # We already used the approximation (remaining_terms-1)/(d+1) to find d_max.
                # That approximation was for the CURRENT d.
                # Here we are checking if the REMAINDER can be formed.
                # The condition used for d_max was:
                # N/D - 1/d <= (remaining_terms-1)/(d+1)
                # Which is exactly new_N/new_D <= (remaining_terms-1)/(d+1).
                # So this condition is already satisfied by the loop range.
                
                # However, the approximation (remaining_terms-1)/(d+1) is an overestimate 
                # of the true max sum sum_{i=1}^{remaining_terms-1} 1/(d+i).
                # So we might pass the d_max check but still fail.
                # We can add a tighter check here if needed, but the recursion will handle it.
                
                path.append(d)
                if dfs(idx + 1, new_N, new_D, d):
                    return True
                path.pop()
            
            return False
        
        # Start with N=1, D=1
        if dfs(0, 1, 1, 4):
            return path
        
        return None

    # Search for k starting from 8
    for k in range(8, 20):
        sol = find_solution(k)
        if sol:
            return sol
            
    # Fallback: Greedy (Sylvester's sequence or similar)
    # This might not be minimal but guarantees a result if search fails
    # But we expect search to succeed for small k.
    
    # If no solution found in range, return a valid non-minimal solution
    # Greedy algorithm for Egyptian fractions
    rem_num, rem_den = 1, 1
    denoms = []
    last = 4
    while rem_num > 0:
        d = max(last + 1, math.ceil(rem_den / rem_num))
        denoms.append(d)
        # rem = rem - 1/d
        new_num = rem_num * d - rem_den
        new_den = rem_den * d
        rem_num, rem_den = new_num, new_den
        last = d
        if len(denoms) > 256:
             break
             
    return denoms
