"""Rewritten program with optimized search for pairwise coprime sumsets."""
import itertools
import math
import time

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    # Fallback solution with 5 distinct pairwise coprime sums
    # A=[1, 7, 13], B=[4, 10, 16] -> Sums: 5, 11, 17, 23, 29 (all primes)
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    max_sum = 256
    
    # Precompute gcd table
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_pairwise_coprime(sums):
        """Check if all values in sums are pairwise coprime."""
        n = len(sums)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[sums[i]][sums[j]] != 1:
                    return False
        return True

    # Search strategy:
    # 1. Try small odd A and even B to ensure sums are odd (and potentially prime).
    # 2. Use pruning: if any pair of sums from the same row/col is not coprime, skip.
    
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    # Limit search space to speed up
    A_candidates = odd_vals[:40]
    B_candidates = even_vals[:40]
    
    for A in itertools.combinations(A_candidates, 3):
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        
        valid_b = []
        for b in B_candidates:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s3 > max_sum:
                continue
            # Check internal coprimality for this B
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Try to pick 3 from valid_b
        n = len(valid_b)
        # Limit combinations of B for performance
        max_b_combos = 1000
        count = 0
        for i in range(n):
            if time.time() > deadline:
                break
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            for j in range(i + 1, n):
                count += 1
                if count > max_b_combos:
                    break
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                # Check cross-coprimality between B1 and B2
                if (gcd_table[s1_1][s1_2] != 1 or gcd_table[s1_1][s2_2] != 1 or
                    gcd_table[s1_1][s3_2] != 1 or gcd_table[s2_1][s1_2] != 1 or
                    gcd_table[s2_1][s2_2] != 1 or gcd_table[s2_1][s3_2] != 1 or
                    gcd_table[s3_1][s1_2] != 1 or gcd_table[s3_1][s2_2] != 1 or
                    gcd_table[s3_1][s3_2] != 1):
                    continue
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    # Check cross-coprimality between B3 and B1, B2
                    if (gcd_table[s1_1][s1_3] != 1 or gcd_table[s1_1][s2_3] != 1 or
                        gcd_table[s1_1][s3_3] != 1 or gcd_table[s2_1][s1_3] != 1 or
                        gcd_table[s2_1][s2_3] != 1 or gcd_table[s2_1][s3_3] != 1 or
                        gcd_table[s3_1][s1_3] != 1 or gcd_table[s3_1][s2_3] != 1 or
                        gcd_table[s3_1][s3_3] != 1 or
                        gcd_table[s1_2][s1_3] != 1 or gcd_table[s1_2][s2_3] != 1 or
                        gcd_table[s1_2][s3_3] != 1 or gcd_table[s2_2][s1_3] != 1 or
                        gcd_table[s2_2][s2_3] != 1 or gcd_table[s2_2][s3_3] != 1 or
                        gcd_table[s3_2][s1_3] != 1 or gcd_table[s3_2][s2_3] != 1 or
                        gcd_table[s3_2][s3_3] != 1):
                        continue
                    
                    all_sums = [s1_1, s2_1, s3_1, s1_2, s2_2, s3_2, s1_3, s2_3, s3_3]
                    distinct_count = len(set(all_sums))
                    if distinct_count > best_score:
                        best_score = distinct_count
                        best_result = [list(A), [b1, b2, b3]]
                        if best_score == 9:
                            return best_result
                    if count > max_b_combos:
                        break
            if count > max_b_combos:
                break
    
    return best_result
