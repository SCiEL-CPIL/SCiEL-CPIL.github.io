"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    max_sum = 256
    
    # Precompute prime factorization for each number up to max_sum
    # For each number, store its set of prime factors
    prime_factors = [set() for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            for j in range(i, max_sum + 1, i):
                prime_factors[j].add(i)
    
    # Precompute gcd table for faster checks
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def check_pairwise_coprime(sums_list):
        """Check if all values in sums_list are pairwise coprime using prime factors."""
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[sums_list[i]][sums_list[j]] != 1:
                    return False
        return True
    
    def get_distinct_count(sums_list):
        return len(set(sums_list))
    
    # Phase 1: Odd A, Even B (avoids even-even sum collisions)
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    A_search = odd_vals[:60]   # 1 to 119
    B_search = even_vals[:60]  # 2 to 120
    
    for A in itertools.combinations(A_search, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s3 > max_sum:
                continue
            # Check that sums within this B are pairwise coprime
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Limit to top candidates for performance
        max_valid = 50
        if len(valid_b) > max_valid:
            valid_b = valid_b[:max_valid]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                # Quick check: 6 sums must be pairwise coprime
                if (gcd_table[s1_1][s1_2] != 1 or gcd_table[s1_1][s2_2] != 1 or
                    gcd_table[s1_1][s3_2] != 1 or gcd_table[s2_1][s1_2] != 1 or
                    gcd_table[s2_1][s2_2] != 1 or gcd_table[s2_1][s3_2] != 1 or
                    gcd_table[s3_1][s1_2] != 1 or gcd_table[s3_1][s2_2] != 1 or
                    gcd_table[s3_1][s3_2] != 1):
                    continue
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    # Check remaining 9 cross-pairs
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
    
    # Phase 2: Try mixed-parity combinations with broader search
    all_vals = list(range(1, 129))
    
    # Try different parity patterns
    for a_parity in ['odd', 'even', 'mixed']:
        if time.time() > deadline:
            return best_result
        
        if a_parity == 'odd':
            A_search_mixed = odd_vals[:50]
        elif a_parity == 'even':
            A_search_mixed = even_vals[:50]
        else:
            # Mix: some odd, some even
            A_search_mixed = odd_vals[:25] + even_vals[:25]
        
        for A in itertools.combinations(A_search_mixed, 3):
            if time.time() > deadline:
                return best_result
            a1, a2, a3 = A
            
            valid_b = []
            for b in all_vals:
                if b in A:
                    continue
                s1 = a1 + b
                s2 = a2 + b
                s3 = a3 + b
                if s3 > max_sum:
                    continue
                if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                    valid_b.append((b, s1, s2, s3))
            
            if len(valid_b) < 3:
                continue
            
            max_valid = 35
            if len(valid_b) > max_valid:
                valid_b = valid_b[:max_valid]
            
            n = len(valid_b)
            for i in range(n):
                if time.time() > deadline:
                    return best_result
                b1, s1_1, s2_1, s3_1 = valid_b[i]
                for j in range(i + 1, n):
                    b2, s1_2, s2_2, s3_2 = valid_b[j]
                    if (gcd_table[s1_1][s1_2] != 1 or gcd_table[s1_1][s2_2] != 1 or
                        gcd_table[s1_1][s3_2] != 1 or gcd_table[s2_1][s1_2] != 1 or
                        gcd_table[s2_1][s2_2] != 1 or gcd_table[s2_1][s3_2] != 1 or
                        gcd_table[s3_1][s1_2] != 1 or gcd_table[s3_1][s2_2] != 1 or
                        gcd_table[s3_1][s3_2] != 1):
                        continue
                    for k in range(j + 1, n):
                        b3, s1_3, s2_3, s3_3 = valid_b[k]
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
    
    # Phase 3: Local optimization - try to improve best_result by perturbing
    if time.time() < deadline - 10 and best_score < 9:
        A_best = list(best_result[0])
        B_best = list(best_result[1])
        
        # Try replacing one element in A or B
        for _ in range(3):  # Up to 3 rounds of local search
            improved = False
            for idx in range(3):
                # Try changing A[idx]
                for new_a in range(1, 129):
                    if new_a in A_best:
                        continue
                    new_A = A_best[:]
                    new_A[idx] = new_a
                    if len(set(new_A)) != 3:
                        continue
                    
                    # Check if this gives better or equal score
                    new_sums = set()
                    valid = True
                    for a in new_A:
                        for b in B_best:
                            s = a + b
                            new_sums.add(s)
                    
                    if len(new_sums) > best_score and check_pairwise_coprime(list(new_sums)):
                        A_best = new_A
                        best_result = [A_best, B_best]
                        best_score = len(new_sums)
                        improved = True
                        if best_score == 9:
                            return best_result
                        break
                
                if time.time() > deadline:
                    return best_result
                
                # Try changing B[idx]
                for new_b in range(1, 129):
                    if new_b in B_best:
                        continue
                    new_B = B_best[:]
                    new_B[idx] = new_b
                    if len(set(new_B)) != 3:
                        continue
                    
                    new_sums = set()
                    for a in A_best:
                        for b in new_B:
                            s = a + b
                            new_sums.add(s)
                    
                    if len(new_sums) > best_score and check_pairwise_coprime(list(new_sums)):
                        B_best = new_B
                        best_result = [A_best, B_best]
                        best_score = len(new_sums)
                        improved = True
                        if best_score == 9:
                            return best_result
                        break
            
            if not improved:
                break
    
    return best_result
# EVOLVE-BLOCK-END
