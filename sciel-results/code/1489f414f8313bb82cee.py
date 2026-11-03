import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def fast_score(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        
        # For very large sets, numpy might be faster, but pure Python sets 
        # are often competitive for n < 1000 if optimized.
        # Let's use numpy for robustness on speed for n up to ~1000.
        import numpy as np
        U_arr = np.array(sorted(U_set), dtype=np.int64)
        
        # Compute sumset and diffset
        # n=1000 -> 1M elements. ~8MB. Acceptable.
        # n=2000 -> 4M elements. ~32MB. Acceptable.
        # n=5000 -> 25M elements. ~200MB. Risky.
        
        if n > 2000:
             # Fallback to set-based if numpy memory is a concern, 
             # but for this task, we target n around 100-500.
             pass

        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        # Count unique elements
        # Using unique is O(N log N) where N = n^2
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_golomb(target_size, max_val=10000):
        """
        Construct a Golomb ruler (or near-Golomb) of given size.
        A Golomb ruler has all pairwise differences distinct.
        This implies |U-U| = 2*n*(n-1) + 1 (including 0 and negatives).
        Actually, if all positive differences are distinct, then U-U has size 2*n*(n-1) + 1.
        For a Sidon set (distinct sums), |U+U| is roughly n^2/2 + n.
        The ratio |U-U|/|U+U| is maximized when differences are unique (Golomb) 
        and sums are as "random" as possible (which Golomb rulers tend to be).
        
        We use a greedy algorithm to build a Golomb ruler.
        """
        U = [0, 1]
        
        for _ in range(target_size - 2):
            best_val = None
            best_len = -1
            
            # We want to add a value v such that all new differences |v - u| for u in U are unique 
            # and do not collide with existing differences.
            
            # Existing differences
            existing_diffs = set()
            for i in range(len(U)):
                for j in range(i+1, len(U)):
                    existing_diffs.add(abs(U[i] - U[j]))
            
            # Candidate range: usually slightly larger than current max
            # Greedy: try smallest v that doesn't create collision
            current_max = max(U)
            
            # We can search from current_max + 1 upwards
            # To be efficient, we can jump. But for n<100, linear scan is fine.
            # Max length of Golomb ruler of order n is approx n^2.
            # So we don't need to search too far.
            
            # Optimization: Pre-compute existing diffs as a set.
            # For each candidate v, check if any (v - u) is in existing_diffs or (u - v) is in existing_diffs.
            # Since v > u, v-u > 0.
            
            v = current_max + 1
            # We might need to search a bit further if collisions happen
            # But greedy usually works well.
            
            while True:
                # Check if v creates any duplicate difference
                valid = True
                for u in U:
                    d = v - u
                    if d in existing_diffs:
                        valid = False
                        break
                if valid:
                    best_val = v
                    break
                v += 1
                
                # Safety break if we search too far (should not happen for small n)
                if v > max_val:
                    break
            
            if best_val is not None:
                U.append(best_val)
                # Update existing_diffs
                for u in U[:-1]:
                    existing_diffs.add(best_val - u)
            else:
                # If we can't find a valid next element, try a slightly larger jump?
                # Or just stop? 
                # For standard Golomb rulers, greedy always works.
                break
                
        return U

    def local_search_refinement(U, time_limit):
        """
        Refine the set using local search to potentially improve the sumset structure
        while maintaining the difference set structure (or sacrificing a little diff for better sum).
        Actually, for Golomb rulers, the diff set is already maximal (2n(n-1)+1).
        Any change that breaks Golomb property reduces |U-U|.
        Does it reduce |U+U| enough to increase the ratio?
        Ratio = (2n(n-1)+1) / |U+U|.
        Minimizing |U+U| is the goal.
        Sidon sets minimize |U+U|.
        Greedy Golomb rulers are often good Sidon sets.
        
        Local search: try swapping or shifting elements to reduce sum collisions.
        """
        current_set = sorted(set(U))
        current_score = fast_score(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        while time.time() < time_limit:
            improved = False
            n = len(current_set)
            
            if n <= 1:
                break
                
            # Pick a random element (not 0) to modify
            idx = random.randint(1, n - 1)
            old_val = current_set[idx]
            
            # Try small perturbations
            for _ in range(20):
                # Shift by small amount
                shift = random.randint(-5, 5)
                new_val = old_val + shift
                
                if new_val <= 0:
                    continue
                
                # Check if new_val is already in set
                if new_val in current_set:
                    continue
                
                candidate_set = current_set[:]
                candidate_set[idx] = new_val
                candidate_set = sorted(set(candidate_set))
                
                new_score = fast_score(candidate_set)
                
                if new_score > current_score:
                    current_set = candidate_set
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    improved = True
                    break
            
            if not improved:
                # If no improvement after a few tries, maybe try a larger jump?
                # Or just continue.
                pass
                
        return best_set, best_score

    best_overall_score = 0.0
    best_overall_set = [0, 1]
    
    # Strategy:
    # 1. Generate Golomb rulers of various sizes.
    # 2. Score them.
    # 3. Pick the best size.
    # 4. Refine the best one with local search.
    
    # The score depends on n and max(U).
    # For Golomb ruler of size n, max(U) ~ n^2.
    # |U-U| ~ 2n^2.
    # |U+U| ~ n^2 (if Sidon).
    # Ratio ~ 2.
    # log(2) / log(2n^2) ~ 1 / (2 log n).
    # Score = 1 + small positive number.
    # As n increases, the additive term decreases.
    # So small n might be better?
    # Let's check n=2: U={0,1}. max=1. |U-U|=3, |U+U|=3. Ratio=1. Score=1.
    # n=3: U={0,1,3} (Golomb). max=3. |U-U|=5 (0,1,2,3,4? No. Diffs: 1,2,3. Set: {-3,-2,-1,0,1,2,3}. Size 7).
    # Sums: 0,1,3,4,6. Size 5. Ratio 7/5 = 1.4.
    # Score = 1 + log(1.4)/log(7) = 1 + 0.142/1.94 = 1.073.
    # n=4: U={0,1,4,6} (Golomb). max=6.
    # Diffs: 1,4,6,3,5,2. All distinct. Size 2*4*3+1 = 25? No.
    # Pairs: (0,1)=1, (0,4)=4, (0,6)=6, (1,4)=3, (1,6)=5, (4,6)=2.
    # Diffs: {-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6}. Size 13.
    # Sums: 0,1,4,6,2,5,8,10,12. 
    # 0+0=0, 0+1=1, 0+4=4, 0+6=6
    # 1+1=2, 1+4=5, 1+6=7
    # 4+4=8, 4+6=10
    # 6+6=12
    # Sums: {0,1,2,4,5,6,7,8,10,12}. Size 10.
    # Ratio 13/10 = 1.3.
    # Score = 1 + log(1.3)/log(13) = 1 + 0.117/2.56 = 1.045.
    # So n=3 is better than n=4.
    
    # It seems the score peaks at small n.
    # Let's test n=5.
    # Golomb: 0, 1, 4, 9, 11? (0,1,4,9,11 is not Golomb? 1-0=1, 4-1=3, 9-4=5, 11-9=2. Diffs: 1,3,5,2,4,8,10,11,7,9. Distinct? Yes.)
    # Max 11.
    # |U-U| = 2*5*4+1 = 41.
    # Sums? Likely ~20-25.
    # Ratio ~ 41/22 = 1.86.
    # Score = 1 + log(1.86)/log(23) = 1 + 0.30/3.13 = 1.096.
    
    # So larger n might be better initially.
    # Let's run greedy for n=3 to n=50 and pick the best.
    
    for n in range(3, 60):
        if time.time() > deadline - 30:
            break
            
        U = greedy_golomb(n, max_val=10000)
        if len(U) < n:
            continue
            
        score = fast_score(U)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = U[:]
            
    # Refine the best set found
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        improved_set, improved_score = local_search_refinement(best_overall_set, time.time() + remaining_time)
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set[:]

    # Also check if a simple arithmetic progression or quadratic is better?
    # Unlikely, but quick check.
    U_quad = [i*i for i in range(10)]
    s_quad = fast_score(U_quad)
    if s_quad > best_overall_score:
        best_overall_score = s_quad
        best_overall_set = U_quad[:]

    return best_overall_set

if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    if U:
        max_U = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a+b)
                diffs.add(a-b)
        if sums:
            ratio = len(diffs)/len(sums)
            score = 1 + math.log(ratio)/math.log(2*max_U+1)
            print(f"Score: {score}")
