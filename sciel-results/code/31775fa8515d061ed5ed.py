import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search for better sets
    # We know [0, 1, 3] gives ~1.079
    # Let's find the maximum possible score for small sets
    
    def calc_score(U):
        if not U:
            return 0.0
        M = max(U)
        if M == 0:
            return 1.0
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
        
        u_set = set(U)
        sum_set = set()
        diff_set = set()
        
        # Optimize: U is small
        for a in u_set:
            for b in u_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
                
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1:
            return 1.0
            
        score = 1.0 + math.log(ratio) / denom
        return score

    # Search strategy
    # Try k from 2 to 12
    # Try M from 1 to 40
    
    max_M = 40
    max_k = 12
    
    for k in range(2, max_k + 1):
        for M in range(1, max_M + 1):
            # We need k-1 elements from 1..M
            # If M < k-1, skip
            if M < k - 1:
                continue
                
            # Generate combinations
            # To avoid duplicates and speed up, we can fix 0 in the set
            # The other k-1 elements are chosen from 1..M
            
            # Pruning: if M is large, score might be small?
            # log(ratio) / log(2M). Ratio is bounded by O(k^2)/O(k^2) = O(1)?
            # Actually ratio can be up to 2.
            # So score ~ 1 + log(2)/log(2M). Decreases with M.
            # So we should prioritize small M.
            
            # We can limit M based on k?
            # For k=3, M=3 was good.
            # For k=4, M=6 was 1.075.
            
            # Let's just iterate.
            # Use itertools.combinations
            # This might be slow for large k, M.
            # We will break if time is up.
            
            # Optimization: sort candidates? No, just check.
            
            # If k is large, M should be small?
            # If k is small, M can be larger?
            
            # Let's restrict M to be proportional to k?
            # Or just check all.
            
            # To speed up, we can skip if 2*M+1 is such that even max ratio 2 gives score < best_score
            # Max ratio is approx 2 (for Sidon-like).
            # 1 + log(2)/log(2M+1) < best_score
            # log(2)/log(2M+1) < best_score - 1
            # log(2M+1) > log(2) / (best_score - 1)
            # 2M+1 > 2^(1/(best_score-1))
            
            threshold = 2 ** (1 / (best_score - 1)) if best_score > 1 else float('inf')
            if 2 * M + 1 > threshold:
                # It's possible that ratio > 2? No, max ratio is 2.
                # Actually |U-U| <= 2|U|-1? No.
                # |U-U| can be (2M+1). |U+U| can be |U|.
                # Ratio (2M+1)/k.
                # If k is small, ratio can be large.
                # e.g. k=2, M=100. U={0, 100}.
                # U+U={0, 100, 200}. Size 3.
                # U-U={-100, 0, 100}. Size 3.
                # Ratio 1.
                
                # U={0, 1}. U+U={0,1,2}. U-U={-1,0,1}. Ratio 1.
                
                # U={0, 2}. U+U={0,2,4}. U-U={-2,0,2}. Ratio 1.
                
                # It seems ratio is bounded.
                
                # Let's just check.
                pass
            
            # Iterate combinations
            # To make it faster, we can stop early if time is up
            for combo in itertools.combinations(range(1, M + 1), k - 1):
                if time.time() > deadline:
                    break
                
                U = [0] + list(combo)
                # Ensure sorted? Not strictly necessary for set ops, but good for consistency
                # U is already sorted since combo is sorted
                
                score = calc_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return best_set
