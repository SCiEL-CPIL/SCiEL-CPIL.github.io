import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Step 1: Greedy initialization
    # We want to maximize sum 1/(512-a).
    # Sort candidates by a descending (which is 1/(512-a) descending).
    candidates = list(range(511, 0, -1))
    
    S = []
    score = 0.0
    
    for a in candidates:
        if all(math.gcd(a, b) == 1 for b in S):
            S.append(a)
            score += 1.0 / (512 - a)
            
    # Step 2: Local Search (1-for-k swaps)
    # Try to remove each element x, then greedily fill the gaps.
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(S)):
            x = S[i]
            # Tentatively remove x
            S_temp = S[:i] + S[i+1:]
            score_temp = score - 1.0 / (512 - x)
            
            # Greedily add back numbers
            # We iterate through all candidates, but we can skip those already in S_temp
            # To be efficient, we can just check all a in 511..1
            # But we must ensure we don't re-add x if it's still the best choice?
            # The greedy fill will re-add x if it's coprime to the rest and has high value.
            # So this naturally handles the "keep x" case too, but we are specifically testing "remove x".
            # Actually, if we remove x and then greedily fill, we might just add x back.
            # In that case, score_temp == score. No improvement.
            # We want to see if we can get *more* than score.
            
            new_elements = []
            for a in range(511, 0, -1):
                if a in S_temp:
                    continue
                if all(math.gcd(a, b) == 1 for b in S_temp):
                    S_temp.append(a)
                    new_elements.append(a)
                    score_temp += 1.0 / (512 - a)
            
            if score_temp > score + 1e-9:
                S = S_temp
                score = score_temp
                improved = True
                # Break inner loop to restart with new S
                break
                
    # Optional: Try 2-for-k swaps?
    # Given the time limit and the nature of the problem, 1-for-k is usually sufficient for significant gains.
    # The 1-for-k swap effectively allows replacing one "bad" blocker with multiple "good" non-blockers.
    
    return S

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    S = []
    score = 0.0
    
    # Greedy Init
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in S):
            S.append(a)
            score += 1.0 / (512 - a)
            
    # Local Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(S)):
            x = S[i]
            S_temp = S[:i] + S[i+1:]
            score_temp = score - 1.0 / (512 - x)
            
            for a in range(511, 0, -1):
                if a in S_temp:
                    continue
                if all(math.gcd(a, b) == 1 for b in S_temp):
                    S_temp.append(a)
                    score_temp += 1.0 / (512 - a)
            
            if score_temp > score + 1e-9:
                S = S_temp
                score = score_temp
                improved = True
                break
                
    return S
# EVOLVE-BLOCK-END
