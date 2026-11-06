import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # Weights as Fractions for exactness in comparison, but float for speed in pruning?
    # The score is floating serialization. The prompt says "exact sum of reciprocals; score is its floating serialization".
    # So we should maximize the exact sum.
    # Using Fraction is slow.
    # However, N=256. The number of elements is small.
    # We can use float for pruning and Fraction for the final best score comparison?
    # Or just float? "Maximize the exact sum".
    # If two sets have very close sums, float might distinguish them incorrectly.
    # But the scores are likely distinct enough.
    # Let's use float for speed in the search loop, and verify the best solution with Fraction at the end?
    # Or use Fraction but optimize the loop.
    # Fraction addition is slow.
    # Let's try float first. If it fails precision, switch.
    # Actually, the "score is its floating serialization" implies the evaluator converts the result to float.
    # But "Maximize the exact sum" implies we should use exact arithmetic to decide which set is better.
    # Let's use Fraction for the best_score comparison, but maybe float for the pruning bound?
    # Suffix sum of floats is a safe upper bound for the exact sum?
    # Sum of 1/i is convex? No.
    # Float error is small.
    # Let's use Fraction for correctness, but optimize the inner loop.
    
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    # Suffix sums for pruning
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    best_score = Fraction(0)
    best_mask = 0
    best_set = []
    
    # Greedy initialization
    mask = 0
    in_S = [False] * (N + 1)
    S_list = []
    
    for i in range(1, N + 1):
        valid = True
        # Check APs ending at i
        # a, b, i => 2b = a+i
        for a in S_list:
            s = a + i
            if s % 2 == 0:
                b = s // 2
                if in_S[b]:
                    valid = False
                    break
        if valid:
            in_S[i] = True
            S_list.append(i)
            mask |= (1 << (i - 1))
            
    if S_list:
        best_score = sum(weights[x] for x in S_list)
        best_mask = mask
        best_set = list(S_list)
        
    # Precompute AP pairs for each x
    # pairs[x] = list of (a, b) such that a, b, x is AP
    # a = x - 2d, b = x - d
    ap_pairs = [[] for _ in range(N + 1)]
    for x in range(3, N + 1):
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1:
                ap_pairs[x].append((a, b))
                
    def is_valid_add(x, mask):
        # Check if adding x creates a 3-AP with existing elements in mask
        # Only need to check if x is the largest term
        for a, b in ap_pairs[x]:
            # Check if a and b are in mask
            # Indices in mask are 0-based: element k is at bit k-1
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))):
                return False
        return True

    def dfs(idx, current_sum, mask):
        nonlocal best_score, best_mask, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix[idx] <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_mask = mask
                # Reconstruct set
                s = []
                for i in range(1, N + 1):
                    if mask & (1 << (i - 1)):
                        s.append(i)
                best_set = s
            return
            
        x = idx
        
        # Option 1: Include x
        if is_valid_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            dfs(idx + 1, current_sum + weights[x], new_mask)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(1, Fraction(0), 0)
    
    return best_set
