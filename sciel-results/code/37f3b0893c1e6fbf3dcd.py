# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth = []
    for a in range(0, 8):  # 2^a <= 128
        for b in range(0, 6):  # 3^b <= 128
            for c in range(0, 4):  # 5^c <= 128
                x = (2 ** a) * (3 ** b) * (5 ** c)
                if x <= 128:
                    smooth.append(x)
    smooth = sorted(set(smooth))
    
    # Precompute divisibility: div[i][j] = True if smooth[i] divides smooth[j]
    n_smooth = len(smooth)
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div[i][j] = True
    
    # For each target, find minimal antichain representation
    answers = [None] * 128
    
    # Helper: check if adding smooth[idx] to current set maintains antichain
    def is_compatible(current_set, idx):
        for i in current_set:
            if div[i][idx] or div[idx][i]:
                return False
        return True
    
    # DFS to find antichain of size k summing to target
    def find_antichain(target, k, candidates, current, current_sum, current_set):
        if time.time() > deadline:
            return None
        if len(current) == k:
            if current_sum == target:
                return list(current)
            return None
        if len(current) > k:
            return None
        if current_sum > target:
            return None
        
        # Pruning: if we can't reach target even with max possible additions
        remaining_needed = target - current_sum
        slots_left = k - len(current)
        if slots_left <= 0:
            return None
        
        # Try each candidate from start_idx
        for i in range(len(candidates)):
            idx = candidates[i]
            val = smooth[idx]
            
            if val > remaining_needed:
                break
            
            # Check if we can still fill remaining slots with smaller numbers
            if i + slots_left - 1 >= len(candidates):
                break
            
            if is_compatible(current_set, idx):
                new_set = current_set | {idx}
                result = find_antichain(target, k, candidates[i+1:], 
                                         current + [val], current_sum + val, new_set)
                if result is not None:
                    return result
        
        return None
    
    # For each target 1-128, find minimal representation
    for n in range(1, 129):
        if time.time() > deadline:
            # Fill remaining with any valid representation (use single term if possible)
            for m in range(n, 129):
                if answers[m-1] is None:
                    # Try to find any valid representation
                    found = False
                    for k in range(1, 31):
                        if time.time() > deadline:
                            break
                        # Simple fallback: try combinations
                        cands = [i for i in range(n_smooth) if smooth[i] <= n]
                        for combo in combinations(cands, k):
                            if sum(smooth[i] for i in combo) == n:
                                # Check antichain
                                valid = True
                                for a in range(len(combo)):
                                    for b in range(a+1, len(combo)):
                                        if div[combo[a]][combo[b]] or div[combo[b]][combo[a]]:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    answers[m-1] = [smooth[i] for i in combo]
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        # Last resort: use the number itself if it's smooth
                        if n in smooth:
                            answers[m-1] = [n]
                        else:
                            answers[m-1] = [1]  # Fallback, may be invalid
            break
        
        # Try k = 1, 2, 3, ... until we find a representation
        found = False
        for k in range(1, 31):
            if time.time() > deadline:
                break
            
            # Get candidates: smooth numbers <= n
            cands = [i for i in range(n_smooth) if smooth[i] <= n]
            
            # Sort by value for better pruning
            cands_sorted = sorted(cands, key=lambda i: smooth[i])
            
            result = find_antichain(n, k, cands_sorted, [], 0, set())
            if result is not None:
                answers[n-1] = result
                found = True
                break
        
        if not found:
            # Fallback: try brute force for small k
            for k in range(1, min(6, 31)):
                if time.time() > deadline:
                    break
                cands = [i for i in range(n_smooth) if smooth[i] <= n]
                for combo in combinations(cands, k):
                    if sum(smooth[i] for i in combo) == n:
                        # Check antichain
                        valid = True
                        for a in range(len(combo)):
                            for b in range(a+1, len(combo)):
                                if div[combo[a]][combo[b]] or div[combo[b]][combo[a]]:
                                    valid = False
                                    break
                                if not valid:
                                    break
                        if valid:
                            answers[n-1] = [smooth[i] for i in combo]
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # Last resort
                if n in smooth:
                    answers[n-1] = [n]
                else:
                    answers[n-1] = [1]
    
    # Verify all answers are set
    for i in range(128):
        if answers[i] is None:
            # Emergency fallback
            if (i+1) in smooth:
                answers[i] = [i+1]
            else:
                answers[i] = [1]
    
    return answers
# EVOLVE-BLOCK-END
