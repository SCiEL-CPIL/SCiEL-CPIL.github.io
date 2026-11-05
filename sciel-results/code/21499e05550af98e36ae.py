import sys
from itertools import combinations

def construct():
    # Generate 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute compatibility matrix
    # compatible[i][j] is True if smooth[i] and smooth[j] do not divide each other
    n_smooth = len(smooth)
    compatible = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if b % a != 0 and a % b != 0:
                compatible[i][j] = True
                compatible[j][i] = True
            # If i==j, they are the same number, but we need distinct terms, so we won't pick same index twice.
            # The antichain condition is for distinct terms.
            
    answers = []
    
    # For each target n from 1 to 128
    for n in range(1, 129):
        found = None
        # Try k from 1 upwards
        for k in range(1, 11): # Max 10 terms is plenty
            # DFS to find k distinct smooth numbers summing to n, forming an antichain
            # We enforce index order i1 < i2 < ... < ik to avoid permutations
            def dfs(start_idx, remaining_sum, remaining_k, chosen_indices):
                if remaining_k == 0:
                    if remaining_sum == 0:
                        return True
                    return False
                
                if remaining_sum <= 0:
                    return False
                    
                # Pruning: even if we pick the largest possible remaining numbers, can we reach the sum?
                # Actually, since we are picking from sorted list, and we want exactly remaining_k numbers,
                # the minimum possible sum for the next 'remaining_k' numbers starting from 'start_idx' 
                # is smooth[start_idx] + ... + smooth[start_idx + remaining_k - 1]
                # If that min sum > remaining_sum, we can prune.
                if start_idx + remaining_k > n_smooth:
                    return False
                min_possible = sum(smooth[start_idx : start_idx + remaining_k])
                if min_possible > remaining_sum:
                    return False
                
                # Max possible sum check is harder, but min sum check is good.
                
                for i in range(start_idx, n_smooth):
                    # If smooth[i] > remaining_sum, we can break because list is sorted
                    if smooth[i] > remaining_sum:
                        break
                    
                    # Check compatibility with all chosen so far
                    is_compatible = True
                    for j in chosen_indices:
                        if not compatible[i][j]:
                            is_compatible = False
                            break
                    
                    if not is_compatible:
                        continue
                    
                    chosen_indices.append(i)
                    if dfs(i + 1, remaining_sum - smooth[i], remaining_k - 1, chosen_indices):
                        return True
                    chosen_indices.pop()
                
                return False

            # Initialize DFS
            chosen_indices = []
            if dfs(0, n, k, chosen_indices):
                found = [smooth[i] for i in chosen_indices]
                break
        
        if found is None:
            # Fallback: should not happen for 1..128, but to be safe
            # Try a simple greedy or just return [n] if n is smooth and no constraint violation?
            # But [n] is always an antichain (size 1).
            # If n is smooth, [n] works.
            if n in smooth:
                found = [n]
            else:
                # This case should not occur as we go up to k=10
                # But if it does, we need a valid representation.
                # For safety, we can just append a dummy valid one if we fail, 
                # but the problem guarantees valid representations exist.
                # Let's assume the DFS found it.
                found = [] 
                # Re-run with higher k if needed?
                # Actually, for n=1, smooth=[1], k=1 works.
                # For n=7, 7 is not smooth. 7 = 2+5? 2,5 smooth. 2|5? No. 5|2? No.
                # So k=2 works for 7.
                # The loop goes up to 10, which is sufficient.
                pass
                
        answers.append(found)
        
    return answers
