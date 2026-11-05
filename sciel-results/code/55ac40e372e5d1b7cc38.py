import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCMs
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Identify bad triples and build masks
    # bad_mask[i-1][j-1] is a bitmask of k (1..128) such that (i, j, k) is bad
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask

    def get_mask(S):
        """Compute conflict mask for set S."""
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask

    def can_add(x, mask_S):
        """Check if x can be added to set with conflict mask mask_S."""
        return (mask_S & (1 << (x - 1))) == 0

    # Greedy Initialization
    # Calculate degrees to sort
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            m = bad_mask[a-1][b-1]
            if m:
                cnt = bin(m).count('1')
                degrees[a] += cnt
                degrees[b] += cnt
                
    sorted_nums = sorted(numbers, key=lambda x: degrees[x])
    
    S = set()
    mask_S = 0
    
    for x in sorted_nums:
        if can_add(x, mask_S):
            # Calculate new mask before adding x to S
            new_mask = mask_S
            for y in S:
                new_mask |= bad_mask[x - 1][y - 1]
            S.add(x)
            mask_S = new_mask
            
    R = set(numbers) - S
    
    # Local Search
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Single Additions
        for x in list(R):
            if can_add(x, mask_S):
                new_mask = mask_S
                for y in S:
                    new_mask |= bad_mask[x - 1][y - 1]
                S.add(x)
                R.remove(x)
                mask_S = new_mask
                improved = True
        
        if not improved:
            # Phase 2: 1-for-2 Swaps
            best_swap = None
            best_swap_mask = None
            best_swap_score = -1 
            
            for u in list(S):
                S_temp = S - {u}
                mask_temp = get_mask(S_temp)
                
                for v in R:
                    if not can_add(v, mask_temp):
                        continue
                    
                    # Compute mask for (S \ {u}) \cup {v}
                    mask_temp2 = mask_temp
                    for y in S_temp:
                        mask_temp2 |= bad_mask[v - 1][y - 1]
                    
                    for w in R:
                        if w == v:
                            continue
                        if not can_add(w, mask_temp2):
                            continue
                        
                        # Valid swap: remove u, add v, w
                        # New Mask
                        final_mask = mask_temp2
                        for y in S_temp:
                            final_mask |= bad_mask[w - 1][y - 1]
                        final_mask |= bad_mask[w - 1][v - 1] # Wait, v is in S_temp2? 
                        # S_temp2 = S_temp U {v}. 
                        # My loop `for y in S_temp` covers S_temp. 
                        # I need to include v.
                        # So final_mask = mask_temp2 | OR(bad_mask[w][y] for y in S_temp2)
                        # mask_temp2 already includes conflicts from v with S_temp.
                        # So we just need to add conflicts from w with S_temp2.
                        # S_temp2 = S_temp U {v}.
                        # So we need OR(bad_mask[w][y] for y in S_temp) | bad_mask[w][v].
                        
                        # My code above:
                        # final_mask = mask_temp2
                        # for y in S_temp: final_mask |= bad_mask[w-1][y-1]
                        # final_mask |= bad_mask[w-1][v-1]
                        # This is correct.
                        
                        score = bin(final_mask).count('1')
                        
                        if best_swap is None or score < best_swap_score:
                            best_swap = (u, v, w)
                            best_swap_mask = final_mask
                            best_swap_score = score
                            
            if best_swap:
                u, v, w = best_swap
                S.remove(u)
                R.add(u)
                S.add(v)
                R.remove(v)
                S.add(w)
                R.remove(w)
                mask_S = best_swap_mask
                improved = True
                
        if not improved:
            # Try random 1-for-1 swaps to escape local optima
            if time.time() < deadline - 5.0 and S and R:
                found_swap = False
                for _ in range(5):
                    u = random.choice(list(S))
                    v = random.choice(list(R))
                    
                    S_temp = S - {u}
                    mask_temp = get_mask(S_temp)
                    if can_add(v, mask_temp):
                        S.remove(u)
                        R.add(u)
                        S.add(v)
                        R.remove(v)
                        mask_S = get_mask(S)
                        found_swap = True
                        break
                if not found_swap:
                    break

    return sorted(S)
