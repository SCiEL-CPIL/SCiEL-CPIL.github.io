import random
from math import gcd
import time

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad_mask[x][y]
    # bad_mask[x][y] is a bitmask of z such that (x, y, z) is a bad triple.
    # (x, y, z) is bad if gcd(x,y) == gcd(y,z) == gcd(z,x).
    # Note: The triple is distinct, so z != x and z != y.
    bad_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            g_xy = gcd_table[x][y]
            mask = 0
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if gcd_table[y][z] == g_xy and gcd_table[z][x] == g_xy:
                    mask |= (1 << z)
            bad_mask[x][y] = mask
            bad_mask[y][x] = mask

    def greedy_build():
        S = 0  # bitmask
        in_S = [False] * (N + 1)
        
        # We want to pick elements that don't create conflicts.
        # Start with all candidates.
        # At each step, pick a candidate that is currently "safe" (no conflict with S).
        # If none are safe, we are stuck? No, we can always add elements one by one if we check.
        # But if S is not empty, adding x requires no bad triple with pairs in S.
        
        # Greedy: pick element with min "potential conflict"?
        # Or just random order?
        
        # Let's try a greedy that picks elements with fewest bad triples overall first?
        # Or most?
        
        # Let's just do a simple greedy:
        # Candidates = all numbers.
        # While candidates:
        #   Pick x from candidates such that it doesn't conflict with S.
        #   If none, break.
        #   Add x to S.
        #   Remove x from candidates.
        
        # This might get stuck early.
        # Better: Sort by degree (number of bad triples involving x).
        # Pick low degree first?
        
        degrees = [0] * (N + 1)
        for x in range(1, N + 1):
            for y in range(x + 1, N + 1):
                if bad_mask[x][y]:
                    # Count how many z are in bad_mask[x][y]
                    # This is expensive to count bits repeatedly.
                    # Just use a proxy: number of y such that bad_mask[x][y] is non-zero.
                    degrees[x] += 1
                    degrees[y] += 1
        
        # Actually, let's just use random restarts with a simple greedy.
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added to S
            can_add = True
            # Check against all pairs in S?
            # Using the bitmask logic:
            # x is safe if for all y in S, (bad_mask[x][y] & S) == 0.
            # We can iterate through bits of S.
            s_temp = S
            while s_temp:
                lsb = s_temp & -s_temp
                y = lsb.bit_length() - 1
                if bad_mask[x][y] & S:
                    can_add = False
                    break
                s_temp ^= lsb
            
            if can_add:
                S |= (1 << x)
                in_S[x] = True
        
        return S

    def local_search(S):
        in_S = [False] * (N + 1)
        s_temp = S
        while s_temp:
            lsb = s_temp & -s_temp
            in_S[lsb.bit_length() - 1] = True
            s_temp ^= lsb
        
        # Precompute blockers for each x not in S
        # blockers[x] = number of y in S such that (bad_mask[x][y] & S) != 0
        blockers = [0] * (N + 1)
        for x in range(1, N + 1):
            if in_S[x]:
                continue
            count = 0
            s_temp = S
            while s_temp:
                lsb = s_temp & -s_temp
                y = lsb.bit_length() - 1
                if bad_mask[x][y] & S:
                    count += 1
                s_temp ^= lsb
            blockers[x] = count
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add
            # Find x not in S with blockers[x] == 0
            addable = [x for x in range(1, N + 1) if not in_S[x] and blockers[x] == 0]
            if addable:
                random.shuffle(addable)
                for x in addable:
                    # Add x
                    S |= (1 << x)
                    in_S[x] = True
                    
                    # Update blockers for all z not in S
                    # For each z, if (bad_mask[z][x] & (S without x)) != 0, then x blocks z.
                    # Wait, the condition for x to block z is:
                    # There exists y in S (including x? No, y in S_old) such that (z, x, y) is bad?
                    # No, the blocker definition:
                    # x blocks z if there is y in S such that (z, x, y) is bad?
                    # My previous derivation:
                    # z is blocked by x if (bad_mask[z][x] & S_old) != 0.
                    # Let's re-verify.
                    # z is addable if no pair (u,v) in S forms bad triple with z.
                    # If we add x to S, S becomes S_new.
                    # z is blocked in S_new if there is a pair (u,v) in S_new forming bad triple with z.
                    # One such pair could be (x, u) where u in S_old.
                    # So z is blocked by x if there exists u in S_old such that (z, x, u) is bad.
                    # This is equivalent to (bad_mask[z][x] & S_old) != 0.
                    
                    # So, for each z not in S_new:
                    # if (bad_mask[z][x] & (S ^ (1<<x))) != 0:
                    #    blockers[z] += 1
                    
                    S_old = S ^ (1 << x)
                    for z in range(1, N + 1):
                        if in_S[z]:
                            continue
                        if bad_mask[z][x] & S_old:
                            blockers[z] += 1
                    
                    improved = True
                    break
            
            # 2. Try swaps (remove one, add one)
            if not improved:
                # Try removing each y in S
                s_list = [i for i in range(1, N + 1) if in_S[i]]
                random.shuffle(s_list)
                for y in s_list:
                    # Remove y
                    S_new = S ^ (1 << y)
                    in_S[y] = False
                    
                    # Update blockers for all z not in S_new
                    # We need to recalculate or update from old blockers.
                    # This is complex. Let's just recalculate blockers for all z not in S_new.
                    # N=128, so O(N^2) is fine.
                    
                    new_blockers = [0] * (N + 1)
                    for z in range(1, N + 1):
                        if in_S[z]:
                            continue
                        count = 0
                        s_temp = S_new
                        while s_temp:
                            lsb = s_temp & -s_temp
                            u = lsb.bit_length() - 1
                            if bad_mask[z][u] & S_new:
                                count += 1
                            s_temp ^= lsb
                        new_blockers[z] = count
                    
                    # Try to add any z with new_blockers[z] == 0
                    addable = [z for z in range(1, N + 1) if not in_S[z] and new_blockers[z] == 0]
                    if addable:
                        random.shuffle(addable)
                        for z in addable:
                            # Check if adding z increases size?
                            # We removed y and added z. Size is same.
                            # We want size to increase.
                            # So we need to add MORE than we removed.
                            # This simple swap doesn't increase size.
                            # We should try to add z, and THEN try to add more.
                            
                            # Let's just accept the swap if it allows further additions?
                            # Or just do a 2-for-3?
                            
                            # For now, let's just try to add z.
                            S_final = S_new | (1 << z)
                            in_S[z] = True
                            
                            # Recalculate blockers for next iteration
                            blockers = [0] * (N + 1)
                            for w in range(1, N + 1):
                                if in_S[w]:
                                    continue
                                count = 0
                                s_temp = S_final
                                while s_temp:
                                    lsb = s_temp & -s_temp
                                    u = lsb.bit_length() - 1
                                    if bad_mask[w][u] & S_final:
                                        count += 1
                                    s_temp ^= lsb
                                blockers[w] = count
                            
                            S = S_final
                            improved = True
                            break
                            
                    if improved:
                        break
                    # Revert y if not improved
                    S = S_new | (1 << y)
                    in_S[y] = True
                    # Recalculate blockers? 
                    # Since we reverted, blockers should be back to original.
                    # But we didn't store them. 
                    # To be safe, recalculate.
                    blockers = [0] * (N + 1)
                    for z in range(1, N + 1):
                        if in_S[z]:
                            continue
                        count = 0
                        s_temp = S
                        while s_temp:
                            lsb = s_temp & -s_temp
                            u = lsb.bit_length() - 1
                            if bad_mask[z][u] & S:
                                count += 1
                        s_temp ^= lsb
                        blockers[z] = count

        return S

    best_S = 0
    
    # Run multiple iterations
    while time.time() < deadline:
        S = greedy_build()
        S = local_search(S)
        if bin(S).count('1') > bin(best_S).count('1'):
            best_S = S
            
    return [i for i in range(1, N + 1) if (best_S >> i) & 1]

if __name__ == "__main__":
    res = construct()
    print(len(res))
