import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    max_sum = 2 * N

    def greedy(order):
        counts = [0] * (max_sum + 1)
        in_set = [False] * (N + 1)
        S = []
        
        for y in order:
            if in_set[y]: continue
            # Check validity
            valid = True
            if counts[2 * y] == 2:
                valid = False
            else:
                for x in S:
                    if counts[x + y] == 2:
                        valid = False
                        break
            if valid:
                for x in S:
                    counts[x + y] += 1
                counts[2 * y] += 1
                in_set[y] = True
                S.append(y)
                
        return S

    def local_search(S_init, time_limit):
        # Reconstruct state
        counts = [0] * (max_sum + 1)
        in_set = [False] * (N + 1)
        S = list(S_init)
        S_set = set(S)
        
        # Build counts
        # Sort S for consistent counting? Not strictly needed for counts, but good for debug.
        # Just iterate all pairs.
        # O(|S|^2)
        for i in range(len(S)):
            x = S[i]
            counts[2*x] += 1
            for j in range(i):
                y = S[j]
                counts[x+y] += 1
        for x in S:
            in_set[x] = True
            
        best_size = len(S)
        best_S = list(S)
        
        # Local search loop
        while time.time() < time_limit:
            # Try to add
            added = False
            # Get candidates
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if not candidates:
                break
            random.shuffle(candidates)
            
            for y in candidates:
                valid = True
                if counts[2 * y] == 2:
                    valid = False
                else:
                    for x in S:
                        if counts[x + y] == 2:
                            valid = False
                            break
                if valid:
                    # Add
                    for x in S:
                        counts[x + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    S_set.add(y)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
            
            if added:
                continue
                
            # Try swap
            # Sample 10 swaps
            for _ in range(10):
                if not S or not candidates:
                    break
                x = random.choice(S)
                y = random.choice(candidates)
                
                # Check if swap is valid
                # We need to check if S' = S - {x} + {y} is valid.
                
                # Remove x
                S.remove(x)
                S_set.remove(x)
                # Update counts
                for z in S:
                    counts[x + z] -= 1
                counts[2 * x] -= 1
                in_set[x] = False
                
                # Check y
                valid = True
                if counts[2 * y] == 2:
                    valid = False
                else:
                    for z in S:
                        if counts[z + y] == 2:
                            valid = False
                            break
                
                if valid:
                    # Accept
                    for z in S:
                        counts[z + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    S_set.add(y)
                    
                    # Update candidates
                    # y is no longer candidate, x is now candidate
                    # We will regenerate candidates next loop, so no need to update list here
                    # But we should break to try additions
                    break
                else:
                    # Revert
                    S.append(x)
                    S_set.add(x)
                    # Re-add x's contributions
                    for z in S:
                        if z != x:
                            counts[x + z] += 1
                    counts[2 * x] += 1
                    in_set[x] = True

        return best_S

    best_S = []
    
    # 1. Ascending
    S1 = greedy(range(1, N + 1))
    if len(S1) > len(best_S): best_S = S1
    
    # 2. Descending
    S2 = greedy(range(N, 0, -1))
    if len(S2) > len(best_S): best_S = S2
    
    # 3. Random
    for _ in range(50):
        if time.time() > deadline - 20: break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > len(best_S):
            best_S = S_rand

    # 4. Local Search on Best
    if time.time() < deadline - 5:
        best_S = local_search(best_S, deadline - 2)
        
    return best_S

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
