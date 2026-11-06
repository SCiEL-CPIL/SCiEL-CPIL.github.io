import random
import time
import sys

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Precompute 3-APs and 4-APs
    # indices 0 to 255 corresponding to 1 to 256
    
    # 3-APs: (i, j, k) with i < j < k and j-i == k-j
    three_aps = []
    three_aps_with = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 2 * d):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            three_aps_with[a].append((b, c))
            three_aps_with[b].append((a, c))
            three_aps_with[c].append((a, b))
            
    # 4-APs: (i, j, k, l) with i < j < k < l and j-i == k-j == l-k
    four_aps = []
    four_aps_with = [[] for _ in range(N)]
    for d in range(1, N // 4 + 1):
        for a in range(0, N - 3 * d):
            b = a + d
            c = a + 2 * d
            ll = a + 3 * d
            four_aps.append((a, b, c, ll))
            four_aps_with[a].append((b, c, ll))
            four_aps_with[b].append((a, c, ll))
            four_aps_with[c].append((a, b, ll))
            four_aps_with[ll].append((a, b, c))
            
    def count_3aps(A):
        count = 0
        for a, b, c in three_aps:
            if a in A and b in A and c in A:
                count += 1
        return count
        
    def has_4ap(A):
        for a, b, c, d in four_aps:
            if a in A and b in A and c in A and d in A:
                return True
        return False
        
    # Initialize A
    # Try a few random starts
    best_A = None
    best_score = -1
    
    num_starts = 10
    time_per_start = time_limit / num_starts
    
    for s in range(num_starts):
        # Random initialization
        # Density around 0.2-0.3?
        # Let's try size 50
        size = random.randint(30, 60)
        A = set(random.sample(range(N), size))
        
        # Repair: Remove elements from 4-APs
        # We can just iterate and remove
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if a in A and b in A and c in A and d in A:
                    # Remove a random one
                    to_remove = random.choice([a, b, c, d])
                    A.remove(to_remove)
                    changed = True
                    if time.time() - start_time > time_limit - 5:
                        break
            if time.time() - start_time > time_limit - 5:
                break
        
        # Local Search
        T = 5.0  # Temperature
        current_score = count_3aps(A)
        
        # If current is better, update best
        if current_score > best_score:
            best_score = current_score
            best_A = set(A)
            
        while time.time() - start_time < time_limit - 5:
            # Pick a move
            # 50% Add, 50% Swap
            if random.random() < 0.5:
                # Add
                if len(A) == N:
                    continue
                candidates = [x for x in range(N) if x not in A]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Check 4-AP violation
                valid = True
                for u, v, w in four_aps_with[x]:
                    if u in A and v in A and w in A:
                        valid = False
                        break
                
                if valid:
                    # Calculate gain
                    gain = 0
                    for u, v in three_aps_with[x]:
                        if u in A and v in A:
                            gain += 1
                    
                    if gain > 0 or random.random() < pow(2.718, gain / T):
                        A.add(x)
                        current_score += gain
                        if current_score > best_score:
                            best_score = current_score
                            best_A = set(A)
            else:
                # Swap
                if not A:
                    continue
                x = random.choice(list(A))
                candidates = [y for y in range(N) if y not in A]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Temporarily remove x
                A.remove(x)
                
                # Check 4-AP violation for y
                valid = True
                for u, v, w in four_aps_with[y]:
                    if u in A and v in A and w in A:
                        valid = False
                        break
                
                if valid:
                    # Calculate gain
                    # Gain from adding y
                    gain_y = 0
                    for u, v in three_aps_with[y]:
                        if u in A and v in A:
                            gain_y += 1
                            
                    # Loss from removing x
                    loss_x = 0
                    for u, v in three_aps_with[x]:
                        if u in A and v in A:
                            loss_x += 1
                            
                    gain = gain_y - loss_x
                    
                    if gain > 0 or random.random() < pow(2.718, gain / T):
                        A.add(y)
                        current_score += gain
                        if current_score > best_score:
                            best_score = current_score
                            best_A = set(A)
                    else:
                        # Revert
                        A.add(x)
                else:
                    # Revert
                    A.add(x)
                    
            # Cool down temperature slightly
            T *= 0.999
            
    # Return best_A as sorted list of 1-based indices
    if best_A is None:
        return [1]
        
    return sorted([x + 1 for x in best_A])
