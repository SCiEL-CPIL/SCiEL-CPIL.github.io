import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute 4-AP neighbors: for each x, list of 4-AP indices containing x
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            four_ap_neighbors[x].append(idx)
            
    # Precompute 3-AP neighbors: for each x, list of 3-AP indices containing x
    three_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            three_ap_neighbors[x].append(idx)
            
    # Helper: count 3-APs in a set (using bitsets for speed)
    def count_3aps_bitset(S_bits):
        count = 0
        for a, b, c in three_ap_tuples:
            if (S_bits >> (a-1)) & 1 and (S_bits >> (b-1)) & 1 and (S_bits >> (c-1)) & 1:
                count += 1
        return count
        
    # Helper: check if adding x to S_bits creates a 4-AP
    # Returns True if valid (no new 4-AP created)
    def is_valid_addition(S_bits, x):
        for idx in four_ap_neighbors[x]:
            a, b, c, d = four_ap_tuples[idx]
            # Check if the other 3 elements are in S_bits
            # Note: x is not in S_bits yet, so we just check the others
            others = [y for y in (a, b, c, d) if y != x]
            valid = True
            for y in others:
                if not ((S_bits >> (y-1)) & 1):
                    valid = False
                    break
            if valid:
                return False
        return True
        
    # Helper: check if removing x from S_bits is safe (always safe for validity, but we need to know if it breaks existing 4-APs? No, removing never creates 4-APs)
    # So removal is always valid.
    
    # Local Search Function
    def local_search(initial_bits):
        S_bits = initial_bits
        current_score = count_3aps_bitset(S_bits)
        
        # We will perform multiple passes of greedy improvement
        # Pass 1: Add best element
        # Pass 2: Remove worst element (if it reduces 4-AP risk? No, we only care about 3-AP count and validity)
        # Actually, since we start from a valid set, we only need to ensure we don't add invalid elements.
        # We can also try swaps: remove x, add y.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements
            for x in range(1, N + 1):
                if not ((S_bits >> (x-1)) & 1):
                    if is_valid_addition(S_bits, x):
                        # Calculate delta
                        new_bits = S_bits | (1 << (x-1))
                        new_score = count_3aps_bitset(new_bits)
                        if new_score > current_score:
                            S_bits = new_bits
                            current_score = new_score
                            improved = True
                            
            # Try removing elements (only if it allows adding something better? 
            # Direct removal reduces score, so we only do it as part of a swap or if stuck.
            # For now, let's just do additions in a pass, then swaps.)
            
            # Try swaps: remove x, add y
            # To be efficient, we only consider swaps if we find a y that adds more 3-APs than x removes.
            # This is O(N^2) per pass, which is 256^2 = 65k checks. 
            # Each check involves count_3aps (O(3-APs)) which is slow.
            # Let's optimize delta calculation for swaps.
            
            # For each x in S, calculate 3-APs lost if x removed.
            # For each y not in S, calculate 3-APs gained if y added (assuming x removed).
            
            # This is complex. Let's stick to simple addition passes first.
            # If no additions improve, try a single swap pass.
            
            # Swap Pass
            best_swap_delta = 0
            best_swap_x = -1
            best_swap_y = -1
            
            # Optimization: Precompute 3-AP contribution of each element in current S?
            # contrib[x] = number of 3-APs in S that contain x.
            # If we remove x, we lose contrib[x] 3-APs? No, only those that don't contain other elements?
            # Actually, a 3-AP is counted if ALL 3 are in S.
            # If we remove x, we lose all 3-APs that contain x.
            
            lost_3aps = {}
            for x in range(1, N + 1):
                if (S_bits >> (x-1)) & 1:
                    cnt = 0
                    for idx in three_ap_neighbors[x]:
                        a, b, c = three_ap_tuples[idx]
                        if ((S_bits >> (a-1)) & 1) and ((S_bits >> (b-1)) & 1) and ((S_bits >> (c-1)) & 1):
                            cnt += 1
                    lost_3aps[x] = cnt
            
            # For each y not in S, calculate gained 3-APs if added.
            # gained_3aps[y] = number of 3-APs containing y where the other 2 are in S.
            gained_3aps = {}
            for y in range(1, N + 1):
                if not ((S_bits >> (y-1)) & 1):
                    cnt = 0
                    for idx in three_ap_neighbors[y]:
                        a, b, c = three_ap_tuples[idx]
                        others = [z for z in (a, b, c) if z != y]
                        if ((S_bits >> (others[0]-1)) & 1) and ((S_bits >> (others[1]-1)) & 1):
                            cnt += 1
                    gained_3aps[y] = cnt
            
            # Now, for a swap (remove x, add y):
            # Delta = gained_3aps[y] - lost_3aps[x] + interaction?
            # Interaction: 3-APs that contain BOTH x and y? 
            # If a 3-AP contains both x and y, it is currently NOT counted (since y not in S).
            # If we remove x and add y, it is STILL not counted (since x removed).
            # So no gain from those.
            # What about 3-APs containing x but not y? Lost.
            # What about 3-APs containing y but not x? Gained.
            # So Delta = gained_3aps[y] - lost_3aps[x] is correct for the net change in 3-AP count, 
            # PROVIDED that the set S - {x} + {y} is valid (no 4-APs).
            
            # We need to check validity for each candidate swap.
            # To speed up, we can pre-filter y: is_valid_addition(S_bits ^ (1 << (x-1)), y)
            
            # This is still expensive. Let's just do a limited number of random swaps if time permits.
            # Or, simpler: just do multiple addition passes from different starting points.
            
            break # Exit while loop after one pass of additions to avoid infinite loop if no improvement
            
        return S_bits, current_score

    # Multi-Start Strategy
    best_bits = 0
    best_score = -1
    
    # Seed 1: The original greedy
    S_bits = 0
    for x in range(1, N + 1):
        if is_valid_addition(S_bits, x):
            S_bits |= (1 << (x-1))
    
    bits, score = local_search(S_bits)
    if score > best_score:
        best_score = score
        best_bits = bits
        
    # Seed 2-10: Random permutations
    for i in range(10):
        if time.time() > deadline - 5:
            break
            
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        S_bits = 0
        for x in perm:
            if is_valid_addition(S_bits, x):
                S_bits |= (1 << (x-1))
                
        bits, score = local_search(S_bits)
        if score > best_score:
            best_score = score
            best_bits = bits
            
    # Convert best_bits to list
    result = []
    for i in range(1, N + 1):
        if (best_bits >> (i-1)) & 1:
            result.append(i)
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
