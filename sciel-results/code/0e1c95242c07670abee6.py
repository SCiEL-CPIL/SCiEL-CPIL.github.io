import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256
    
    # Precompute bit masks for elements 1..N
    # Element i corresponds to bit (i-1)
    bits = [1 << i for i in range(N)]
    
    # Precompute 4-AP conflicts
    # For each x, list of masks (3 bits set) such that if (A & mask) == mask, x creates a 4-AP
    four_ap_masks = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            p1, p2, p3, p4 = a, a + d, a + 2 * d, a + 3 * d
            # For each element in the 4-AP, the other 3 form a conflict mask
            # x=p1: mask = bits[p2-1] | bits[p3-1] | bits[p4-1]
            m = bits[p2-1] | bits[p3-1] | bits[p4-1]
            four_ap_masks[p1].append(m)
            
            m = bits[p1-1] | bits[p3-1] | bits[p4-1]
            four_ap_masks[p2].append(m)
            
            m = bits[p1-1] | bits[p2-1] | bits[p4-1]
            four_ap_masks[p3].append(m)
            
            m = bits[p1-1] | bits[p2-1] | bits[p3-1]
            four_ap_masks[p4].append(m)
            
    # Precompute 3-AP gains
    # For each x, list of pairs (u, v) such that {x, u, v} is a 3-AP
    # We store indices 0..N-1 for bit access
    three_ap_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2):
            b = (a + c) // 2
            # 3-AP: a, b, c
            # x=a: pair (b, c)
            three_ap_pairs[a].append((b - 1, c - 1))
            # x=b: pair (a, c)
            three_ap_pairs[b].append((a - 1, c - 1))
            # x=c: pair (a, b)
            three_ap_pairs[c].append((a - 1, b - 1))

    def is_valid(x, A_mask):
        for m in four_ap_masks[x]:
            if (A_mask & m) == m:
                return False
        return True

    def count_gain(x, A_mask):
        gain = 0
        for u_idx, v_idx in three_ap_pairs[x]:
            if (A_mask & bits[u_idx]) and (A_mask & bits[v_idx]):
                gain += 1
        return gain

    def total_score(A_mask):
        # Count 3-APs
        # Sum of gains / 3 ? No, each 3-AP is counted 3 times (once for each element)
        # So total = sum(gain(x) for x in A) / 3
        # But we can just calculate it directly or maintain it incrementally.
        # For simplicity in local search, we'll maintain current_score.
        # This function is for initial calculation or verification.
        count = 0
        for x in range(1, N + 1):
            if A_mask & bits[x - 1]:
                count += count_gain(x, A_mask)
        return count // 3

    A_mask = 0
    current_score = 0
    remaining = set(range(1, N + 1))
    
    # Greedy Phase
    while remaining and time.time() < deadline - 30:
        best_x = -1
        best_score = -1
        # To speed up, we might want to sample, but N=256 is small enough to scan all
        for x in remaining:
            if is_valid(x, A_mask):
                gain = count_gain(x, A_mask)
                # Heuristic: prefer higher gain. Tie-breaker: random or centrality?
                # Let's add a small random noise to break ties and avoid local minima
                score = gain + random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_x = x
        
        if best_x == -1:
            break
        
        A_mask |= bits[best_x - 1]
        current_score += count_gain(best_x, A_mask)
        # Note: current_score is sum of gains, not divided by 3 yet.
        # We want to maximize sum of gains, which is proportional to number of 3-APs.
        remaining.remove(best_x)
        
    # Local Search Phase
    # 1-swap: remove u from A, add v from remaining
    # Change in score = -gain(u, A\{u}) + gain(v, A\{u})
    # Note: gain(u, A) includes contributions from pairs in A.
    # When we remove u, the gain of u becomes 0 (since u is not in A).
    # The gain of v is calculated w.r.t A\{u}.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A_list = [x for x in range(1, N + 1) if A_mask & bits[x - 1]]
        R_list = list(remaining)
        
        if not A_list or not R_list:
            break
            
        # Precompute gains for A elements w.r.t A (to know what we lose)
        # Actually, loss of u is the number of 3-APs containing u and 2 other elements in A.
        # This is exactly count_gain(u, A) because the pairs are (a,b) in A.
        # Wait, count_gain(u, A) counts pairs (a,b) in A such that {u,a,b} is 3-AP.
        # Yes, this is the number of 3-APs completed by u.
        
        # We need to be careful: if we remove u, the "gain" of u is lost.
        # But also, u might have been part of 3-APs that are still valid? No, 3-APs are defined by the set.
        # If u is removed, any 3-AP containing u is gone.
        
        # So delta_score = -count_gain(u, A) + count_gain(v, A \ {u})
        
        # To speed up, we can precompute count_gain for all v in R w.r.t A
        # And for all u in A w.r.t A
        
        gains_R = {}
        for v in R_list:
            if is_valid(v, A_mask):
                gains_R[v] = count_gain(v, A_mask)
            else:
                gains_R[v] = -1
                
        # Try swaps
        # Sample to avoid O(N^2) if N is large, but N=256 is small.
        # Let's try all u in A and top K v in R.
        
        # Sort R by gain
        sorted_R = sorted([v for v in R_list if gains_R[v] >= 0], key=lambda x: gains_R[x], reverse=True)
        
        for u in A_list:
            if time.time() > deadline:
                break
                
            u_gain = count_gain(u, A_mask)
            
            # Temporarily remove u
            A_mask_temp = A_mask & ~bits[u - 1]
            
            for v in sorted_R[:20]: # Top 20 candidates
                if time.time() > deadline:
                    break
                    
                # v must be valid in A_mask_temp
                if not is_valid(v, A_mask_temp):
                    continue
                
                v_gain_temp = count_gain(v, A_mask_temp)
                
                new_score = current_score - u_gain + v_gain_temp
                
                if new_score > current_score:
                    # Accept
                    A_mask = A_mask_temp | bits[v - 1]
                    current_score = new_score
                    remaining.remove(v)
                    improved = True
                    break
            
            # Restore u if we didn't accept a swap involving u?
            # Wait, the loop above modifies A_mask_temp locally.
            # If we accept, we update A_mask.
            # If we don't accept, A_mask is unchanged.
            # So we don't need to restore A_mask explicitly if we only modify A_mask on acceptance.
            
        # 2-swap moves (optional, if time permits)
        if not improved and time.time() < deadline - 5:
            # Remove u1, u2; add v1, v2
            # This is more complex. Skip for now or implement simply.
            # Given time, 1-swap might be sufficient with the fast bitset ops.
            pass
            
    # Convert back to list
    result = [x for x in range(1, N + 1) if A_mask & bits[x - 1]]
    return result
