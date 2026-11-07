import math
import time
from collections import defaultdict

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # Generate events
    events = []
    for i, d in enumerate(divisors):
        for m in range(0, L, d):
            events.append((m, i))
            
    events.sort(key=lambda x: x[0])
    
    # Duplicate for circular handling
    extended_events = events + [(v + L, i) for v, i in events]
    N_ext = len(extended_events)
    
    # Find all minimal covering intervals
    # A window [l, r] is minimal covering if it covers all divisors and 
    # either l is the first occurrence of its divisor in the window 
    # or r is the last occurrence of its divisor in the window?
    # Actually, standard sliding window for "minimum size window containing all chars"
    # finds minimal windows in terms of indices.
    # We want minimal windows in terms of value span?
    # No, we want to check distinct assignment.
    # Let's just collect all windows where unique_count == n.
    # But there are O(N) such windows in a sliding pass?
    # No, for each right, we shrink left as much as possible.
    # The window [left, right] is minimal with respect to left.
    # It might not be minimal with respect to right (i.e. we could shrink right?).
    # But since we process right increasing, and left increasing, 
    # the window [left, right] is the smallest window ending at right that covers all.
    # So these are candidates.
    
    best_span = float('inf')
    best_result = None
    
    counts = [0] * n
    unique_count = 0
    left = 0
    
    # We also need to handle the case where the optimal window wraps around.
    # The extended events handle this.
    # We only need to consider windows starting with left < N (original length) 
    # to avoid duplicates?
    # Or just take the first one found with min span?
    
    # To speed up, we can pre-calculate for each position, the list of values for each divisor?
    # No, we maintain it dynamically.
    
    vals_by_idx = defaultdict(list)
    # We need to be able to remove from the front of lists?
    # Since left moves forward, we remove events from the left.
    # We can use a deque or just a list and pop(0) is O(N).
    # Given window size is small, list is fine.
    
    # However, maintaining vals_by_idx dynamically in the sliding window is tricky 
    # because we need to remove the specific value at `left`.
    
    # Let's stick to the previous structure but optimize the inner check.
    # Instead of scanning the window every time, 
    # we can just check the window when `unique_count == n`.
    # The window size is small.
    
    # Optimization: 
    # 1. Don't run DFS if the window span is already >= best_span.
    # 2. Use MRV in DFS.
    # 3. Use a global used set.
    
    for right in range(N_ext):
        val_r, idx_r = extended_events[right]
        counts[idx_r] += 1
        if counts[idx_r] == 1:
            unique_count += 1
            vals_by_idx[idx_r].append(val_r)
        else:
            vals_by_idx[idx_r].append(val_r)
            
        while unique_count == n:
            val_l, idx_l = extended_events[left]
            span = val_r - val_l
            
            # If current span is not better, we can skip DFS?
            # But we need to find the min span.
            # If span >= best_span, we can try to shrink further?
            # Shrinking reduces span. So we should continue shrinking.
            # We only run DFS if we think this window might be optimal.
            # Actually, if span >= best_span, we don't need to check this specific window 
            # unless we can't shrink further?
            # No, if we can shrink, the new span is smaller.
            # So we only need to check the *tightest* window for this right?
            # The while loop shrinks left until unique_count < n.
            # So the last window with unique_count == n before it breaks is the tightest for this right.
            # But we enter the while loop immediately if unique_count == n.
            # So we check [left, right]. Then we shrink left.
            # If it's still n, we check [left+1, right].
            # So we check all minimal windows ending at right?
            # No, we check [left, right], then remove left. 
            # If still n, we check [left+1, right].
            # So yes, we check all windows ending at right that are minimal with respect to left.
            
            if span < best_span:
                # Try to find distinct assignment
                # Build local lists
                # Since we maintain vals_by_idx, we can use it directly?
                # But vals_by_idx contains values for the *current* window [left, right].
                # Yes.
                
                # MRV: sort indices by len(vals_by_idx[i])
                order = sorted(range(n), key=lambda i: len(vals_by_idx[i]))
                
                used = set()
                
                def dfs(idx_pos):
                    if idx_pos == n:
                        return True
                    i = order[idx_pos]
                    for v in vals_by_idx[i]:
                        if v not in used:
                            used.add(v)
                            if dfs(idx_pos + 1):
                                return True
                            used.remove(v)
                    return False
                
                if dfs(0):
                    # Extract result
                    res = [0]*n
                    used2 = set()
                    # We need to retrieve the actual values chosen.
                    # The DFS above didn't store them.
                    # Let's modify DFS to store path.
                    
                    path = {}
                    def dfs_path(idx_pos):
                        if idx_pos == n:
                            return True
                        i = order[idx_pos]
                        for v in vals_by_idx[i]:
                            if v not in used:
                                path[i] = v
                                used.add(v)
                                if dfs_path(idx_pos + 1):
                                    return True
                                used.remove(v)
                                del path[i]
                        return False
                    
                    if dfs_path(0):
                        best_span = span
                        best_result = [path[i] for i in range(n)]
                        
            # Shrink left
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                unique_count -= 1
                # Remove from vals_by_idx
                # We need to remove val_l from vals_by_idx[idx_l]
                # It's the first one?
                # The events are sorted by value.
                # left moves forward, so val_l is the smallest in the window.
                # So it should be at the front of the list for idx_l?
                # Yes, because we added them in order.
                vals_by_idx[idx_l].pop(0) # O(K) where K is count, small
            else:
                vals_by_idx[idx_l].pop(0)
                
            left += 1
            
            # Time check?
            # If right > N_ext/2 and we have a good result, we might stop?
            # The pattern repeats every L.
            # So we only need to check left < N (original length).
            # If left >= N, we are in the duplicated part.
            # The window [left, right] corresponds to a window in the original circle.
            # If we found a solution, is it better to stop?
            # The span is periodic.
            # So checking left in [0, N) is sufficient.
            if left >= N_ext // 2: # N_ext is 2 * len(events)
                 # We have covered the whole circle twice?
                 # Actually, extended_events has 2 * total_events.
                 # We just need to ensure we've seen all starting points.
                 # Starting points are indices 0 to total_events-1.
                 # So if left >= total_events, we are done.
                 break

    if best_result is not None:
        return [int(v) for v in best_result]
    
    # Fallback
    return divisors[:]

if __name__ == '__main__':
    res = construct()
    print(res)
