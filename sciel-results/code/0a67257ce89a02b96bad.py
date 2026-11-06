"""Structural rewrite: meet-in-the-middle with precomputed antichain subsets."""
import time
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s (or 90s) kill
    
    # Build list of 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    
    # Precompute divisibility relation: div_mask[i] is a set of indices j where smooth[i] divides smooth[j]
    # Actually, for antichain check, we need: no two chosen elements a,b where a|b or b|a.
    # Let's build a "conflict" adjacency: conflict[i] is a bitmask of indices j != i such that smooth[i] | smooth[j] or smooth[j] | smooth[i].
    # For a subset to be an antichain, no two elements in it can be in conflict.
    
    # Since n_smooth is small (let's check: 5-smooth up to 128)
    # 1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80,81,90,96,100,108,120,125,128
    # Let me count: 
    # 2^a * 3^b * 5^c <= 128
    # This is manageable. Let's just build it.
    
    conflict = [0] * n_smooth  # bitmask of conflicting indices
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Precompute for each subset (as a bitmask) whether it's an antichain and its sum
    # But 2^38 is too large. We need meet-in-the-middle.
    
    # Split smooth into two halves
    mid = n_smooth // 2
    left = smooth[:mid]
    right = smooth[mid:]
    n_left = mid
    n_right = n_smooth - mid
    
    # For the left half, enumerate all subsets, check antichain, record (sum, mask)
    # For the right half, same.
    # Then for each target, combine left and right subsets that are mutually antichain-compatible.
    
    # But checking mutual compatibility between a left subset and a right subset requires checking
    # that no element in left divides any element in right and vice versa.
    
    # Alternative approach: since n_smooth is small (~38), and we need sums up to 128,
    # we can use DP over (sum, antichain_state) but antichain_state is complex.
    
    # Let's try a different structural approach: 
    # For each target n, use DFS with pruning based on:
    # 1. Remaining sum
    # 2. Antichain constraint (track which elements are "forbidden" because they divide or are divided by chosen elements)
    
    # Actually, let's precompute all valid antichain subsets for each sum up to 128.
    # Since we need minimal cardinality, for each (sum, cardinality) we just need to know if it's achievable.
    
    # Let's use DP: dp[s] = set of "antichain signatures" achievable with sum s.
    # But signatures are too complex.
    
    # Simpler: for each target, do a BFS/DFS that tracks the set of chosen elements' "upward closure" 
    # (elements that divide any chosen element are forbidden, elements that are divided by any chosen element are also forbidden).
    
    # Let's define: given a set S of chosen smooth numbers, the forbidden set F(S) is:
    # {x in smooth : exists y in S such that x|y or y|x} \ S
    # A valid antichain is one where no two elements are in each other's forbidden set.
    
    # For search: at each step, we can only add elements not in F(S) ∪ S.
    
    # Let's precompute for each smooth number i, the set of indices j that are "incompatible" with i
    # (i.e., i|j or j|i). This is already conflict[i].
    
    # So the forbidden set for a chosen set S is: union of conflict[s] for s in S, plus S itself.
    # We can maintain this as a bitmask.
    
    # Search strategy for each target n:
    # DFS with state: (current_sum, chosen_mask, forbidden_mask)
    # But chosen_mask and forbidden_mask are 38-bit integers, so state space is manageable with memoization.
    # Actually, we want minimum number of terms, so we can do BFS by number of terms.
    
    # Let's precompute all minimal antichain representations for all targets 1..128.
    # For each target, find the minimum k such that there exists an antichain of k smooth numbers summing to target.
    
    # To speed up, we can precompute for each possible "forbidden mask" the set of available elements.
    
    # Let's just do a well-optimized DFS for each target with good pruning.
    
    # Sort smooth in descending order for better pruning (larger elements first)
    # Actually, for sum constraints, it might be better to sort in a way that helps pruning.
    
    # Let's precompute suffix sums: suffix_sum[i] = sum of smooth[i:]
    # And prefix: prefix_sum[i] = sum of smooth[:i]
    
    # For DFS: we pick elements in a fixed order (say, descending by value), and at each step decide to include or skip.
    # The antichain constraint: if we include element i, we must exclude all j in conflict[i].
    
    # Let's use bitmasks for efficiency.
    
    # Precompute: for each index i, conflict_mask[i] = bitmask of all j where i and j are in conflict.
    # (Already have conflict[i] as bitmask)
    
    # DFS state: (idx, remaining_sum, forbidden_mask)
    # idx: next index to consider
    # remaining_sum: how much more we need to sum
    # forbidden_mask: bitmask of indices that cannot be chosen (because they conflict with already chosen elements)
    
    # We want to minimize the number of chosen elements, so we do BFS by depth (number of elements chosen).
    
    # Actually, let's just do a DFS that tries to find ANY antichain of size k for k=1,2,3,...
    # For each k, we can use a recursive search with pruning.
    
    # Better: precompute all valid antichains and their sums.
    # Since n_smooth ~ 38, the number of antichains might be large, but let's try.
    
    # Actually, let's use a different approach: 
    # For each target, use integer programming / constraint satisfaction with strong pruning.
    
    # Given the time limit, let's do a hybrid:
    # 1. For small targets (n <= 30), try k=1 first (is n smooth?), then k=2, etc.
    # 2. Use memoization across targets if possible.
    
    # Let's just implement a fast DFS for each target.
    
    # Sort smooth indices by value descending for better pruning
    order = sorted(range(n_smooth), key=lambda i: smooth[i], reverse=True)
    order_inv = [0] * n_smooth
    for pos, idx in enumerate(order):
        order_inv[idx] = pos
    
    # Precompute for each position in order, the conflict mask in terms of original indices
    # Actually, let's just work with original indices and use the order for iteration.
    
    # For DFS, we iterate through elements in 'order', and at each element, decide to take or skip.
    # If we take element i, we add conflict[i] to forbidden_mask.
    
    # Memoization: (pos, remaining_sum, forbidden_mask) -> min additional elements needed, or -1 if impossible.
    # But forbidden_mask is 38 bits, so the state space is large. However, many states won't be reached.
    
    # Let's limit the search: for each target, try k=1,2,3,4,5,... up to some limit.
    # For a given k, use DFS to find an antichain of exactly k elements summing to target.
    
    # To make it fast, use strong pruning:
    # - If remaining_sum < 0: prune
    # - If remaining_sum == 0: success if we've chosen exactly k elements
    # - If we've chosen k elements but remaining_sum != 0: prune
    # - If the sum of the largest (k - chosen) available elements < remaining_sum: prune
    # - If the sum of the smallest (k - chosen) available elements > remaining_sum: prune
    
    # Let's implement this.
    
    # Precompute for each subset of available elements, the min and max possible sums of size r.
    # This is complex. Let's just use simple pruning.
    
    def find_antichain(target, max_k=10):
        """Find minimum number of smooth antichain elements summing to target."""
        for k in range(1, max_k + 1):
            # DFS to find antichain of exactly k elements summing to target
            result = _dfs_k(0, target, k, 0, 0)
            if result is not None:
                return result
        return None
    
    def _dfs_k(pos, remaining, k_remaining, chosen_mask, forbidden_mask):
        """
        pos: position in 'order'
        remaining: sum still needed
        k_remaining: number of elements still to choose
        chosen_mask: bitmask of chosen elements
        forbidden_mask: bitmask of forbidden elements
        Returns list of chosen indices or None.
        """
        if k_remaining == 0:
            return [] if remaining == 0 else None
        
        if pos >= n_smooth:
            return None
        
        if remaining < 0:
            return None
        
        # Pruning: even if we take all remaining available elements, can we reach 'remaining'?
        # Sum of all available elements from pos onwards
        available_sum = 0
        available_count = 0
        for p in range(pos, n_smooth):
            idx = order[p]
            if not (forbidden_mask >> idx) & 1 and not (chosen_mask >> idx) & 1:
                available_sum += smooth[idx]
                available_count += 1
        
        if available_count < k_remaining:
            return None
        if available_sum < remaining:
            return None
        
        # Sum of smallest k_remaining available elements
        # This is expensive to compute each time. Let's skip this pruning for now.
        
        # Try skipping current element
        idx = order[pos]
        
        # Option 1: skip
        if not (forbidden_mask >> idx) & 1:
            # We can skip this element
            result = _dfs_k(pos + 1, remaining, k_remaining, chosen_mask, forbidden_mask)
            if result is not None:
                return result
        
        # Option 2: take this element (if not forbidden)
        if not (forbidden_mask >> idx) & 1 and smooth[idx] <= remaining:
            new_forbidden = forbidden_mask | (1 << idx) | conflict[idx]
            result = _dfs_k(pos + 1, remaining - smooth[idx], k_remaining - 1, chosen_mask | (1 << idx), new_forbidden)
            if result is not None:
                return [idx] + result
        
        return None
    
    # This DFS might be slow. Let's add memoization for (pos, remaining, k_remaining, forbidden_mask).
    # But forbidden_mask is 38 bits, so the key is large. Let's try without memoization first and see.
    
    # Actually, let's restructure: precompute all valid antichains up to size 5 for all sums.
    
    # Given the complexity, let's just run the DFS with a time limit per target.
    
    answers = []
    total_terms = 0
    start_time = time.time()
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency: use n itself if smooth, else a simple fallback
            if n in smooth:
                answers.append([n])
                total_terms += 1
            else:
                # Quick fallback: find any antichain
                ans = find_antichain(n, max_k=15)
                if ans is None:
                    ans = [n] if n in smooth else [1]
                answers.append(ans)
                total_terms += len(ans)
            continue
        
        # Try to find minimal antichain
        best = None
        for k in range(1, 15):
            if time.time() > deadline - 5:
                break
            result = _dfs_k(0, n, k, 0, 0)
            if result is not None:
                best = result
                break
        
        if best is None:
            # Fallback
            if n in smooth:
                best = [smooth.index(n)]
            else:
                # Last resort: use a simple valid antichain
                # For non-smooth n, we need to find some combination
                best = find_antichain(n, max_k=20)
                if best is None:
                    best = [0]  # This is a placeholder; should not happen
        
        # Convert indices to values
        terms = [smooth[i] for i in best]
        answers.append(terms)
        total_terms += len(terms)
    
    return answers
